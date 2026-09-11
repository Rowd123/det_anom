# trainer.py

from collections import deque
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import pyro
import torch
from torch import Tensor, nn

from utils import BatchSlidingWindow, MinMaxScaler, validate_checkpoint_model_dimension


class SigmaEpochStatistics:
    """Agrège les sigma d'une époque avec un échantillon mémoire borné."""

    _QUANTILES = {
        "p01": 0.01,
        "p10": 0.10,
        "median": 0.50,
        "p90": 0.90,
        "p99": 0.99,
    }

    def __init__(
        self,
        near_floor_threshold: float,
        reservoir_size: int = 65536,
        batch_sample_size: int = 4096,
    ):
        self._near_floor_threshold = near_floor_threshold
        self._reservoir_size = reservoir_size
        self._batch_sample_size = batch_sample_size
        # Générateur indépendant : le diagnostic ne consomme pas l'état RNG
        # utilisé pour les échantillons latents et ne change donc pas le modèle.
        self._generator = torch.Generator(device="cpu").manual_seed(0)
        self._sample = torch.empty(0, dtype=torch.float32)
        self._priorities = torch.empty(0, dtype=torch.float32)
        self._count = 0
        self._sum = 0.0
        self._near_floor_count = 0
        self._minimum = float("inf")
        self._maximum = float("-inf")

    def update(self, sigma: Tensor) -> None:
        values = sigma.detach().reshape(-1)
        count = values.numel()
        if count == 0:
            return

        self._count += count
        self._sum += values.sum(dtype=torch.float64).item()
        self._near_floor_count += (
            values <= self._near_floor_threshold
        ).sum().item()
        self._minimum = min(self._minimum, values.min().item())
        self._maximum = max(self._maximum, values.max().item())

        sample_size = min(count, self._batch_sample_size)
        if sample_size == count:
            batch_sample = values.to(device="cpu", dtype=torch.float32)
        else:
            indices = torch.randint(
                count,
                (sample_size,),
                generator=self._generator,
                device="cpu",
            ).to(values.device)
            batch_sample = values.index_select(0, indices).to(
                device="cpu",
                dtype=torch.float32,
            )

        priorities = torch.rand(
            sample_size,
            generator=self._generator,
            dtype=torch.float32,
        )
        self._sample = torch.cat((self._sample, batch_sample))
        self._priorities = torch.cat((self._priorities, priorities))
        if self._sample.numel() > self._reservoir_size:
            kept = self._priorities.topk(self._reservoir_size).indices
            self._sample = self._sample[kept]
            self._priorities = self._priorities[kept]

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            raise ValueError("Aucun sigma n'a été agrégé")

        quantile_levels = torch.tensor(list(self._QUANTILES.values()))
        quantile_values = torch.quantile(self._sample, quantile_levels)
        statistics = {
            "sigma_min": self._minimum,
            "sigma_max": self._maximum,
            "sigma_mean": self._sum / self._count,
            "sigma_fraction_near_floor": (
                self._near_floor_count / self._count
            ),
        }
        statistics.update({
            f"sigma_{name}": value.item()
            for name, value in zip(self._QUANTILES, quantile_values)
        })
        return statistics


class Trainer:
    """Entraîne OmniAnomaly et valide une fois à la fin de chaque époque."""

    def __init__(
        self,
        model: nn.Module,
        train_n_z: int = 1,
        valid_n_z: int = 8,
        max_epoch: Optional[int] = 256,
        max_step: Optional[int] = None,
        batch_size: int = 256,
        valid_batch_size: int = 1024,
        train_log_step_freq: int = 100,
        initial_lr: float = 1e-3,
        lr_anneal_epochs: int = 10,
        lr_anneal_factor: float = 0.75,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[dict] = None,
        grad_clip_norm: Optional[float] = None,
        check_numerics: bool = True,
        weight_decay: float = 0.0,
        early_stopping_validation_patience: int = 20,
        anomaly_filter_enabled: bool = True,
        anomaly_filter_warmup_epochs: int = 2,
        anomaly_filter_k: float = 5.0,
        anomaly_filter_history_size: int = 10000,
        anomaly_filter_min_history: int = 1000,
        device: Optional[str | torch.device] = None,
        feature_names=None,
    ):
        if max_epoch is None and max_step is None:
            raise ValueError("max_epoch ou max_step doit être défini")
        if train_n_z < 1 or valid_n_z < 1:
            raise ValueError("train_n_z et valid_n_z doivent être positifs")
        if batch_size < 1 or valid_batch_size < 1:
            raise ValueError("Les tailles de batch doivent être positives")
        if train_log_step_freq < 1:
            raise ValueError(
                "train_log_step_freq doit être supérieur ou égal à 1"
            )
        if early_stopping_validation_patience < 1:
            raise ValueError(
                "early_stopping_validation_patience doit être positif"
            )
        if anomaly_filter_history_size < 1:
            raise ValueError("anomaly_filter_history_size doit être positif")
        self._model = model
        self._train_n_z = train_n_z
        self._valid_n_z = valid_n_z
        self._max_epoch = max_epoch
        self._max_step = max_step
        self._batch_size = batch_size
        self._valid_batch_size = valid_batch_size
        self._train_log_step_freq = train_log_step_freq
        self._grad_clip_norm = grad_clip_norm
        self._check_numerics = check_numerics
        self._early_stopping_validation_patience = (
            early_stopping_validation_patience
        )
        self._anomaly_filter_enabled = anomaly_filter_enabled
        self._anomaly_filter_warmup_epochs = anomaly_filter_warmup_epochs
        self._anomaly_filter_k = anomaly_filter_k
        self._anomaly_filter_min_history = anomaly_filter_min_history
        self._loss_history = deque(maxlen=anomaly_filter_history_size)
        self.scaler = MinMaxScaler()

        self._device = torch.device(
            device
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model.to(self._device)

        optimizer_params = (
            {} if optimizer_params is None else dict(optimizer_params)
        )
        optimizer_params.setdefault("lr", initial_lr)
        optimizer_params.setdefault("weight_decay", weight_decay)
        self._optimizer = optimizer_class(
            self._model.parameters(),
            **optimizer_params,
        )

        self._scheduler = (
            torch.optim.lr_scheduler.StepLR(
                self._optimizer,
                step_size=lr_anneal_epochs,
                gamma=lr_anneal_factor,
            )
            if lr_anneal_epochs > 0
            else None
        )
        self._global_step = 0
        self._epoch = 0

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def global_step(self) -> int:
        return self._global_step

    def _check_loss(self, loss: Tensor) -> None:
        if self._check_numerics and not torch.isfinite(loss).all():
            raise FloatingPointError("La perte contient des NaN ou des infinis")

    def _check_gradients(self) -> None:
        if not self._check_numerics:
            return
        for name, parameter in self._model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    f"Le gradient de {name} contient des NaN ou infinis"
                )

    def _training_step(self, batch_x: Tensor, epoch: int) -> dict[str, float]:
        self._model.train()
        batch_x = batch_x.to(device=self._device, dtype=torch.float32)
        self._optimizer.zero_grad(set_to_none=True)

        components = self._model.get_per_window_loss(
            x=batch_x,
            n_z=self._train_n_z,
        )
        per_window_losses = components["loss"]
        self._check_loss(per_window_losses)

        detached_losses = per_window_losses.detach()
        batch_rejected = False
        if (
            self._anomaly_filter_enabled
            and epoch > self._anomaly_filter_warmup_epochs
            and len(self._loss_history) >= self._anomaly_filter_min_history
        ):
            history = torch.tensor(
                list(self._loss_history),
                device=per_window_losses.device,
            )
            median = history.median()
            mad = (history - median).abs().median()
            threshold = median + self._anomaly_filter_k * 1.4826 * mad
            batch_rejected = bool((detached_losses > threshold).any().item())

        metrics = {
            "loss": detached_losses.mean().item(),
            "nll": components["nll"].detach().mean().item(),
            "kl": components["kl"].detach().mean().item(),
            "mse": components["mse"].detach().mean().item(),
            "rejected_windows": len(batch_x) if batch_rejected else 0,
            "total_windows": len(batch_x),
            "batch_rejected": batch_rejected,
            "sigma": components["sigma"],
        }
        if batch_rejected:
            return metrics

        batch_loss = per_window_losses.mean()
        self._check_loss(batch_loss)
        batch_loss.backward()
        self._check_gradients()

        if self._grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                max_norm=self._grad_clip_norm,
            )

        self._optimizer.step()
        self._global_step += 1
        self._loss_history.extend(detached_losses.cpu().tolist())
        return metrics

    @torch.no_grad()
    def _validation_loss(
        self,
        valid_iterator: BatchSlidingWindow,
        valid_values: Tensor,
    ) -> dict[str, float]:
        self._model.eval()
        totals = {"loss": 0.0, "nll": 0.0, "kl": 0.0, "mse": 0.0}
        sigma_statistics = SigmaEpochStatistics(
            near_floor_threshold=2.0 * self._model.config.std_epsilon,
        )
        observations = 0

        for (batch_x,) in valid_iterator.get_iterator([valid_values]):
            batch_x = batch_x.to(device=self._device, dtype=torch.float32)
            components = self._model.get_per_window_loss(
                x=batch_x,
                n_z=self._valid_n_z,
            )
            self._check_loss(components["loss"])
            sigma_statistics.update(components["sigma"])
            batch_size = len(batch_x)
            observations += batch_size
            for name in totals:
                totals[name] += components[name].mean().item() * batch_size

        if observations == 0:
            raise ValueError("Aucun batch de validation n'a été produit")
        metrics = {
            name: value / observations for name, value in totals.items()
        }
        metrics.update(sigma_statistics.compute())
        return metrics

    def fit(
        self,
        values,
        valid_portion: float = 0.3,
        excludes: Optional[Tensor] = None,
        checkpoint_path: Optional[str | Path] = None,
    ) -> dict[str, float]:
        if not 0.0 < valid_portion < 1.0:
            raise ValueError(
                "valid_portion doit être compris strictement entre 0 et 1"
            )

        values = torch.as_tensor(values, dtype=torch.float32)
        if values.ndim != 2:
            raise ValueError("values doit avoir la forme [temps, x_dim]")
        if values.shape[1] != self.model.x_dims:
            raise ValueError(
                f"values contient {values.shape[1]} dimensions, "
                f"mais le modèle en attend {self.model.x_dims}"
            )

        if excludes is not None:
            excludes = torch.as_tensor(np.array(excludes), dtype=torch.bool)
            if excludes.shape != (len(values),):
                raise ValueError(
                    f"excludes doit avoir la forme ({len(values)},)"
                )

        validation_size = int(len(values) * valid_portion)
        if validation_size < self.model.window_length:
            raise ValueError(
                "La partie validation est plus courte que window_length"
            )
        split_index = len(values) - validation_size
        train_values = values[:split_index]
        valid_values = values[split_index:]
        if len(train_values) < self.model.window_length:
            raise ValueError(
                "La partie entraînement est plus courte que window_length"
            )

        if excludes is None:
            train_excludes = torch.zeros(len(train_values), dtype=torch.bool)
            valid_excludes = torch.zeros(len(valid_values), dtype=torch.bool)
        else:
            train_excludes = excludes[:split_index]
            valid_excludes = excludes[split_index:]

        # Le split temporel précède le fit : le scaler de l'embedding ne voit que
        # les observations valides de la partie entraînement.
        if not self.scaler.fitted:
            self.scaler.fit(train_values, train_excludes)
        train_values = self.scaler.transform(train_values)
        valid_values = self.scaler.transform(valid_values)

        train_sliding_window = BatchSlidingWindow(
            array_size=len(train_values),
            window_size=self.model.window_length,
            batch_size=self._batch_size,
            excludes=train_excludes,
            shuffle=True,
            ignore_incomplete_batch=True,
        )
        valid_sliding_window = BatchSlidingWindow(
            array_size=len(valid_values),
            window_size=self.model.window_length,
            batch_size=self._valid_batch_size,
            excludes=valid_excludes,
            shuffle=False,
            ignore_incomplete_batch=False,
        )
        if len(train_sliding_window) < self._batch_size:
            raise ValueError(
                "Il n'y a pas assez de fenêtres pour former un batch "
                "d'entraînement complet"
            )

        pyro.clear_param_store()
        best_valid_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0
        train_batch_times = []
        validation_times = []
        should_stop = False

        while not should_stop:
            epoch = self._epoch + 1
            if self._max_epoch is not None and epoch > self._max_epoch:
                break
            self._epoch = epoch

            sums = {"loss": 0.0, "nll": 0.0, "kl": 0.0, "mse": 0.0}
            train_sigma_statistics = SigmaEpochStatistics(
                near_floor_threshold=2.0 * self._model.config.std_epsilon,
            )
            observations = 0
            rejected_windows = 0
            total_windows = 0
            rejected_batches = 0
            epoch_batch_count = len(train_sliding_window) // self._batch_size

            train_iterator = train_sliding_window.get_iterator([train_values])
            for batch_index, (batch_x,) in enumerate(train_iterator, start=1):
                batch_start = perf_counter()
                step = self._training_step(batch_x, epoch)
                train_batch_times.append(perf_counter() - batch_start)
                batch_size = len(batch_x)
                if not step["batch_rejected"]:
                    train_sigma_statistics.update(step["sigma"])
                    observations += batch_size
                    for name in sums:
                        sums[name] += step[name] * batch_size
                else:
                    rejected_batches += 1
                rejected_windows += step["rejected_windows"]
                total_windows += step["total_windows"]

                if (
                    batch_index % self._train_log_step_freq == 0
                    or batch_index == epoch_batch_count
                ):
                    running_metrics = {
                        name: value / observations if observations else float("nan")
                        for name, value in sums.items()
                    }
                    print(
                        f"Epoch {epoch:03d} | "
                        f"batch={batch_index}/{epoch_batch_count} | "
                        f"global_step={self._global_step} | "
                        f"train_loss={running_metrics['loss']:.6f} | "
                        f"train_nll={running_metrics['nll']:.6f} | "
                        f"train_kl={running_metrics['kl']:.6f} | "
                        f"train_mse={running_metrics['mse']:.6f} | "
                        f"rejected_windows="
                        f"{rejected_windows / total_windows:.2%} | "
                        f"rejected_batches={rejected_batches}/{batch_index} | "
                        f"learning_rate="
                        f"{self._optimizer.param_groups[0]['lr']:.8f}",
                        flush=True,
                    )

                if self._max_step is not None and self._global_step >= self._max_step:
                    should_stop = True
                    break

            if observations == 0:
                raise ValueError("Aucun batch d'entraînement n'a été produit")

            if self._scheduler is not None:
                self._scheduler.step()

            validation_start = perf_counter()
            valid_metrics = self._validation_loss(
                valid_sliding_window,
                valid_values,
            )
            validation_times.append(perf_counter() - validation_start)
            train_metrics = {
                name: value / observations for name, value in sums.items()
            }
            train_metrics.update(train_sigma_statistics.compute())
            rejection_rate = rejected_windows / total_windows
            learning_rate = self._optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"valid_loss={valid_metrics['loss']:.6f} | "
                f"train_nll={train_metrics['nll']:.6f} | "
                f"train_kl={train_metrics['kl']:.6f} | "
                f"train_mse={train_metrics['mse']:.6f} | "
                f"valid_nll={valid_metrics['nll']:.6f} | "
                f"valid_kl={valid_metrics['kl']:.6f} | "
                f"valid_mse={valid_metrics['mse']:.6f} | "
                f"train_sigma_min={train_metrics['sigma_min']:.6e} | "
                f"train_sigma_p01={train_metrics['sigma_p01']:.6e} | "
                f"train_sigma_p10={train_metrics['sigma_p10']:.6e} | "
                f"train_sigma_median={train_metrics['sigma_median']:.6e} | "
                f"train_sigma_mean={train_metrics['sigma_mean']:.6e} | "
                f"train_sigma_p90={train_metrics['sigma_p90']:.6e} | "
                f"train_sigma_p99={train_metrics['sigma_p99']:.6e} | "
                f"train_sigma_max={train_metrics['sigma_max']:.6e} | "
                f"train_sigma_fraction_near_floor="
                f"{train_metrics['sigma_fraction_near_floor']:.2%} | "
                f"valid_sigma_min={valid_metrics['sigma_min']:.6e} | "
                f"valid_sigma_p01={valid_metrics['sigma_p01']:.6e} | "
                f"valid_sigma_p10={valid_metrics['sigma_p10']:.6e} | "
                f"valid_sigma_median={valid_metrics['sigma_median']:.6e} | "
                f"valid_sigma_mean={valid_metrics['sigma_mean']:.6e} | "
                f"valid_sigma_p90={valid_metrics['sigma_p90']:.6e} | "
                f"valid_sigma_p99={valid_metrics['sigma_p99']:.6e} | "
                f"valid_sigma_max={valid_metrics['sigma_max']:.6e} | "
                f"valid_sigma_fraction_near_floor="
                f"{valid_metrics['sigma_fraction_near_floor']:.2%} | "
                f"learning_rate={learning_rate:.8f} | "
                f"rejected_windows={rejection_rate:.2%}"
                f" | rejected_batches={rejected_batches}/{epoch_batch_count}"
            )

            if valid_metrics["loss"] < best_valid_loss:
                best_valid_loss = valid_metrics["loss"]
                epochs_without_improvement = 0
                best_model_state = deepcopy(self._model.state_dict())
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path)
            else:
                epochs_without_improvement += 1
                if (
                    epochs_without_improvement
                    >= self._early_stopping_validation_patience
                ):
                    print(
                        "Arrêt anticipé après "
                        f"{epochs_without_improvement} époques sans amélioration"
                    )
                    should_stop = True

        if best_model_state is not None:
            if checkpoint_path is not None and Path(checkpoint_path).is_file():
                self.load_checkpoint(checkpoint_path)
            else:
                self._model.load_state_dict(best_model_state)

        return {
            "best_valid_loss": best_valid_loss,
            "train_time": (
                sum(train_batch_times) / len(train_batch_times)
                if train_batch_times else 0.0
            ),
            "valid_time": (
                sum(validation_times) / len(validation_times)
                if validation_times else 0.0
            ),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": (
                    self._scheduler.state_dict()
                    if self._scheduler is not None else None
                ),
                "global_step": self._global_step,
                "epoch": self._epoch,
                "scaler_state_dict": self.scaler.state_dict(),
                "loss_history": list(self._loss_history),
                "model_x_dim": self._model.x_dims,
                "feature_representation": "amplitude_phase_cos_sin_v1",
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(
            path,
            map_location=self._device,
            weights_only=True,
        )
        if "scaler_state_dict" not in checkpoint:
            raise ValueError(
                "Ce checkpoint ancien ne contient pas le scaler requis pour "
                "une reprise ou une inférence sûre"
            )

        validate_checkpoint_model_dimension(checkpoint, self._model.x_dims)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            self._scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self._global_step = checkpoint["global_step"]
        self._epoch = checkpoint.get("epoch", 0)
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self._loss_history.clear()
        self._loss_history.extend(checkpoint.get("loss_history", []))
