# trainer.py

from copy import deepcopy
from collections import deque
import json
from pathlib import Path
from time import perf_counter
from typing import Optional
import numpy as np
import pyro
import torch
from torch import Tensor, nn

from utils import BatchSlidingWindow, MinMaxScaler


class Trainer:
    """
    Entraîne un modèle OmniAnomaly avec PyTorch/Pyro.

    Paramètres:
    -------------
        model: le modèle (omni_anomaly)
        n_z: nombre d'échantillons utilisés pour estimer l'ELBO à chaque batch
        max_epoch: Nombre maximal de passages complets sur les données
        max_step: nombre maximal de mises à jour des paramètres. Avec None seule la limite d'époque sera utilisée
        batch_size: Nombre de fenêtres dans un batch d'entraînement
        valide_batch_size: Nombre de fenêtres dans un batch de validation
        valid_ster_freq: la fréquence de lancement du procédé de la validation
        initial_lr: Taux d'apprentissage initial
        lr_anneal_epochs: la fréquence de la réduction du taux d'apprentissage
        lr_anneal_factor: Facteur de réduction du taux d'apprentissage :
                                            nouveau_lr = ancien_lr * lr_anneal_factor
        optimizer_class: classe d'optimiseur utilisée
        optimizer_params: Paramètres supplémentaires pour l'optimiseur
        weight_decay: Régularisation appliquée aux poids
        grad_clip_norm: Limite la norme totale des gradients pour éviter leur explosion
        check_numerics: Vérifie que la perte et les gradients ne contiennent pas de NaN ou d'infinis
        early_stopping_validation_patience: Nombre de contrôles de validation
                                            consécutifs sans amélioration
        device: où exécuter les calculs (cude - cpu)


    """

    def __init__(
        self,
        model: nn.Module,
        train_n_z: int = 1,
        valid_n_z: int = 8,
        max_epoch: Optional[int] = 256,
        max_step: Optional[int] = None,
        batch_size: int = 256,
        valid_batch_size: int = 1024,
        valid_step_freq: int = 100,
        initial_lr: float = 1e-3,
        lr_anneal_epochs: int = 10,
        lr_anneal_factor: float = 0.75,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[dict] = None,
        grad_clip_norm: Optional[float] = 50.0,
        check_numerics: bool = True,
        weight_decay: float = 0.0,
        early_stopping_validation_patience: int = 20,
        anomaly_filter_enabled: bool = True,
        anomaly_filter_warmup_epochs: int = 2,
        anomaly_filter_k: float = 5.0,
        anomaly_filter_history_size: int = 10000,
        anomaly_filter_min_history: int = 1000,
        anomaly_filter_max_fraction: float = 0.10,
        device: Optional[str | torch.device] = None,
    ):
        if max_epoch is None and max_step is None:
            raise ValueError(
                "max_epoch ou max_step doit être défini"
            )

        if train_n_z < 1 or valid_n_z < 1:
            raise ValueError("train_n_z et valid_n_z doivent être positifs")

        if batch_size < 1 or valid_batch_size < 1:
            raise ValueError(
                "Les tailles de batch doivent être positives"
            )

        if valid_step_freq < 1:
            raise ValueError(
                "valid_step_freq doit être supérieur ou égal à 1"
            )

        self._model = model
        if anomaly_filter_history_size < 1:
            raise ValueError("anomaly_filter_history_size doit être positif")
        if not 0.0 <= anomaly_filter_max_fraction < 1.0:
            raise ValueError("anomaly_filter_max_fraction doit être dans [0, 1[")

        self._train_n_z = train_n_z
        self._valid_n_z = valid_n_z
        self._max_epoch = max_epoch
        self._max_step = max_step
        self._batch_size = batch_size
        self._valid_batch_size = valid_batch_size
        self._valid_step_freq = valid_step_freq
        self._grad_clip_norm = grad_clip_norm
        self._check_numerics = check_numerics
        self._early_stopping_validation_patience = (
            early_stopping_validation_patience
        )
        self._anomaly_filter_enabled = anomaly_filter_enabled
        self._anomaly_filter_warmup_epochs = anomaly_filter_warmup_epochs
        self._anomaly_filter_k = anomaly_filter_k
        self._anomaly_filter_min_history = anomaly_filter_min_history
        self._anomaly_filter_max_fraction = anomaly_filter_max_fraction
        self._loss_history = deque(maxlen=anomaly_filter_history_size)
        self._recent_batch_losses = deque(maxlen=100)
        self.scaler = MinMaxScaler()

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            device = torch.device(device)

        self._device = device
        self._model.to(device)

        if optimizer_params is None:
            optimizer_params = {}
        else:
            optimizer_params = dict(optimizer_params)

        optimizer_params.setdefault("lr", initial_lr)
        optimizer_params.setdefault("weight_decay", weight_decay)

        self._optimizer = optimizer_class(
            self._model.parameters(),
            **optimizer_params,
        )

        if lr_anneal_epochs > 0:
            self._scheduler = torch.optim.lr_scheduler.StepLR(
                self._optimizer,
                step_size=lr_anneal_epochs,
                gamma=lr_anneal_factor,
            )
        else:
            self._scheduler = None

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
            raise FloatingPointError(
                "La perte contient des NaN ou des infinis"
            )

    def _check_gradients(self) -> None:
        if not self._check_numerics:
            return

        for name, parameter in self._model.named_parameters():
            if parameter.grad is None:
                continue

            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    f"Le gradient de {name} contient des NaN ou infinis"
                )

    @staticmethod
    def _tensor_stats(values: Tensor) -> tuple[float, float, float]:
        values = values.detach()
        return (
            values.min().item(),
            values.median().item(),
            values.max().item(),
        )

    def _gradient_norm(self) -> float:
        squared_norm = sum(
            parameter.grad.detach().norm(2).item() ** 2
            for parameter in self._model.parameters()
            if parameter.grad is not None
        )
        return squared_norm ** 0.5

    def _gradient_norms_by_group(self) -> dict[str, float]:
        squared = {"encoder": 0.0, "posterior": 0.0, "decoder": 0.0, "other": 0.0}
        for name, parameter in self._model.named_parameters():
            if parameter.grad is None:
                continue
            if "posterior" in name:
                group = "posterior"
            elif "encoder" in name:
                group = "encoder"
            elif "decoder" in name:
                group = "decoder"
            else:
                group = "other"
            squared[group] += parameter.grad.detach().norm(2).item() ** 2
        return {name: value ** 0.5 for name, value in squared.items()}

    @staticmethod
    def _scale_quantiles(values: Tensor) -> dict[str, float]:
        values = values.detach().flatten().float()
        quantiles = torch.quantile(
            values,
            values.new_tensor([0.01, 0.05, 0.50, 0.95, 0.99]),
        )
        return {
            "min": values.min().item(),
            "q01": quantiles[0].item(),
            "q05": quantiles[1].item(),
            "q50": quantiles[2].item(),
            "q95": quantiles[3].item(),
            "q99": quantiles[4].item(),
            "max": values.max().item(),
        }

    @staticmethod
    def _correlation(left: Tensor, right: Tensor) -> float:
        left = left.detach().float().flatten()
        right = right.detach().float().flatten()
        left = left - left.mean()
        right = right - right.mean()
        denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
        if denominator.item() == 0.0:
            return float("nan")
        return (left * right).sum().div(denominator).item()

    @staticmethod
    def _ranks(values: Tensor) -> Tensor:
        order = values.detach().flatten().argsort()
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(len(order), device=order.device).float()
        return ranks

    def _window_diagnostics(self, components: dict[str, Tensor]) -> dict:
        if "decoder_scale_mean" in components:
            per_window_scale_mean = components["decoder_scale_mean"].detach()
            per_window_scale_min = components["decoder_scale_min"].detach()
            per_window_scale_median = components["decoder_scale_median"].detach()
        else:
            scale = components["decoder_scale"].detach()
            per_window_scale_mean = scale.mean(dim=(0, 2, 3))
            per_window_scale_min = scale.amin(dim=(0, 2, 3))
            per_window_scale_median = (
                scale.permute(1, 0, 2, 3)
                .flatten(1)
                .median(dim=1)
                .values
            )
        losses = components["loss"].detach()
        mse = components["mse"].detach()
        correlations = {
            "pearson_loss_mse": self._correlation(losses, mse),
            "pearson_loss_scale_mean": self._correlation(losses, per_window_scale_mean),
            "pearson_loss_scale_min": self._correlation(losses, per_window_scale_min),
            "spearman_loss_mse": self._correlation(self._ranks(losses), self._ranks(mse)),
            "spearman_loss_scale_mean": self._correlation(
                self._ranks(losses), self._ranks(per_window_scale_mean)
            ),
            "spearman_loss_scale_min": self._correlation(
                self._ranks(losses), self._ranks(per_window_scale_min)
            ),
        }
        top_indices = torch.topk(losses, min(10, len(losses))).indices
        top_windows = []
        for index in top_indices.tolist():
            top_windows.append({
                "index_in_batch": index,
                "loss": losses[index].item(),
                "nll": components["nll"][index].detach().item(),
                "kl": components["kl"][index].detach().item(),
                "mse": mse[index].item(),
                "mae": components["mae"][index].detach().item(),
                "decoder_scale_min": per_window_scale_min[index].item(),
                "decoder_scale_median": per_window_scale_median[index].item(),
                "decoder_scale_mean": per_window_scale_mean[index].item(),
            })
        return {"correlations": correlations, "top_windows": top_windows}

    def _training_step(self, batch_x: Tensor, epoch: int) -> dict:
        self._model.train()

        batch_x = batch_x.to(
            device=self._device,
            dtype=torch.float32,
        )

        self._optimizer.zero_grad(set_to_none=True)

        components = self._model.get_per_window_loss(
            x=batch_x,
            n_z=self._train_n_z,
        )
        per_window_losses = components["loss"]
        self._check_loss(per_window_losses)

        detached_losses = per_window_losses.detach()
        threshold = None
        filter_status = "disabled"
        accepted_mask = torch.ones_like(per_window_losses, dtype=torch.bool)

        if self._anomaly_filter_enabled:
            if epoch <= self._anomaly_filter_warmup_epochs:
                filter_status = "warm-up"
            elif len(self._loss_history) < self._anomaly_filter_min_history:
                filter_status = "insufficient-history"
            else:
                history = torch.tensor(
                    list(self._loss_history),
                    device=per_window_losses.device,
                )
                median = history.median()
                mad = (history - median).abs().median()
                threshold_tensor = median + self._anomaly_filter_k * 1.4826 * mad
                threshold = threshold_tensor.item()
                rejected = detached_losses > threshold_tensor
                max_rejected = min(
                    int(len(batch_x) * self._anomaly_filter_max_fraction),
                    len(batch_x) - 1,
                )
                if rejected.sum().item() > max_rejected:
                    rejected = torch.zeros_like(rejected)
                    if max_rejected:
                        highest = torch.topk(detached_losses, max_rejected).indices
                        rejected[highest] = True
                accepted_mask = ~rejected
                filter_status = "active"

        loss = per_window_losses[accepted_mask].mean()

        self._check_loss(loss)
        loss.backward()

        self._check_gradients()
        gradient_norm_before = self._gradient_norm()
        gradient_norms_by_group = self._gradient_norms_by_group()

        if self._grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                max_norm=self._grad_clip_norm,
            )
        gradient_norm_after = self._gradient_norm()

        self._optimizer.step()
        self._global_step += 1

        history_values = detached_losses[accepted_mask]
        self._loss_history.extend(history_values.cpu().tolist())
        batch_loss = detached_losses.mean().item()
        self._recent_batch_losses.append(batch_loss)
        rejected_windows = (~accepted_mask).sum().item()
        window_diagnostics = self._window_diagnostics(components)
        return {
            "batch_loss_mean": batch_loss,
            "filtered_loss": loss.detach().item(),
            "nll": components["nll"].detach().mean().item(),
            "kl": components["kl"].detach().mean().item(),
            "decoder_scale": self._tensor_stats(components["decoder_scale"]),
            "posterior_scale": self._tensor_stats(components["posterior_scale"]),
            "decoder_scale_quantiles": self._scale_quantiles(
                components["decoder_scale"]
            ),
            "mse": components["mse"].detach().mean().item(),
            "mae": components["mae"].detach().mean().item(),
            "log_sigma_term": components["log_sigma_term"].detach().mean().item(),
            "standardized_error_term": components[
                "standardized_error_term"
            ].detach().mean().item(),
            "gradient_norm_before": gradient_norm_before,
            "gradient_norm_after": gradient_norm_after,
            "gradient_norms_by_group": gradient_norms_by_group,
            "threshold": threshold,
            "filter_status": filter_status,
            "rejected_windows": rejected_windows,
            "total_windows": len(batch_x),
            "recent_loss": sum(self._recent_batch_losses) / len(self._recent_batch_losses),
            **window_diagnostics,
        }

    @torch.no_grad()
    def _validation_loss(
        self,
        valid_iterator: BatchSlidingWindow,
        valid_values: Tensor,
        n_z: Optional[int] = None,
    ) -> dict:
        self._model.eval()

        collected = {
            name: [] for name in (
                "loss", "nll", "kl", "mse", "mae", "log_sigma_term",
                "standardized_error_term", "decoder_scale_mean",
                "decoder_scale_min", "decoder_scale_median",
            )
        }
        scale_sample = torch.empty(0, dtype=torch.float32)
        scale_min = float("inf")
        scale_max = float("-inf")
        max_scale_sample_size = 200_000

        iterator = valid_iterator.get_iterator([valid_values])

        for (batch_x,) in iterator:
            batch_x = batch_x.to(
                device=self._device,
                dtype=torch.float32,
            )

            per_window_losses = self._model.get_per_window_loss(
                x=batch_x,
                n_z=self._valid_n_z if n_z is None else n_z,
            )

            self._check_loss(per_window_losses["loss"])

            for name in (
                "loss", "nll", "kl", "mse", "mae", "log_sigma_term",
                "standardized_error_term",
            ):
                collected[name].append(per_window_losses[name].detach().cpu())

            scale = per_window_losses["decoder_scale"].detach()
            scale_by_window = scale.permute(1, 0, 2, 3).flatten(1)
            collected["decoder_scale_mean"].append(
                scale_by_window.mean(dim=1).cpu()
            )
            collected["decoder_scale_min"].append(
                scale_by_window.amin(dim=1).cpu()
            )
            collected["decoder_scale_median"].append(
                scale_by_window.median(dim=1).values.cpu()
            )

            flat_scale = scale.flatten().float()
            scale_min = min(scale_min, flat_scale.min().item())
            scale_max = max(scale_max, flat_scale.max().item())
            sample_step = max(
                1,
                (flat_scale.numel() + 19_999) // 20_000,
            )
            batch_sample = flat_scale[::sample_step][:20_000].cpu()
            scale_sample = torch.cat((scale_sample, batch_sample))
            if scale_sample.numel() > max_scale_sample_size:
                indices = torch.linspace(
                    0,
                    scale_sample.numel() - 1,
                    max_scale_sample_size,
                ).long()
                scale_sample = scale_sample[indices]

        if not collected["loss"]:
            raise ValueError(
                "Aucun batch de validation n'a été produit"
            )

        components = {
            name: torch.cat(values, dim=0)
            for name, values in collected.items()
        }
        result = {
            name: components[name].mean().item()
            for name in (
                "loss", "nll", "kl", "mse", "mae", "log_sigma_term",
                "standardized_error_term",
            )
        }
        result["decoder_scale_quantiles"] = self._scale_quantiles(
            scale_sample
        )
        result["decoder_scale_quantiles"]["min"] = scale_min
        result["decoder_scale_quantiles"]["max"] = scale_max
        result["decoder_scale_quantiles"]["sample_size"] = int(
            scale_sample.numel()
        )
        result["decoder_scale_quantiles"]["approximate"] = True
        result.update(self._window_diagnostics(components))
        return result

    def fit(
        self,
        values,
        valid_portion: float = 0.3,
        excludes: Optional[Tensor] = None,
        checkpoint_path: Optional[str | Path] = None,
        diagnostics_path: Optional[str | Path] = None,
    ) -> dict[str, float]:
        """
        Entraîne OmniAnomaly.

        values :
            Tensor [nombre_de_points, x_dim].

        excludes :
            Masque booléen indiquant les points à exclure.
        """
        if not 0.0 < valid_portion < 1.0:
            raise ValueError(
                "valid_portion doit être compris strictement entre 0 et 1"
            )

        values = torch.as_tensor(
            values,
            dtype=torch.float32,
        )

        if values.ndim != 2:
            raise ValueError(
                "values doit avoir la forme [temps, x_dim]"
            )

        if excludes is not None:
            excludes = torch.as_tensor(
                np.array(excludes),
                dtype=torch.bool,
            )

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

        # Le split est fait sur les données physiques brutes. Le scaler ne voit
        # que la partie entraînement et ignore les lignes exclues/non finies.
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
                "Il n'y a pas assez de fenêtres pour former "
                "un batch d'entraînement complet"
            )

        pyro.clear_param_store()

        best_valid_loss = float("inf")
        best_model_state = None
        validation_checks_without_improvement = 0

        train_batch_times = []
        valid_batch_times = []
        epoch_diagnostics = []

        epoch = self._epoch
        should_stop = False

        while not should_stop:
            epoch += 1
            self._epoch = epoch

            if self._max_epoch is not None and epoch > self._max_epoch:
                break

            train_iterator = train_sliding_window.get_iterator(
                [train_values]
            )

            epoch_loss = 0.0
            epoch_observations = 0
            epoch_sums = {
                "nll": 0.0,
                "mse": 0.0,
                "mae": 0.0,
                "log_sigma_term": 0.0,
                "standardized_error_term": 0.0,
            }
            epoch_scale_min = float("inf")
            epoch_scale_medians = []

            for (batch_x,) in train_iterator:
                batch_start = perf_counter()

                step_metrics = self._training_step(batch_x, epoch)

                train_batch_times.append(
                    perf_counter() - batch_start
                )

                epoch_loss += step_metrics["batch_loss_mean"] * len(batch_x)
                epoch_observations += len(batch_x)
                for name in epoch_sums:
                    epoch_sums[name] += step_metrics[name] * len(batch_x)
                epoch_scale_min = min(
                    epoch_scale_min,
                    step_metrics["decoder_scale_quantiles"]["min"],
                )
                epoch_scale_medians.append(
                    step_metrics["decoder_scale_quantiles"]["q50"]
                )

                if (
                    self._global_step % self._valid_step_freq == 0
                ):
                    validation_start = perf_counter()

                    valid_metrics = self._validation_loss(
                        valid_sliding_window,
                        valid_values,
                    )
                    valid_loss = valid_metrics["loss"]

                    valid_batch_times.append(
                        perf_counter() - validation_start
                    )

                    decoder_scale = step_metrics["decoder_scale"]
                    posterior_scale = step_metrics["posterior_scale"]
                    gradient_groups = step_metrics["gradient_norms_by_group"]
                    rejected = step_metrics["rejected_windows"]
                    total = step_metrics["total_windows"]
                    print(
                        f"Epoch {epoch:03d} | "
                        f"step {self._global_step:06d} | "
                        f"batch_loss_mean {step_metrics['batch_loss_mean']:.6f} | "
                        f"filtered_loss {step_metrics['filtered_loss']:.6f} | "
                        f"recent_loss {step_metrics['recent_loss']:.6f} | "
                        f"train_loss {epoch_loss / epoch_observations:.6f} | "
                        f"valid_loss {valid_loss:.6f} | "
                        f"train_mse {step_metrics['mse']:.6f} | "
                        f"train_mae {step_metrics['mae']:.6f} | "
                        f"valid_mse {valid_metrics['mse']:.6f} | "
                        f"valid_mae {valid_metrics['mae']:.6f} | "
                        f"train_log_sigma_term {step_metrics['log_sigma_term']:.6f} | "
                        f"train_standardized_error_term "
                        f"{step_metrics['standardized_error_term']:.6f} | "
                        f"valid_log_sigma_term {valid_metrics['log_sigma_term']:.6f} | "
                        f"valid_standardized_error_term "
                        f"{valid_metrics['standardized_error_term']:.6f} | "
                        f"threshold {step_metrics['threshold']} | "
                        f"filter {step_metrics['filter_status']} | "
                        f"rejected {rejected}/{total} ({rejected / total:.2%}) | "
                        f"history_size {len(self._loss_history)} | "
                        f"decoder_scale[min/median/max] {decoder_scale} | "
                        f"posterior_scale[min/median/max] {posterior_scale} | "
                        f"grad_norm[before/after] "
                        f"{step_metrics['gradient_norm_before']:.6f}/"
                        f"{step_metrics['gradient_norm_after']:.6f} | "
                        f"grad_groups {gradient_groups} | "
                        f"lr {self._optimizer.param_groups[0]['lr']:.8f}"
                    )
                    print(
                        "Diagnostic fenêtres train | "
                        f"scale={step_metrics['decoder_scale_quantiles']} | "
                        f"corr={step_metrics['correlations']} | "
                        f"top10={step_metrics['top_windows']}"
                    )
                    print(
                        "Diagnostic fenêtres valid | "
                        f"scale={valid_metrics['decoder_scale_quantiles']} | "
                        f"corr={valid_metrics['correlations']} | "
                        f"top10={valid_metrics['top_windows']}"
                    )

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss

                        best_model_state = deepcopy(
                            self._model.state_dict()
                        )

                        validation_checks_without_improvement = 0

                        if checkpoint_path is not None:
                            self.save_checkpoint(checkpoint_path)
                    else:
                        validation_checks_without_improvement += 1

                    if (
                        validation_checks_without_improvement
                        >= self._early_stopping_validation_patience
                    ):
                        print("Arrêt anticipé")
                        should_stop = True
                        break

                if (
                    self._max_step is not None
                    and self._global_step >= self._max_step
                ):
                    should_stop = True
                    break

            if self._scheduler is not None:
                self._scheduler.step()

                learning_rate = (
                    self._optimizer.param_groups[0]["lr"]
                )

                if epoch % self._scheduler.step_size == 0:
                    print(
                        f"Learning rate diminué à {learning_rate:.8f}"
                    )

            if epoch_observations:
                valid_metrics = self._validation_loss(
                    valid_sliding_window,
                    valid_values,
                )
                epoch_metrics = {
                    "epoch": epoch,
                    "train_loss": epoch_loss / epoch_observations,
                    "valid_loss": valid_metrics["loss"],
                    **{
                        f"train_{name}": value / epoch_observations
                        for name, value in epoch_sums.items()
                    },
                    **{
                        f"valid_{name}": valid_metrics[name]
                        for name in epoch_sums
                    },
                    "train_decoder_scale_min": epoch_scale_min,
                    "train_decoder_scale_median": float(
                        np.median(epoch_scale_medians)
                    ),
                    "valid_decoder_scale_min": valid_metrics[
                        "decoder_scale_quantiles"
                    ]["min"],
                    "valid_decoder_scale_median": valid_metrics[
                        "decoder_scale_quantiles"
                    ]["q50"],
                }
                epoch_diagnostics.append(epoch_metrics)
                print(f"Résumé époque | {epoch_metrics}")
                if diagnostics_path is not None:
                    diagnostics_path = Path(diagnostics_path)
                    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                    diagnostics_path.write_text(
                        json.dumps(epoch_diagnostics, indent=2),
                        encoding="utf-8",
                    )

        if best_model_state is not None:
            if checkpoint_path is not None and Path(checkpoint_path).is_file():
                # Restaure ensemble modèle, optimiseur, scheduler, compteur et
                # scaler afin que le checkpoint final reste reprenable.
                self.load_checkpoint(checkpoint_path)
            else:
                self._model.load_state_dict(best_model_state)

        mean_train_time = (
            sum(train_batch_times) / len(train_batch_times)
            if train_batch_times
            else 0.0
        )

        mean_valid_time = (
            sum(valid_batch_times) / len(valid_batch_times)
            if valid_batch_times
            else 0.0
        )

        return {
            "best_valid_loss": best_valid_loss,
            "train_time": mean_train_time,
            "valid_time": mean_valid_time,
            "diagnostic_summary": self._interpret_diagnostics(epoch_diagnostics),
        }

    @staticmethod
    def _interpret_diagnostics(history: list[dict]) -> str:
        if len(history) < 2:
            return "Historique insuffisant pour interpréter l'évolution."
        first, last = history[0], history[-1]
        mse_ratio = last["train_mse"] / max(first["train_mse"], 1e-12)
        scale_ratio = last["train_decoder_scale_median"] / max(
            first["train_decoder_scale_median"], 1e-12
        )
        valid_gap = last["valid_mse"] / max(last["train_mse"], 1e-12)
        return (
            f"train_mse final/initial={mse_ratio:.4g}; "
            f"decoder_scale_median final/initial={scale_ratio:.4g}; "
            f"valid_mse/train_mse={valid_gap:.4g}. "
            "Une baisse forte de MSE indique une mémorisation réelle; une baisse "
            "surtout portée par sigma avec MSE stable indique un effet de variance."
        )

    @torch.no_grad()
    def monte_carlo_validation(
        self,
        values,
        valid_portion: float,
        excludes=None,
        repetitions: int = 10,
        n_z: Optional[int] = None,
    ) -> dict:
        """Répète la validation à poids fixes, sans appel à optimizer.step()."""
        if repetitions < 2:
            raise ValueError("repetitions doit être supérieur ou égal à 2")
        values = torch.as_tensor(values, dtype=torch.float32)
        validation_size = int(len(values) * valid_portion)
        valid_values = self.scaler.transform(values[-validation_size:])
        if excludes is None:
            valid_excludes = None
        else:
            valid_excludes = torch.as_tensor(
                np.asarray(excludes)[-validation_size:], dtype=torch.bool
            )
        iterator = BatchSlidingWindow(
            array_size=len(valid_values),
            window_size=self.model.window_length,
            batch_size=self._valid_batch_size,
            excludes=valid_excludes,
            shuffle=False,
            ignore_incomplete_batch=False,
        )
        repeated = []
        for _ in range(repetitions):
            metrics = self._validation_loss(iterator, valid_values, n_z=n_z)
            repeated.append({
                name: metrics[name]
                for name in ("loss", "nll", "kl", "mse", "mae")
            })
        summary = {}
        for name in repeated[0]:
            samples = np.asarray([item[name] for item in repeated])
            summary[name] = {
                "mean": float(samples.mean()),
                "std": float(samples.std(ddof=1)),
                "min": float(samples.min()),
                "max": float(samples.max()),
            }
        print(
            f"Diagnostic Monte-Carlo validation "
            f"(repetitions={repetitions}, n_z={n_z or self._valid_n_z}) | "
            f"{summary}"
        )
        return summary

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": (
                    self._scheduler.state_dict()
                    if self._scheduler is not None
                    else None
                ),
                "global_step": self._global_step,
                "epoch": self._epoch,
                "scaler_state_dict": self.scaler.state_dict(),
                "loss_history": list(self._loss_history),
            },
            path,
        )

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
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

        self._model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self._optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if (
            self._scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):
            self._scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        self._global_step = checkpoint["global_step"]
        self._epoch = checkpoint.get("epoch", 0)
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self._loss_history.clear()
        self._loss_history.extend(checkpoint.get("loss_history", []))
