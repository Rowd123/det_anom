# trainer.py

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Optional

import pyro
import torch
from torch import Tensor, nn

from utils import BatchSlidingWindow


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
        early_stopping_patience: Arrête l'entraînement si la perte de validation ne s'améliore pas pendant
                                le nombre assingé
        device: où exécuter les calculs (cude - cpu)


    """

    def __init__(
        self,
        model: nn.Module,
        n_z: int = 1,
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
        early_stopping_patience: int = 20,
        device: Optional[str | torch.device] = None,
    ):
        if max_epoch is None and max_step is None:
            raise ValueError(
                "max_epoch ou max_step doit être défini"
            )

        if n_z < 1:
            raise ValueError("n_z doit être supérieur ou égal à 1")

        if batch_size < 1 or valid_batch_size < 1:
            raise ValueError(
                "Les tailles de batch doivent être positives"
            )

        self._model = model
        self._n_z = n_z
        self._max_epoch = max_epoch
        self._max_step = max_step
        self._batch_size = batch_size
        self._valid_batch_size = valid_batch_size
        self._valid_step_freq = valid_step_freq
        self._grad_clip_norm = grad_clip_norm
        self._check_numerics = check_numerics
        self._early_stopping_patience = early_stopping_patience

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

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def global_step(self) -> int:
        return self._global_step

    def _check_loss(self, loss: Tensor) -> None:
        if self._check_numerics and not torch.isfinite(loss):
            raise FloatingPointError(
                f"La perte n'est pas finie : {loss.item()}"
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

    def _training_step(self, batch_x: Tensor) -> float:
        self._model.train()

        batch_x = batch_x.to(
            device=self._device,
            dtype=torch.float32,
        )

        self._optimizer.zero_grad(set_to_none=True)

        loss = self._model.get_training_loss(
            x=batch_x,
            n_z=self._n_z,
        )

        self._check_loss(loss)

        loss.backward()

        self._check_gradients()

        if self._grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                max_norm=self._grad_clip_norm,
            )

        self._optimizer.step()
        self._global_step += 1

        return loss.detach().item()

    @torch.no_grad()
    def _validation_loss(
        self,
        valid_iterator: BatchSlidingWindow,
        valid_values: Tensor,
    ) -> float:
        self._model.eval()

        total_loss = 0.0
        total_observations = 0

        iterator = valid_iterator.get_iterator([valid_values])

        for (batch_x,) in iterator:
            batch_x = batch_x.to(
                device=self._device,
                dtype=torch.float32,
            )

            loss = self._model.get_training_loss(
                x=batch_x,
                n_z=self._n_z,
            )

            self._check_loss(loss)

            total_loss += loss.item()
            total_observations += len(batch_x)

        if total_observations == 0:
            raise ValueError(
                "Aucun batch de validation n'a été produit"
            )

        return total_loss / total_observations

    def fit(
        self,
        values,
        valid_portion: float = 0.3,
        excludes: Optional[Tensor] = None,
        checkpoint_path: Optional[str | Path] = None,
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
                excludes,
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
            train_excludes = None
            valid_excludes = None
        else:
            train_excludes = excludes[:split_index]
            valid_excludes = excludes[split_index:]

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
        epochs_without_improvement = 0

        train_batch_times = []
        valid_batch_times = []

        epoch = 0
        should_stop = False

        while not should_stop:
            epoch += 1

            if self._max_epoch is not None and epoch > self._max_epoch:
                break

            train_iterator = train_sliding_window.get_iterator(
                [train_values]
            )

            epoch_loss = 0.0
            epoch_observations = 0

            for (batch_x,) in train_iterator:
                batch_start = perf_counter()

                loss = self._training_step(batch_x)

                train_batch_times.append(
                    perf_counter() - batch_start
                )

                epoch_loss += loss
                epoch_observations += len(batch_x)

                if (
                    self._global_step % self._valid_step_freq == 0
                ):
                    validation_start = perf_counter()

                    valid_loss = self._validation_loss(
                        valid_sliding_window,
                        valid_values,
                    )

                    valid_batch_times.append(
                        perf_counter() - validation_start
                    )

                    print(
                        f"Epoch {epoch:03d} | "
                        f"step {self._global_step:06d} | "
                        f"train_loss "
                        f"{epoch_loss / epoch_observations:.6f} | "
                        f"valid_loss {valid_loss:.6f}"
                    )

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss

                        best_model_state = deepcopy(
                            self._model.state_dict()
                        )

                        epochs_without_improvement = 0

                        if checkpoint_path is not None:
                            self.save_checkpoint(checkpoint_path)
                    else:
                        epochs_without_improvement += 1

                    if (
                        epochs_without_improvement
                        >= self._early_stopping_patience
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

        if best_model_state is not None:
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
        }

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