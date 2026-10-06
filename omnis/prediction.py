# predictor.py

from time import perf_counter
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

from .utils import BatchSlidingWindow


class Predictor:
    """
    Calcule les scores de reconstruction d'OmniAnomaly.

    Paramètres
    ----------
    model:
        Modèle OmniAnomaly entraîné.

    n_z:
        Nombre d'échantillons latents par fenêtre.

    batch_size:
        Nombre de fenêtres traitées simultanément.

    last_point_only:
        Si True, retourne seulement le score du dernier point
        de chaque fenêtre.

    device:
        CPU ou GPU utilisé pour l'inférence.
    """

    def __init__(
        self,
        model: nn.Module,
        n_z: int = 1024,
        batch_size: int = 32,
        last_point_only: bool = True,
        device: Optional[str | torch.device] = None,
        scaler=None,
        normalizer=None,
    ):
        if n_z < 1:
            raise ValueError(
                "n_z doit être supérieur ou égal à 1"
            )

        if batch_size < 1:
            raise ValueError(
                "batch_size doit être supérieur ou égal à 1"
            )

        self._model = model
        self._n_z = n_z
        self._batch_size = batch_size
        self._last_point_only = last_point_only
        self.normalizer = normalizer or scaler
        self.scaler = self.normalizer

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        else:
            device = torch.device(device)

        self._device = device
        self._model.to(device)

    @property
    def model(self) -> nn.Module:
        return self._model

    def _synchronize_device(self) -> None:
        """
        Attend la fin des calculs GPU pour mesurer correctement le temps.
        """
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    @torch.inference_mode()
    def get_score(
        self,
        values,
        excludes: Optional[Tensor] = None,
        normalized: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Calcule les scores de reconstruction.

        Entrée
        ------
        values:
            Données de forme [nombre_de_points, x_dim].

        excludes:
            Masque booléen des points à exclure.

        Retour
        ------
        scores:
            Scores de reconstruction.

        latent_statistics:
            Moyennes et écarts-types des latents.

        mean_prediction_time:
            Temps moyen par mini-batch.
        """
        if not normalized:
            if self.normalizer is None or not self.normalizer.fitted:
                raise RuntimeError(
                    "Un scaler ajusté est obligatoire pour traiter des données brutes"
                )
            if not self.normalizer.requires_windows:
                values, _ = self.normalizer.transform(values)

        values = torch.as_tensor(
            values,
            dtype=torch.float32,
        )

        if values.ndim != 2:
            raise ValueError(
                "values doit avoir la forme [temps, x_dim]"
            )
        if values.shape[1] != self.model.x_dims:
            raise ValueError(
                f"values contient {values.shape[1]} dimensions, "
                f"mais le modèle en attend {self.model.x_dims}"
            )

        if len(values) < self.model.window_length:
            raise ValueError(
                "values doit contenir au moins window_length points"
            )

        if excludes is not None:
            excludes = np.array(excludes)
            excludes = torch.as_tensor(
                excludes,
                dtype=torch.bool,
            )

            if excludes.shape != (len(values),):
                raise ValueError(
                    f"excludes doit avoir la forme ({len(values)},)"
                )

        self._model.eval()

        sliding_window = BatchSlidingWindow(
            array_size=len(values),
            window_size=self.model.window_length,
            batch_size=self._batch_size,
            excludes=excludes,
            shuffle=False,
            ignore_incomplete_batch=False,
        )

        score_collector = []
        latent_collector = []
        prediction_times = []

        iterator = sliding_window.get_iterator([values])

        for (batch_x,) in iterator:
            if not normalized and self.normalizer.requires_windows:
                batch_x, _ = self.normalizer.transform(batch_x)
            batch_x = batch_x.to(
                device=self._device,
                dtype=torch.float32,
            )

            self._synchronize_device()
            start_time = perf_counter()

            batch_scores, batch_latents = self.model.get_score(
                x=batch_x,
                n_z=self._n_z,
                last_point_only=self._last_point_only,
            )

            self._synchronize_device()

            prediction_times.append(
                perf_counter() - start_time
            )

            score_collector.append(
                batch_scores.detach().cpu()
            )

            latent_collector.append(
                batch_latents.detach().cpu()
            )

        if not score_collector:
            raise ValueError(
                "Aucune fenêtre valide n'a été produite"
            )

        scores = torch.cat(
            score_collector,
            dim=0,
        ).numpy()

        latent_statistics = torch.cat(
            latent_collector,
            dim=0,
        ).numpy()

        mean_prediction_time = float(
            np.mean(prediction_times)
        )

        return (
            scores,
            latent_statistics,
            mean_prediction_time,
        )
