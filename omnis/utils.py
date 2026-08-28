from collections.abc import Iterable, Iterator

import torch
from torch import Tensor


def preprocess(data) -> Tensor:
    """
    Convertit les données en float32, remplace les NaN,
    puis normalise chaque variable entre 0 et 1.

    Entrée :
        [n_observations, n_features]

    Sortie :
        Tensor float32 de même forme.
    """
    data = torch.as_tensor(data, dtype=torch.float32)

    if data.ndim != 2:
        raise ValueError(
            "Les données doivent être un tableau à deux dimensions"
        )

    if torch.isnan(data).any():
        print("Les données contiennent des NaN : remplacement par 0")

        data = torch.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    minimum = data.amin(dim=0, keepdim=True)
    maximum = data.amax(dim=0, keepdim=True)

    difference = maximum - minimum

    # Évite une division par zéro pour les colonnes constantes.
    safe_difference = torch.where(
        difference > 0,
        difference,
        torch.ones_like(difference),
    )

    data = (data - minimum) / safe_difference

    print("Données normalisées")

    return data


def minibatch_slices_iterator(
    length: int,
    batch_size: int,
    ignore_incomplete_batch: bool = False,
) -> Iterator[slice]:
    """
    Produit les slices correspondant aux mini-batchs.
    """
    if length < 0:
        raise ValueError("length doit être positif ou nul")

    if batch_size < 1:
        raise ValueError("batch_size doit être supérieur ou égal à 1")

    start = 0

    while start + batch_size <= length:
        yield slice(start, start + batch_size)
        start += batch_size

    if not ignore_incomplete_batch and start < length:
        yield slice(start, length)


class BatchSlidingWindow:
    """
    Produit des mini-batchs de fenêtres glissantes.

    Chaque fenêtre possède la forme :

        [window_size, ...]

    Chaque mini-batch possède la forme :

        [batch_size, window_size, ...]
    """

    def __init__(
        self,
        array_size: int,
        window_size: int,
        batch_size: int,
        excludes: Tensor | None = None,
        shuffle: bool = False,
        ignore_incomplete_batch: bool = False,
    ):
        if window_size < 1:
            raise ValueError(
                "window_size doit être supérieur ou égal à 1"
            )

        if array_size < window_size:
            raise ValueError(
                "array_size doit être supérieur ou égal à window_size"
            )

        if batch_size < 1:
            raise ValueError(
                "batch_size doit être supérieur ou égal à 1"
            )

        # Les indices possibles représentent les fins des fenêtres.
        window_end_indices = torch.arange(
            window_size - 1,
            array_size,
            dtype=torch.long,
        )

        if excludes is not None:
            excludes = torch.as_tensor(
                excludes,
                dtype=torch.bool,
            )

            if excludes.shape != (array_size,):
                raise ValueError(
                    "excludes doit avoir la forme "
                    f"({array_size},), reçu {tuple(excludes.shape)}"
                )

            # Forme :
            # [nombre_de_fenêtres, window_size]
            exclusion_windows = excludes.unfold(
                dimension=0,
                size=window_size,
                step=1,
            )

            # Une fenêtre est valide si aucun de ses points n'est exclu.
            valid_windows = ~exclusion_windows.any(dim=1)

            window_end_indices = window_end_indices[valid_windows]

        self._indices = window_end_indices

        self._offsets = torch.arange(
            -window_size + 1,
            1,
            dtype=torch.long,
        )

        self._array_size = array_size
        self._window_size = window_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def __len__(self) -> int:
        """
        Retourne le nombre de fenêtres valides.
        """
        return len(self._indices)

    def get_iterator(
        self,
        arrays: Iterable[Tensor],
    ) -> Iterator[tuple[Tensor, ...]]:
        """
        Produit les fenêtres pour chaque tenseur fourni.
        """
        arrays = tuple(torch.as_tensor(array) for array in arrays)

        if not arrays:
            raise ValueError("arrays ne doit pas être vide")

        for array in arrays:
            if len(array) != self._array_size:
                raise ValueError(
                    "Tous les tenseurs doivent avoir une première "
                    f"dimension égale à {self._array_size}"
                )

        indices = self._indices

        if self._shuffle:
            permutation = torch.randperm(len(indices))
            indices = indices[permutation]

        for batch_slice in minibatch_slices_iterator(
            length=len(indices),
            batch_size=self._batch_size,
            ignore_incomplete_batch=self._ignore_incomplete_batch,
        ):
            batch_end_indices = indices[batch_slice].unsqueeze(dim=1)

            # [batch_size, 1] + [window_size]
            # -> [batch_size, window_size]
            window_indices = batch_end_indices + self._offsets

            windows = tuple(
                array[window_indices.to(array.device)]
                for array in arrays
            )

            yield windows