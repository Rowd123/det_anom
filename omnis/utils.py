from collections.abc import Iterable, Iterator

import torch
from torch import Tensor

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIMENSIONS = {
    "SMAP": 25,
    "MSL": 55,
    "PMU": 14
}


def notre_data_tri(data_file_path):
    df = pd.read_csv(data_file_path, sep=";")

    df["DATE_MESURE"] = pd.to_datetime(df["DATE_MESURE"])

    df.set_index("DATE_MESURE", inplace=True)

    flag_cols = [c for c in df.columns if c.endswith("QF")]

    duplicated_mask = df.index.duplicated(keep="first")
    flagged_df = df.loc[:, flag_cols].ne("NORMAL").any(axis=1).astype("bool")

    excluded = (duplicated_mask |
                flagged_df |
                df.isna().any(axis=1)
                )

    print(excluded.shape)

    cols_to_drop = [col for col in df.columns if
                    not col.endswith("MAG") and
                    not col.endswith("FREQ") and
                    not col.endswith("DFDT") and
                    not col.endswith("ANG")]

    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    for col in df.columns:
        if col.endswith("MAG") or not col.endswith("ANG"):
            df.rename(columns={col: col[-6:]}, inplace=True)

        else:
            df.rename(columns={col: col[-4:]}, inplace=True)

    return df.values, excluded


def to_numpy(data) -> np.ndarray:
    """
    Convertit un Tenseur PyTorch ou un autre tableau
    en numpy
    """
    if isinstance(data, Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def save_z(
        z,
        filename: str | Path = "z",
        time_step: int = 20,
) -> None:
    """
    Sauvegarde les représentations latentes dans des fichiers texte.

    Forme attendue :
        [nombre_de_fenêtre, window_length, latent_dim]

    Un fichier est créé tous les 'time_step' instants, ainsi que pour
    le dernier instant.
    """
    if time_step < 1:
        raise ValueError(
            "time_step doit être supérieur ou égal à 1"
        )

    z = to_numpy(z)

    if z.ndim != 3:
        raise ValueError(
            "z doit avoir la forme "
            "[nombre_de_fenêtres, window_length, latent_dim]"
        )
    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_length = z.shape[1]

    # Ajouter tous les multiples de time_step
    time_indices = set(
        range(0, window_length, time_step)
    )

    # Toujours ajouter le dernier instant
    time_indices.add(window_length - 1)

    for time_index in sorted(time_indices):
        output_path = filename.parent / (
            f"{filename.name}_{time_index}.txt"
        )

        # z[:, time_index; :] :
        # [nombre_de_fenêtre, latent_dim]
        np.savetxt(
            output_path,
            z[:, time_index, :],
            fmt="%.6f",
        )


def get_data_dim(dataset: str) -> int:
    """
    Retourne le nombre de variables du dataset.
    """
    if dataset in DATA_DIMENSIONS:
        return DATA_DIMENSIONS[dataset]

    if dataset.startswith("machine"):
        return 38

    raise ValueError(
        f"Dataset inconnu : {dataset}"
    )


def load_pickle(
        path: Path,
        required: bool = True,
):
    """
    Charge un fichier Pickle.

    Attention : ne charger que des fichiers Pickle fiables.
    """
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"Fichier introuvable : {path}"
            )

        return None

    with path.open("rb") as file:
        return pickle.load(file)


def get_data(
        dataset: str,
        data_directory: str | Path,
        max_train_size: Optional[float] = None,
        max_test_size: Optional[int] = None,
        print_log: bool = True,
        do_preprocess: bool = True,
        train_start: int = 0,
):
    """
    Charge les données d'entraînement et de test.

    Retour
    ------
    (
        (train_data, train_label, train_excludes),
        (test_data, test_label, test_excludes),
    )

    Les données ont la forme :
        [nombre_de_points, x_dim]

    Les masques excludes ont la forme :
        [nombre_de_points]
    """
    if train_start < 0:
        raise ValueError(
            "train_start et test_start doivent être positifs ou nuls"
        )

    if max_train_size is not None and max_train_size <= 0:
        raise ValueError(
            "max_train_size doit être positif ou None"
        )

    if max_test_size is not None and max_test_size < 1:
        raise ValueError(
            "max_test_size doit être positif ou None"
        )

    DATA, excluded = notre_data_tri(data_directory)

    x_dim = get_data_dim(dataset)

    max_train_size = int(DATA.shape[0] * max_train_size)

    train_end = (
        None
        if max_train_size is None
        else train_start + max_train_size
    )

    test_start = train_start + max_train_size

    test_end = (
        None
        if max_test_size is None
        else test_start + max_test_size
    )

    if print_log:
        print(f"Chargement du dataset : {dataset}")
        print(f"Train : [{train_start}:{train_end}]")
        print(f"Test  : [{test_start}:{test_end}]")

    train_data = DATA[train_start:train_end]
    train_exclude = excluded[train_start:train_end]
    test_data = DATA[test_start:test_end]
    test_exclude = excluded[test_start:test_end]
    test_label = None

    # ------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------
    """
    train_path = (
        data_directory
        / f"{dataset}_train.pkl"
    )

    train_data = load_pickle(
        train_path,
        required=True,
    )

    train_data = np.asarray(
        train_data,
        dtype=np.float32,
    ).reshape(-1, x_dim)

    train_data = train_data[
        train_start:train_end
    ]

    # ------------------------------------------------------------
    # Test
    # ------------------------------------------------------------

    test_path = (
        data_directory
        / f"{dataset}_test.pkl"
    )

    test_data = load_pickle(
        test_path,
        required=False,
    )

    if test_data is not None:
        test_data = np.asarray(
            test_data,
            dtype=np.float32,
        ).reshape(-1, x_dim)

        test_data = test_data[
            test_start:test_end
        ]

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------

    test_label_path = (
        data_directory
        / f"{dataset}_test_label.pkl"
    )

    test_label = load_pickle(
        test_label_path,
        required=False,
    )

    if test_label is not None:
        test_label = np.asarray(
            test_label,
        ).reshape(-1)

        test_label = test_label[
            test_start:test_end
        ]

        test_label = torch.as_tensor(
            test_label,
            dtype=torch.bool,
        )

    """

    # ------------------------------------------------------------
    # Prétraitement
    # ------------------------------------------------------------

    if do_preprocess:
        train_data = preprocess(
            train_data,
        )

        if test_data is not None:
            test_data = preprocess(
                test_data,
            )
        else:
            test_exclude = None

    else:
        train_data = torch.as_tensor(
            train_data,
            dtype=torch.float32,
        )

        train_exclude = torch.zeros(
            len(train_data),
            dtype=torch.bool,
        )

        if test_data is not None:
            test_data = torch.as_tensor(
                test_data,
                dtype=torch.float32,
            )

            test_exclude = torch.zeros(
                len(test_data),
                dtype=torch.bool,
            )
        else:
            test_exclude = None

    if print_log:
        print(
            "Train shape :",
            tuple(train_data.shape),
        )

        print(
            "Test shape :",
            (
                tuple(test_data.shape)
                if test_data is not None
                else None
            ),
        )

        print(
            "Labels shape :",
            (
                tuple(test_label.shape)
                if test_label is not None
                else None
            ),
        )

        print(
            "Fenêtres exclues du train :",
            int(train_exclude.sum()),
        )

        if test_exclude is not None:
            print(
                "Fenêtres exclues du test :",
                int(test_exclude.sum()),
            )

    return (
        train_data,
        None,
        train_exclude,
    ), (
        test_data,
        test_label,
        test_exclude,
    )


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
