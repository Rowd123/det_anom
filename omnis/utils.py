from collections.abc import Iterable, Iterator

import torch
from torch import Tensor

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIMENSIONS = {
    "SMAP": 25,
    "MSL": 55,
    "PMU": 14
}


@dataclass(frozen=True)
class PhasorFeatureMapping:
    amplitude_name: str
    phase_name: str
    real_name: str
    imag_name: str
    amplitude_index: int
    phase_index: int


@dataclass(frozen=True)
class FeatureRepresentation:
    """Correspondance stable entre les variables physiques et celles du modèle."""

    original_names: tuple[str, ...]
    model_names: tuple[str, ...]
    phasors: tuple[PhasorFeatureMapping, ...]


PMU_FIRST_PHASOR_INDEX = 2


def _readable_feature_name(name: str) -> str:
    return name.strip().replace(" ", "_").lower()


def _phasor_name(amplitude_name: str, phase_name: str) -> str:
    """Produit un nom lisible sans utiliser les noms pour décider des couples."""
    if amplitude_name.endswith("MAG"):
        return amplitude_name[:-len("MAG")].rstrip(" _.-")

    common_length = 0
    for amplitude_character, phase_character in zip(amplitude_name, phase_name):
        if amplitude_character != phase_character:
            break
        common_length += 1
    common_name = amplitude_name[:common_length].rstrip(" _.-")
    return common_name or amplitude_name


def build_feature_representation(column_names) -> FeatureRepresentation:
    """Décrit tous les couples PMU (2, 3), (4, 5), ..., dans l'ordre du CSV."""
    original_names = tuple(str(name) for name in column_names)
    phasor_feature_count = len(original_names) - PMU_FIRST_PHASOR_INDEX
    if phasor_feature_count < 0 or phasor_feature_count % 2:
        raise ValueError(
            "Les variables PMU à partir de l'indice 2 doivent former des "
            "couples amplitude/phase"
        )

    model_names = [_readable_feature_name(name) for name in original_names]
    phasors = []
    for amplitude_index in range(
        PMU_FIRST_PHASOR_INDEX,
        len(original_names),
        2,
    ):
        phase_index = amplitude_index + 1
        amplitude_name = original_names[amplitude_index]
        phase_name = original_names[phase_index]
        readable_stem = _readable_feature_name(
            _phasor_name(amplitude_name, phase_name)
        )
        mapping = PhasorFeatureMapping(
            amplitude_name=amplitude_name,
            phase_name=phase_name,
            real_name=f"{readable_stem}_real",
            imag_name=f"{readable_stem}_imag",
            amplitude_index=amplitude_index,
            phase_index=phase_index,
        )
        phasors.append(mapping)
        model_names[amplitude_index] = mapping.real_name
        model_names[phase_index] = mapping.imag_name
    return FeatureRepresentation(
        original_names=original_names,
        model_names=tuple(model_names),
        phasors=tuple(phasors),
    )


def phasor_to_cartesian(data, representation: FeatureRepresentation):
    """Convertit les MAG/ANG physiques (degrés) en Re/Im physiques."""
    is_tensor = isinstance(data, Tensor)
    values = torch.as_tensor(data).clone() if is_tensor else np.array(data, copy=True)
    for phasor in representation.phasors:
        if is_tensor:
            amplitude = values[..., phasor.amplitude_index].clone()
            phase = values[..., phasor.phase_index].clone()
            radians = torch.deg2rad(phase)
            real, imag = amplitude * torch.cos(radians), amplitude * torch.sin(radians)
        else:
            amplitude = values[..., phasor.amplitude_index].copy()
            phase = values[..., phasor.phase_index].copy()
            radians = np.deg2rad(phase)
            real, imag = amplitude * np.cos(radians), amplitude * np.sin(radians)
        values[..., phasor.amplitude_index] = real
        values[..., phasor.phase_index] = imag
    return values


def cartesian_to_phasor(data, representation: FeatureRepresentation):
    """Restaure amplitude et phase en degrés dans [-180, 180]."""
    is_tensor = isinstance(data, Tensor)
    values = torch.as_tensor(data).clone() if is_tensor else np.array(data, copy=True)
    for phasor in representation.phasors:
        if is_tensor:
            real = values[..., phasor.amplitude_index].clone()
            imag = values[..., phasor.phase_index].clone()
            amplitude = torch.sqrt(real.square() + imag.square())
            phase = torch.rad2deg(torch.atan2(imag, real))
        else:
            real = values[..., phasor.amplitude_index].copy()
            imag = values[..., phasor.phase_index].copy()
            amplitude = np.sqrt(np.square(real) + np.square(imag))
            phase = np.rad2deg(np.arctan2(imag, real))
        values[..., phasor.amplitude_index] = amplitude
        values[..., phasor.phase_index] = phase
    return values


def inverse_transform_physical(data, scaler, representation: FeatureRepresentation):
    """Annule le scaling cartésien avant de revenir aux grandeurs physiques."""
    return cartesian_to_phasor(scaler.inverse_transform(data), representation)


class MinMaxScaler:
    """Scaler Min-Max ajusté uniquement sur des observations valides."""

    def __init__(self):
        self.minimum = None
        self.maximum = None

    @property
    def fitted(self) -> bool:
        return self.minimum is not None and self.maximum is not None

    def fit(self, data, excludes=None):
        data = torch.as_tensor(data, dtype=torch.float32)
        if data.ndim != 2:
            raise ValueError("data doit avoir la forme [temps, variables]")
        valid_rows = torch.isfinite(data).all(dim=1)
        if excludes is not None:
            excludes = torch.as_tensor(
                np.asarray(excludes),
                dtype=torch.bool,
                device=data.device,
            )
            if excludes.shape != (len(data),):
                raise ValueError(f"excludes doit avoir la forme ({len(data)},)")
            valid_rows &= ~excludes
        if not valid_rows.any():
            raise ValueError("Aucune observation valide pour ajuster le scaler")
        valid_data = data[valid_rows]
        self.minimum = valid_data.amin(dim=0, keepdim=True)
        self.maximum = valid_data.amax(dim=0, keepdim=True)
        return self

    def transform(self, data) -> Tensor:
        self._check_fitted()
        data = torch.as_tensor(data, dtype=torch.float32)
        minimum, maximum = self._parameters_for(data)
        difference = maximum - minimum
        scale = torch.where(difference > 0, difference, torch.ones_like(difference))
        return torch.nan_to_num((data - minimum) / scale)

    def inverse_transform(self, data) -> Tensor:
        self._check_fitted()
        data = torch.as_tensor(data, dtype=torch.float32)
        minimum, maximum = self._parameters_for(data)
        difference = maximum - minimum
        scale = torch.where(difference > 0, difference, torch.ones_like(difference))
        return data * scale + minimum

    def state_dict(self) -> dict[str, Tensor]:
        self._check_fitted()
        return {
            "minimum": self.minimum.detach().cpu().clone(),
            "maximum": self.maximum.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict[str, Tensor]):
        # Le scaler reste sur CPU dans le checkpoint. transform() déplace ses
        # paramètres vers le device de l'entrée au moment de leur utilisation.
        self.minimum = torch.as_tensor(
            state["minimum"], dtype=torch.float32, device="cpu"
        )
        self.maximum = torch.as_tensor(
            state["maximum"], dtype=torch.float32, device="cpu"
        )
        return self

    def _parameters_for(self, data: Tensor) -> tuple[Tensor, Tensor]:
        if data.ndim < 1 or data.shape[-1] != self.minimum.shape[-1]:
            raise ValueError(
                "La dernière dimension des données ne correspond pas au scaler : "
                f"{data.shape[-1] if data.ndim else None} reçu, "
                f"{self.minimum.shape[-1]} attendu"
            )
        return (
            self.minimum.to(device=data.device, dtype=data.dtype),
            self.maximum.to(device=data.device, dtype=data.dtype),
        )

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Le scaler doit être ajusté ou restauré avant utilisation")


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

    representation = build_feature_representation(df.columns)
    data = df.to_numpy(dtype=np.float64)

    # Les valeurs non finies ne doivent ni ajuster le scaler ni entrer dans
    # une fenêtre d'entraînement, même si elles ne sont pas des NaN pandas.
    finite_rows = np.isfinite(data).all(axis=1)
    excluded = np.asarray(excluded, dtype=bool) | ~finite_rows

    # Aucune substitution : si MAG ou ANG est invalide, Re et Im le restent et
    # la ligne demeure exclue par le masque calculé ci-dessus.
    data = phasor_to_cartesian(data, representation)
    excluded |= ~np.isfinite(data).all(axis=1)

    return data, excluded, representation


def get_data_simple(data_file_path):
    """Charge un CSV PMU avec exactement la même représentation que get_data."""
    return notre_data_tri(data_file_path)


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
        return_metadata: bool = False,
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

    DATA, excluded, representation = get_data_simple(data_directory)

    x_dim = get_data_dim(dataset)
    if DATA.ndim != 2 or DATA.shape[1] != x_dim:
        raise ValueError(
            f"Le dataset {dataset} doit contenir {x_dim} variables, "
            "mais le fichier en contient "
            f"{DATA.shape[1] if DATA.ndim == 2 else 'un nombre invalide'}"
        )

    if max_train_size is None:
        train_size = DATA.shape[0] - train_start
    elif isinstance(max_train_size, float) and max_train_size <= 1.0:
        train_size = int(DATA.shape[0] * max_train_size)
    else:
        train_size = int(max_train_size)

    if train_start + train_size > DATA.shape[0]:
        raise ValueError(
            "La tranche d'entraînement dépasse la taille du dataset"
        )

    train_end = (
        None
        if max_train_size is None
        else train_start + train_size
    )

    test_start = train_start + train_size

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
    if len(test_data) == 0:
        test_data = None
        test_exclude = None
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

    # Le scaler est ajusté après le split temporel train/validation dans
    # Trainer.fit. get_data conserve donc les valeurs physiques brutes.
    train_data = torch.as_tensor(train_data, dtype=torch.float32)

    if not do_preprocess:
        train_exclude = torch.zeros(len(train_data), dtype=torch.bool)

    if test_data is not None:
        test_data = torch.as_tensor(test_data, dtype=torch.float32)

        if not do_preprocess:
            test_exclude = torch.zeros(
                len(test_data),
                dtype=torch.bool,
            )

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

    result = (
        train_data,
        None,
        train_exclude,
    ), (
        test_data,
        test_label,
        test_exclude,
    )
    if return_metadata:
        return (*result, representation)
    return result


def preprocess(
    data,
    minimum: Optional[Tensor] = None,
    maximum: Optional[Tensor] = None,
    return_scaler: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
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

    fit_scaler = minimum is None and maximum is None
    if (minimum is None) != (maximum is None):
        raise ValueError("minimum et maximum doivent être fournis ensemble")

    if fit_scaler:
        minimum = data.amin(dim=0, keepdim=True)
        maximum = data.amax(dim=0, keepdim=True)
    else:
        minimum = torch.as_tensor(minimum, dtype=data.dtype, device=data.device)
        maximum = torch.as_tensor(maximum, dtype=data.dtype, device=data.device)

        expected_shape = (1, data.shape[1])
        if minimum.shape != expected_shape or maximum.shape != expected_shape:
            raise ValueError(
                "minimum et maximum doivent avoir la forme "
                f"{expected_shape}"
            )

    difference = maximum - minimum

    # Évite une division par zéro pour les colonnes constantes.
    safe_difference = torch.where(
        difference > 0,
        difference,
        torch.ones_like(difference),
    )

    data = (data - minimum) / safe_difference

    print("Données normalisées")

    if return_scaler:
        return data, minimum, maximum

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
