from collections.abc import Iterable, Iterator

import torch
from torch import Tensor

import pickle
import re
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
    model_amplitude_name: str
    phase_cos_name: str
    phase_sin_name: str
    physical_amplitude_index: int
    physical_phase_index: int
    model_amplitude_index: int
    model_phase_cos_index: Optional[int]
    model_phase_sin_index: Optional[int]
    sequence: Optional[int]

    @property
    def includes_phase(self) -> bool:
        return self.model_phase_cos_index is not None


@dataclass(frozen=True)
class DirectFeatureMapping:
    physical_name: str
    model_name: str
    physical_index: int
    model_index: int


@dataclass(frozen=True)
class FeatureRepresentation:
    """Correspondance stable entre les variables physiques et celles du modèle."""

    original_names: tuple[str, ...]
    model_names: tuple[str, ...]
    direct_features: tuple[DirectFeatureMapping, ...]
    phasors: tuple[PhasorFeatureMapping, ...]
    excluded_model_names: tuple[str, ...]

    @property
    def physical_dim(self) -> int:
        return len(self.original_names)

    @property
    def model_dim(self) -> int:
        return len(self.model_names)


FEATURE_REPRESENTATION_VERSION = "phasor_ang_mag_corrected_v3"
_KIND_PATTERN = re.compile(r"(?i)(ANG|MAG)$")


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


def _phasor_column(name: str) -> tuple[str, str] | None:
    """Retourne (identifiant du phasor, ANG|MAG) depuis le nom explicite."""
    match = _KIND_PATTERN.search(name)
    if match is None:
        return None
    return name[:match.start()].rstrip("._ -"), match.group(1).upper()


_SEQUENCE_PATTERN = re.compile(r"(?:^|[^A-Z0-9])(?:V|I)([012])(?:[^A-Z0-9]|$)", re.I)


def _phasor_sequence(amplitude_name: str, phase_name: str) -> Optional[int]:
    """Identifie explicitement V0/V1/V2/I0/I1/I2 dans les noms du phasor."""
    sequences = {
        int(match.group(1))
        for name in (amplitude_name, phase_name)
        for match in _SEQUENCE_PATTERN.finditer(name)
    }
    if len(sequences) > 1:
        raise ValueError(
            f"Séquences incohérentes pour le couple {amplitude_name!r}, {phase_name!r}"
        )
    return next(iter(sequences), None)


def build_feature_representation(
    column_names,
    exclude_zero_sequence_phase: bool = False,
) -> FeatureRepresentation:
    """Construit les couples PMU uniquement à partir des suffixes ANG et MAG."""
    original_names = tuple(str(name) for name in column_names)
    model_names = []
    direct_features = []
    phasor_columns = {}
    phasor_order = []
    for physical_index, name in enumerate(original_names):
        semantic = _phasor_column(name)
        if semantic is not None:
            stem, kind = semantic
            if stem not in phasor_columns:
                phasor_columns[stem] = {}
                phasor_order.append(stem)
            if kind in phasor_columns[stem]:
                raise ValueError(f"Colonne {kind} dupliquée pour le phasor {stem!r}")
            phasor_columns[stem][kind] = (physical_index, name)
            continue
        model_index = len(model_names)
        model_name = _readable_feature_name(name)
        model_names.append(model_name)
        direct_features.append(DirectFeatureMapping(
            physical_name=name,
            model_name=model_name,
            physical_index=physical_index,
            model_index=model_index,
        ))
    phasors = []
    excluded_model_names = []
    for stem in phasor_order:
        columns = phasor_columns[stem]
        if set(columns) != {"ANG", "MAG"}:
            missing = {"ANG", "MAG"} - set(columns)
            raise ValueError(f"Phasor {stem!r} incomplet, colonne(s) manquante(s): {sorted(missing)}")
        physical_phase_index, phase_name = columns["ANG"]
        physical_amplitude_index, amplitude_name = columns["MAG"]
        readable_stem = _readable_feature_name(
            _phasor_name(amplitude_name, phase_name)
        )
        sequence = _phasor_sequence(amplitude_name, phase_name)
        include_phase = not (exclude_zero_sequence_phase and sequence == 0)
        model_amplitude_index = len(model_names)
        model_phase_cos_index = len(model_names) + 1 if include_phase else None
        model_phase_sin_index = len(model_names) + 2 if include_phase else None
        mapping = PhasorFeatureMapping(
            amplitude_name=amplitude_name,
            phase_name=phase_name,
            model_amplitude_name=f"{readable_stem}_mag",
            phase_cos_name=f"{readable_stem}_phase_cos",
            phase_sin_name=f"{readable_stem}_phase_sin",
            physical_amplitude_index=physical_amplitude_index,
            physical_phase_index=physical_phase_index,
            model_amplitude_index=model_amplitude_index,
            model_phase_cos_index=model_phase_cos_index,
            model_phase_sin_index=model_phase_sin_index,
            sequence=sequence,
        )
        phasors.append(mapping)
        model_names.append(mapping.model_amplitude_name)
        if include_phase:
            model_names.extend((mapping.phase_cos_name, mapping.phase_sin_name))
        else:
            excluded_model_names.extend((mapping.phase_cos_name, mapping.phase_sin_name))
    return FeatureRepresentation(
        original_names=original_names,
        model_names=tuple(model_names),
        direct_features=tuple(direct_features),
        phasors=tuple(phasors),
        excluded_model_names=tuple(excluded_model_names),
    )


def phasor_to_phase_embedding(data, representation: FeatureRepresentation):
    """Développe les données physiques en (amplitude, cos phase, sin phase)."""
    is_tensor = isinstance(data, Tensor)
    if data.shape[-1] != representation.physical_dim:
        raise ValueError("La dernière dimension ne correspond pas aux données physiques")
    shape = (*data.shape[:-1], representation.model_dim)
    values = data.new_empty(shape) if is_tensor else np.empty(shape, dtype=data.dtype)
    for feature in representation.direct_features:
        values[..., feature.model_index] = data[..., feature.physical_index]
    for phasor in representation.phasors:
        amplitude = data[..., phasor.physical_amplitude_index]
        values[..., phasor.model_amplitude_index] = amplitude
        if phasor.includes_phase and is_tensor:
            phase = data[..., phasor.physical_phase_index]
            radians = torch.deg2rad(phase)
            cosine, sine = torch.cos(radians), torch.sin(radians)
        elif phasor.includes_phase:
            phase = data[..., phasor.physical_phase_index]
            radians = np.deg2rad(phase)
            cosine, sine = np.cos(radians), np.sin(radians)
        if phasor.includes_phase:
            values[..., phasor.model_phase_cos_index] = cosine
            values[..., phasor.model_phase_sin_index] = sine
    return values


def phase_embedding_to_phasor(data, representation: FeatureRepresentation):
    """Restaure les 14 variables physiques depuis la représentation modèle."""
    is_tensor = isinstance(data, Tensor)
    if data.shape[-1] != representation.model_dim:
        raise ValueError("La dernière dimension ne correspond pas à la représentation modèle")
    shape = (*data.shape[:-1], representation.physical_dim)
    values = data.new_empty(shape) if is_tensor else np.empty(shape, dtype=data.dtype)
    for feature in representation.direct_features:
        values[..., feature.physical_index] = data[..., feature.model_index]
    for phasor in representation.phasors:
        amplitude = data[..., phasor.model_amplitude_index]
        if phasor.includes_phase and is_tensor:
            cosine = data[..., phasor.model_phase_cos_index]
            sine = data[..., phasor.model_phase_sin_index]
            phase = torch.rad2deg(torch.atan2(sine, cosine))
        elif phasor.includes_phase:
            cosine = data[..., phasor.model_phase_cos_index]
            sine = data[..., phasor.model_phase_sin_index]
            phase = np.rad2deg(np.arctan2(sine, cosine))
        values[..., phasor.physical_amplitude_index] = amplitude
        # Une phase volontairement absente n'est pas reconstructible. Zéro est
        # un placeholder fini et explicite, plutôt qu'une valeur issue du modèle.
        values[..., phasor.physical_phase_index] = phase if phasor.includes_phase else 0.0
    return values


def inverse_transform_physical(data, scaler, representation: FeatureRepresentation):
    """Annule le scaling modèle, puis restaure les grandeurs physiques."""
    return phase_embedding_to_phasor(scaler.inverse_transform(data), representation)


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


def validate_checkpoint_model_dimension(
    checkpoint: dict,
    expected_dim: int,
    expected_feature_names=None,
    expected_representation_version: Optional[str] = None,
) -> None:
    """Refuse explicitement les checkpoints d'une autre représentation."""
    saved_dim = checkpoint.get("model_x_dim")
    scaler_state = checkpoint.get("scaler_state_dict")
    if saved_dim is None and scaler_state is not None and "minimum" in scaler_state:
        saved_dim = int(torch.as_tensor(scaler_state["minimum"]).shape[-1])
    if saved_dim is not None and saved_dim != expected_dim:
        raise ValueError(
            "Checkpoint incompatible : il utilise une représentation de "
            f"dimension {saved_dim}, tandis que le modèle courant attend "
            f"{expected_dim} dimensions (amplitude/cosinus/sinus). "
            "Conservez ce checkpoint séparément et relancez en mode scratch."
        )
    saved_feature_names = checkpoint.get("model_feature_names")
    if (
        saved_feature_names is not None
        and expected_feature_names is not None
        and tuple(saved_feature_names) != tuple(expected_feature_names)
    ):
        raise ValueError("Checkpoint incompatible : les features modèle diffèrent")
    saved_version = checkpoint.get("feature_representation")
    if (
        expected_representation_version is not None
        and saved_version != expected_representation_version
    ):
        raise ValueError(
            "Checkpoint incompatible : la sémantique ANG/MAG a changé "
            f"({saved_version!r} reçu, {expected_representation_version!r} attendu). "
            "Relancez l'entraînement depuis scratch."
        )


def resolve_data_column_order(columns, data_column_order) -> list[str]:
    """Associe sans ambiguïté l'ordre configuré aux noms réels du CSV."""
    available = [str(column) for column in columns]
    resolved = []
    for expected in data_column_order:
        expected_upper = str(expected).upper()
        candidates = [
            actual for actual in available
            if actual.upper() == expected_upper
            or actual.upper().endswith(":" + expected_upper)
        ]
        if len(candidates) != 1:
            qualifier = "manquante" if not candidates else "ambiguë"
            raise ValueError(
                f"Colonne configurée {expected!r} {qualifier}; candidats={candidates}"
            )
        resolved.append(candidates[0])
    if len(set(resolved)) != len(resolved):
        raise ValueError("data_column_order associe plusieurs entrées à la même colonne")
    return resolved


def notre_data_tri(
    data_file_path,
    exclude_zero_sequence_phase: bool = False,
    data_column_order=None,
):
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

    if not data_column_order:
        raise ValueError("data_column_order est obligatoire pour les données PMU")
    selected_columns = resolve_data_column_order(df.columns, data_column_order)
    df = df.loc[:, selected_columns]

    representation = build_feature_representation(
        df.columns,
        exclude_zero_sequence_phase=exclude_zero_sequence_phase,
    )
    data = df.to_numpy(dtype=np.float64)

    # Les valeurs non finies ne doivent ni ajuster le scaler ni entrer dans
    # une fenêtre d'entraînement, même si elles ne sont pas des NaN pandas.
    finite_rows = np.isfinite(data).all(axis=1)
    excluded = np.asarray(excluded, dtype=bool) | ~finite_rows

    # Aucune substitution : si MAG ou ANG est invalide, l'embedding le reste et
    # la ligne demeure exclue par le masque calculé ci-dessus.
    data = phasor_to_phase_embedding(data, representation)
    excluded |= ~np.isfinite(data).all(axis=1)

    return data, excluded, representation


def get_data_simple(
    data_file_path,
    exclude_zero_sequence_phase: bool = False,
    data_column_order=None,
):
    """Charge un CSV PMU avec exactement la même représentation que get_data."""
    return notre_data_tri(
        data_file_path,
        exclude_zero_sequence_phase,
        data_column_order,
    )


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
        exclude_zero_sequence_phase: bool = False,
        data_column_order=None,
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

    DATA, excluded, representation = get_data_simple(
        data_directory,
        exclude_zero_sequence_phase=exclude_zero_sequence_phase,
        data_column_order=data_column_order,
    )

    physical_dim = get_data_dim(dataset)
    if representation.physical_dim != physical_dim:
        raise ValueError(
            f"Le dataset {dataset} doit contenir {physical_dim} variables "
            f"physiques, mais le fichier en contient {representation.physical_dim}"
        )
    if DATA.ndim != 2 or DATA.shape[1] != representation.model_dim:
        raise ValueError("La transformation PMU a produit une dimension invalide")

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
