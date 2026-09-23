# main.py

import hashlib
import json
import logging
import pickle
import re
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from eval import bf_search, pot_eval
from model import OmniAnomaly
from prediction import Predictor
from trainer import Trainer
from utils import build_normalizer, get_data, save_z


def _filename_token(value) -> str:
    """Convertit une valeur de configuration en composante de nom portable."""
    token = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value)).strip("-_.")
    return token.lower() or "none"


def best_checkpoint_filename(config, feature_representation) -> str:
    """Nomme le meilleur modèle à partir des paramètres qui le rendent incompatible.

    La liste complète des features est représentée par une empreinte stable afin
    de garder un nom lisible même lorsque beaucoup de grandeurs sont utilisées.
    Les exclusions restent visibles pour faciliter les comparaisons de runs.
    """
    feature_names = "\n".join(feature_representation.model_names).encode("utf-8")
    feature_digest = hashlib.sha256(feature_names).hexdigest()[:8]
    excluded = sorted(
        (_filename_token(name) for name in getattr(config, "excluded_features", []))
    )
    if getattr(config, "exclude_zero_sequence_phase", False):
        excluded.append("zero-sequence-phase")
    exclusion_token = "+".join(excluded) if excluded else "none"
    if len(exclusion_token) > 64:
        exclusion_digest = hashlib.sha256(exclusion_token.encode("utf-8")).hexdigest()[:8]
        exclusion_token = f"multiple-{exclusion_digest}"

    parts = (
        "best_omni_anomaly",
        f"norm-{_filename_token(getattr(config, 'normalization_mode', 'train_minmax'))}",
        f"features-{feature_representation.model_dim}-{feature_digest}",
        f"excl-{exclusion_token}",
        f"window-{config.window_length}",
        f"z-{config.z_dim}",
        f"rnn-{_filename_token(config.rnn_cell)}-{config.rnn_num_hidden}",
        f"decoder-{_filename_token(config.decoder_sigma_mode)}",
    )
    return "__".join(parts) + ".pt"


def load_config(path: str | Path) -> SimpleNamespace:
    """
    Charge un fichier YAML et permet l'accès avec config.parametre.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file)

    if not isinstance(values, dict):
        raise ValueError(
            "Le contenu du fichier YAML doit être un dictionnaire"
        )

    return SimpleNamespace(**values)


def save_pickle(
    value,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("wb") as file:
        pickle.dump(value, file)


def save_metrics(
    metrics: dict,
    path: Path,
) -> None:
    """
    Enregistre les métriques dans un fichier JSON.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=2,
            default=lambda value: (
                value.item()
                if hasattr(value, "item")
                else str(value)
            ),
        )


def main(config) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s [%(levelname)s] "
            "%(name)s: %(message)s"
        ),
    )

    result_directory = Path(config.result_dir)
    result_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_directory = (
        Path(config.save_dir)
        if config.save_dir is not None
        else None
    )

    if save_directory is not None:
        save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------
    # Données
    # ------------------------------------------------------------

    (
        (x_train, _, x_train_exclude),
        (x_test, x_label, x_test_exclude),
        feature_representation,
    ) = get_data(
        dataset=config.dataset,
        data_directory=config.data_dir,
        max_train_size=config.max_train_size,
        max_test_size=config.max_test_size,
        train_start=config.train_start,
        return_metadata=True,
        exclude_zero_sequence_phase=config.exclude_zero_sequence_phase,
        data_column_order=config.data_column_order,
        excluded_features=getattr(config, "excluded_features", []),
    )
    config.x_dim = feature_representation.model_dim
    config.normalization_mode = getattr(config, "normalization_mode", "train_minmax")
    normalizer = build_normalizer(
        config.normalization_mode,
        feature_representation,
        getattr(config, "nominal_normalization", None),
    )
    best_checkpoint_path = (
        save_directory / best_checkpoint_filename(config, feature_representation)
        if save_directory is not None else None
    )
    if best_checkpoint_path is not None:
        print(f"Best checkpoint path: {best_checkpoint_path}")

    print(f"Physical dimension: {feature_representation.physical_dim}")
    print(f"Model dimension: {feature_representation.model_dim}")
    print("\nPhysical input order:")
    for index, feature_name in enumerate(feature_representation.original_names):
        print(f"{index}  {feature_name}")
    print("\nPhasor mapping:")
    for phasor in feature_representation.phasors:
        sequence = (
            f"sequence {phasor.sequence}"
            if phasor.sequence is not None
            else "unclassified"
        )
        phase_status = "phase included" if phasor.includes_phase else "phase excluded"
        print(
            f"- {sequence}: phase={phasor.phase_name}, "
            f"magnitude={phasor.amplitude_name}, {phase_status}"
        )
    print("\nModel features:")
    for feature_name in feature_representation.model_names:
        print(f"- {feature_name}")
    print("\nExcluded from model representation:")
    if feature_representation.excluded_model_names:
        for feature_name in feature_representation.excluded_model_names:
            print(f"- {feature_name}")
    else:
        print("- none")

    # Conserver la configuration réellement utilisée, dimension dérivée incluse.
    with (result_directory / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(vars(config), file, sort_keys=False)

    # ------------------------------------------------------------
    # Modèle
    # ------------------------------------------------------------

    model = OmniAnomaly(
        config=config,
    )

    trainer = Trainer(
        model=model,
        train_n_z=config.train_n_z,
        valid_n_z=config.valid_n_z,
        max_epoch=config.max_epoch,
        max_step=config.max_step,
        batch_size=config.batch_size,
        valid_batch_size=config.valid_batch_size,
        train_log_step_freq=config.train_log_step_freq,
        deep_training_diagnostics=config.deep_training_diagnostics,
        initial_lr=config.initial_lr,
        lr_anneal_epochs=config.lr_anneal_epoch_freq,
        lr_anneal_factor=config.lr_anneal_factor,
        grad_clip_norm=config.gradient_clip_norm,
        weight_decay=config.l2_reg,
        early_stopping_validation_patience=(
            config.early_stopping_validation_patience
            if config.early_stop
            else config.max_epoch + 1
        ),
        anomaly_filter_enabled=config.anomaly_filter_enabled,
        anomaly_filter_warmup_epochs=config.anomaly_filter_warmup_epochs,
        anomaly_filter_k=config.anomaly_filter_k,
        anomaly_filter_history_size=config.anomaly_filter_history_size,
        anomaly_filter_min_history=config.anomaly_filter_min_history,
        device=config.device,
        feature_names=feature_representation.model_names,
        normalizer=normalizer,
    )

    predictor = Predictor(
        model=model,
        n_z=config.test_n_z,
        batch_size=config.test_batch_size,
        last_point_only=True,
        device=config.device,
        normalizer=trainer.normalizer,
    )

    metrics = {}

    # ------------------------------------------------------------
    # Restauration
    # ------------------------------------------------------------

    valid_modes = {"scratch", "resume", "evaluate"}
    if config.run_mode not in valid_modes:
        raise ValueError(f"run_mode doit appartenir à {sorted(valid_modes)}")

    if config.run_mode in {"resume", "evaluate"}:
        if config.restore_checkpoint is None:
            raise ValueError(
                "restore_checkpoint est obligatoire en mode resume ou evaluate"
            )
        checkpoint_path = Path(config.restore_checkpoint)

        trainer.load_checkpoint(checkpoint_path, learning_rate=confg.initial_lr)

        print(
            f"Modèle restauré depuis {checkpoint_path}"
        )

    # ------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------

    if config.run_mode in {"scratch", "resume"} and config.max_epoch > 0:
        training_start = perf_counter()

        training_metrics = trainer.fit(
            values=x_train,
            valid_portion=config.valid_portion,
            excludes=x_train_exclude,
            checkpoint_path=best_checkpoint_path,
        )
        if best_checkpoint_path is not None:
            metrics["best_checkpoint"] = str(best_checkpoint_path)

        training_duration = (
            perf_counter() - training_start
        )

        metrics.update(training_metrics)
        metrics["train_total_time"] = training_duration

    # ------------------------------------------------------------
    # Scores d'entraînement pour POT
    # ------------------------------------------------------------

    train_score, train_z, train_prediction_time = (
        predictor.get_score(x_train, x_train_exclude)
    )

    metrics["train_prediction_time"] = (
        train_prediction_time
    )

    if config.train_score_filename is not None:
        save_pickle(
            train_score,
            result_directory
            / config.train_score_filename,
        )

    if config.save_z:
        save_z(
            train_z,
            "train_z",
        )

    # ------------------------------------------------------------
    # Test
    # ------------------------------------------------------------

    if x_test is not None:
        testing_start = perf_counter()

        test_score, test_z, prediction_time = (
            predictor.get_score(x_test, x_test_exclude)
        )

        total_test_time = (
            perf_counter() - testing_start
        )

        metrics.update({
            "pred_time": prediction_time,
            "pred_total_time": total_test_time,
        })

        if config.save_z:
            save_z(
                test_z,
                "test_z",
            )

        if config.test_score_filename is not None:
            save_pickle(
                test_score,
                result_directory
                / config.test_score_filename,
            )

        # --------------------------------------------------------
        # Évaluation supervisée
        # --------------------------------------------------------

        if (
            x_label is not None
            and len(x_label) >= len(test_score)
        ):
            labels = x_label[-len(test_score):]

            if config.get_score_on_dim:
                test_score = np.sum(
                    test_score,
                    axis=-1,
                )

                train_score = np.sum(
                    train_score,
                    axis=-1,
                )

            number_of_search_steps = int(
                abs(
                    config.bf_search_max
                    - config.bf_search_min
                )
                / config.bf_search_step_size
            )

            best_result, best_threshold = bf_search(
                score=test_score,
                label=labels,
                start=config.bf_search_min,
                end=config.bf_search_max,
                step_num=number_of_search_steps,
                display_freq=50,
            )

            pot_result = pot_eval(
                init_score=train_score,
                score=test_score,
                label=labels,
                q=config.pot_q,
                level=config.level,
            )

            metrics.update({
                "best-f1": best_result[0],
                "precision": best_result[1],
                "recall": best_result[2],
                "TP": best_result[3],
                "TN": best_result[4],
                "FP": best_result[5],
                "FN": best_result[6],
                "latency": best_result[-1],
                "threshold": best_threshold,
            })

            metrics.update(pot_result)

    # ------------------------------------------------------------
    # Sauvegarde finale
    # ------------------------------------------------------------

    if save_directory is not None and config.run_mode in {"scratch", "resume"}:
        final_checkpoint_path = (
            save_directory
            / "omni_anomaly.pt"
        )

        trainer.save_checkpoint(
            final_checkpoint_path
        )

        print(
            f"Modèle sauvegardé dans {final_checkpoint_path}"
        )

    metrics_path = (
        result_directory
        / config.metrics_filename
    )

    save_metrics(
        metrics,
        metrics_path,
    )

    print("=" * 30 + " Résultats " + "=" * 30)
    pprint(metrics)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Chemin vers le fichier YAML",
    )

    arguments = parser.parse_args()

    config = load_config(
        arguments.config
    )

    print("GPU available: ", torch.cuda.is_available())

    print("=" * 30 + " Configuration " + "=" * 30)
    pprint(vars(config))

    torch.manual_seed(42)
    np.random.seed(42)

    main(config)
