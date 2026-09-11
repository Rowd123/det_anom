# main.py

import json
import logging
import pickle
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
from utils import get_data, get_data_dim, save_z


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
        _feature_representation,
    ) = get_data(
        dataset=config.dataset,
        data_directory=config.data_dir,
        max_train_size=config.max_train_size,
        max_test_size=config.max_test_size,
        train_start=config.train_start,
        return_metadata=True,
    )

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
    )

    predictor = Predictor(
        model=model,
        n_z=config.test_n_z,
        batch_size=config.test_batch_size,
        last_point_only=True,
        device=config.device,
        scaler=trainer.scaler,
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

        trainer.load_checkpoint(checkpoint_path)

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
            checkpoint_path=(
                save_directory / "best_omni_anomaly.pt"
                if save_directory is not None
                else None
            ),
        )

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

    # x_dim dépend du dataset sélectionné.
    config.x_dim = get_data_dim(
        config.dataset
    )

    print("=" * 30 + " Configuration " + "=" * 30)
    pprint(vars(config))

    # Conserver une copie de la configuration réellement utilisée.
    result_directory = Path(config.result_dir)
    result_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        result_directory / "config.yaml"
    ).open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            vars(config),
            file,
            sort_keys=False,
        )



    torch.manual_seed(42)
    np.random.seed(42)

    main(config)
