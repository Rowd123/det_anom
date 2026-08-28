# evaluation.py

from typing import Optional

import numpy as np
from torch import Tensor

from omni_anomaly.spot import SPOT


def to_numpy_1d(values) -> np.ndarray:
    """
    Convertit une liste, un tableau NumPy ou un Tensor PyTorch
    en tableau NumPy unidimensionnel.
    """
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()

    return np.asarray(values).reshape(-1)


def calc_point2point(
    predict,
    actual,
) -> tuple[float, float, float, int, int, int, int]:
    """
    Calcule les métriques de classification point par point.
    """
    predict = to_numpy_1d(predict).astype(bool)
    actual = to_numpy_1d(actual).astype(bool)

    if len(predict) != len(actual):
        raise ValueError(
            "predict et actual doivent avoir la même longueur"
        )

    true_positive = np.logical_and(
        predict,
        actual,
    ).sum()

    true_negative = np.logical_and(
        ~predict,
        ~actual,
    ).sum()

    false_positive = np.logical_and(
        predict,
        ~actual,
    ).sum()

    false_negative = np.logical_and(
        ~predict,
        actual,
    ).sum()

    precision = true_positive / max(
        true_positive + false_positive,
        1,
    )

    recall = true_positive / max(
        true_positive + false_negative,
        1,
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )

    return (
        float(f1),
        float(precision),
        float(recall),
        int(true_positive),
        int(true_negative),
        int(false_positive),
        int(false_negative),
    )


def adjust_predicts(
    score,
    label,
    threshold: Optional[float] = None,
    pred=None,
    calc_latency: bool = False,
):
    """
    Ajuste les prédictions au niveau des segments anormaux.

    Dès qu'un point d'un segment anormal est détecté, le segment
    entier est marqué comme anomalie.

    Le code suppose qu'un score faible est anormal :

        score < threshold
    """
    score = to_numpy_1d(score)
    actual = to_numpy_1d(label) > 0.1

    if len(score) != len(actual):
        raise ValueError(
            "score et label doivent avoir la même longueur"
        )

    if pred is None:
        if threshold is None:
            raise ValueError(
                "threshold est obligatoire lorsque pred n'est pas fourni"
            )

        predict = score < threshold
    else:
        predict = to_numpy_1d(pred).astype(bool).copy()

        if len(predict) != len(actual):
            raise ValueError(
                "pred et label doivent avoir la même longueur"
            )

    latency = 0
    anomaly_count = 0
    anomaly_state = False

    for index in range(len(score)):
        # Premier point détecté dans un segment réellement anormal.
        if (
            actual[index]
            and predict[index]
            and not anomaly_state
        ):
            anomaly_state = True
            anomaly_count += 1

            # Revenir jusqu'au début du segment anormal.
            for previous_index in range(index, -1, -1):
                if not actual[previous_index]:
                    break

                if not predict[previous_index]:
                    predict[previous_index] = True
                    latency += 1

        elif not actual[index]:
            anomaly_state = False

        # Une fois le segment détecté, marquer sa suite.
        if anomaly_state:
            predict[index] = True

    if calc_latency:
        average_latency = (
            latency / anomaly_count
            if anomaly_count > 0
            else 0.0
        )

        return predict, average_latency

    return predict


def calc_seq(
    score,
    label,
    threshold: float,
    calc_latency: bool = False,
):
    """
    Calcule les métriques pour un seuil donné.
    """
    if calc_latency:
        predict, latency = adjust_predicts(
            score=score,
            label=label,
            threshold=threshold,
            calc_latency=True,
        )

        metrics = list(
            calc_point2point(predict, label)
        )

        metrics.append(latency)

        return metrics

    predict = adjust_predicts(
        score=score,
        label=label,
        threshold=threshold,
    )

    return calc_point2point(predict, label)


def bf_search(
    score,
    label,
    start: float,
    end: Optional[float] = None,
    step_num: int = 1,
    display_freq: int = 1,
    verbose: bool = True,
):
    """
    Recherche le seuil donnant le meilleur score F1.
    """
    if end is None:
        end = start

    if step_num < 1:
        raise ValueError(
            "step_num doit être supérieur ou égal à 1"
        )

    thresholds = np.linspace(
        start=start,
        stop=end,
        num=step_num,
        endpoint=True,
    )

    best_metrics = None
    best_threshold = float(thresholds[0])

    if verbose:
        print(f"Intervalle de recherche : [{start}, {end}]")

    for index, threshold in enumerate(thresholds):
        metrics = calc_seq(
            score=score,
            label=label,
            threshold=float(threshold),
            calc_latency=True,
        )

        if (
            best_metrics is None
            or metrics[0] > best_metrics[0]
        ):
            best_metrics = metrics
            best_threshold = float(threshold)

        if verbose and index % display_freq == 0:
            print(
                f"seuil courant : {threshold:.6f} | "
                f"métriques : {metrics} | "
                f"meilleur seuil : {best_threshold:.6f}"
            )

    return best_metrics, best_threshold


def pot_eval(
    init_score,
    score,
    label,
    q: float = 1e-3,
    level: float = 0.02,
) -> dict[str, float]:
    """
    Évalue les scores d'anomalie avec SPOT.

    init_score :
        Scores du jeu d'entraînement.

    score :
        Scores du jeu de test.

    label :
        Étiquettes réelles du jeu de test.
    """
    init_score = to_numpy_1d(init_score)
    score = to_numpy_1d(score)
    label = to_numpy_1d(label)

    spot = SPOT(q=q)

    spot.fit(
        init_data=init_score,
        data=score,
    )

    # min_extrema=True car, une faible log-probabilité est anormale.
    spot.initialize(
        level=level,
        min_extrema=True,
        verbose=False,
    )

    result = spot.run(
        dynamic=False,
    )

    thresholds = np.asarray(
        result["thresholds"],
        dtype=np.float64,
    )

    if thresholds.size == 0:
        raise RuntimeError(
            "SPOT n'a produit aucun seuil"
        )

    # SPOT a inversé les données avec min_extrema=True.
    pot_threshold = -float(thresholds.mean())

    predict, latency = adjust_predicts(
        score=score,
        label=label,
        threshold=pot_threshold,
        calc_latency=True,
    )

    metrics = calc_point2point(
        predict,
        label,
    )

    return {
        "pot-f1": metrics[0],
        "pot-precision": metrics[1],
        "pot-recall": metrics[2],
        "pot-TP": metrics[3],
        "pot-TN": metrics[4],
        "pot-FP": metrics[5],
        "pot-FN": metrics[6],
        "pot-threshold": pot_threshold,
        "pot-latency": float(latency),
    }