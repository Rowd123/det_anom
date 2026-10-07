"""Exporter les fenêtres par score Isolation Forest décroissant, sans entraînement."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_anomaly.artifacts import SCORE_CONVENTION, SEGMENT_KEYS, read_table


def rank_windows(scores_path, output_path, *, top_n=None):
    """Conserver les colonnes des scores et ajouter un rang à partir de 1."""
    source, output = Path(scores_path), Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("Le fichier de sortie doit différer du fichier des scores.")
    if output.suffix.lower() != ".csv":
        raise ValueError("La sortie doit être un fichier .csv.")
    if top_n is not None and top_n < 1:
        raise ValueError("--top-n doit être un entier strictement positif.")

    scores, manifest = read_table(source, expected_type="scores")
    if manifest.score_convention != SCORE_CONVENTION:
        raise ValueError("Scores incompatibles : le classement attend -score_samples.")
    required = {*SEGMENT_KEYS, "window_start", "window_end", "score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Colonnes absentes du fichier des scores : {sorted(missing)}")
    if "rank" in scores:
        raise ValueError("La colonne 'rank' existe déjà ; utiliser les scores d'origine.")
    if scores[list(SEGMENT_KEYS)].isna().any().any():
        raise ValueError("Les identifiants des fenêtres ne doivent pas être manquants.")
    if scores.duplicated(list(SEGMENT_KEYS)).any():
        raise ValueError("Le fichier contient des identifiants de fenêtres en double.")
    if not pd.api.types.is_numeric_dtype(scores["score"]):
        raise ValueError("La colonne 'score' doit être numérique.")
    if not np.isfinite(scores["score"].to_numpy(dtype=float)).all():
        raise ValueError("La colonne 'score' contient une valeur manquante ou infinie.")

    # Les égalités gardent l'ordre du fichier d'entrée. Ne pas trier les identifiants.
    ranked = scores.sort_values("score", ascending=False, kind="stable")
    if top_n is not None:
        ranked = ranked.head(top_n)
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output, index=False, encoding="utf-8")
    return ranked


def parser():
    root = argparse.ArgumentParser(
        description="Classer les fenêtres par score IF décroissant et exporter un CSV."
    )
    root.add_argument("scores", help="Parquet produit par score-iforest, avec son manifeste.")
    root.add_argument("output", help="CSV à écrire, par exemple artifacts/if/ranked_windows.csv.")
    root.add_argument("--top-n", type=int, default=None,
                      help="Ne conserver que les N premiers rangs (défaut : toutes les fenêtres).")
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        ranked = rank_windows(args.scores, args.output, top_n=args.top_n)
    except (OSError, ValueError) as exc:
        root.error(str(exc))
    print(f"{len(ranked)} fenêtres classées par score IF décroissant : {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
