"""Inference-only SPOT analysis and ranked physical window reconstructions."""
from argparse import ArgumentParser
import csv
import heapq
from html import escape
import json
from pathlib import Path

import numpy as np
import torch

from spot import SPOT
from utils import BatchSlidingWindow, get_data, phase_embedding_to_phasor
from visualize_reconstruction import (
    load_config, load_trained_model, resolve_from_config,
    physical_reconstruction_indices,
)


def apply_spot(calibration, scores, q=0.001, level=0.98):
    """Run in stream order; report the threshold used BEFORE each observation."""
    calibration = np.asarray(calibration, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if any(x.ndim != 1 or not len(x) or not np.isfinite(x).all()
           for x in (calibration, scores)):
        raise ValueError("Calibration et scores doivent être des vecteurs finis non vides")
    if not 0 < level < 1 or not 0 < q < 1 - level:
        raise ValueError("Il faut 0 < level < 1 et 0 < q < 1-level")
    initial = np.sort(calibration)[int(level * len(calibration))]
    peaks = calibration[calibration > initial] - initial
    if len(peaks) < 5 or np.ptp(peaks) == 0:
        raise ValueError("Calibration SPOT insuffisante : au moins 5 excès non constants requis")
    detector = SPOT(q)
    detector.fit(calibration.copy(), scores.copy())
    detector.initialize(level=level, min_extrema=False, verbose=False)
    first = detector.extreme_quantile
    result = detector.run(with_alarm=True, dynamic=True)
    # Existing SPOT.run records thresholds after its update. Shift back to
    # preserve the actual decision threshold, including at non-alarm peaks.
    thresholds = np.r_[first, np.asarray(result['thresholds'])[:-1]]
    if not np.isfinite(thresholds).all():
        raise ValueError("SPOT a produit un seuil non fini")
    alarms = np.zeros(len(scores), dtype=bool)
    alarms[np.asarray(result['alarms'], dtype=int)] = True
    if not np.array_equal(alarms, scores > thresholds):
        raise RuntimeError("Incohérence entre seuils SPOT et décisions")
    return thresholds, alarms


@torch.inference_mode()
def score_windows(model, normalizer, values, excludes, batch_size, n_z, device,
                  retain=0, stride=1):
    """Retain only top-k reconstructions (all if retain=None, none if 0)."""
    values = torch.as_tensor(values, dtype=torch.float32)
    iterator = BatchSlidingWindow(
        len(values), model.window_length, batch_size=batch_size,
        excludes=excludes, shuffle=False, ignore_incomplete_batch=False,
        stride=stride,
    ).get_iterator([values, torch.arange(len(values))])
    all_scores, all_ends, retained = [], [], []
    model.eval()
    for windows, indices in iterator:
        normalized, context = normalizer.transform(windows.to(device))
        log_prob, _, reconstruction = model.get_score(
            normalized, n_z=n_z, last_point_only=True, return_reconstruction=True,
        )
        if log_prob.ndim == 2:
            log_prob = log_prob.sum(dim=-1)
        scores = -log_prob.cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError("Score OmniAnomaly non fini")
        ends = indices[:, -1].numpy()
        all_scores.extend(scores.tolist())
        all_ends.extend(ends.tolist())
        if retain == 0:
            continue
        reconstruction = normalizer.inverse_transform(reconstruction, context).cpu()
        for i, (score, end) in enumerate(zip(scores, ends)):
            # Stable ranking: earlier window first on equal scores.
            key = (float(score), -int(end))
            if retain is None or len(retained) < retain or key > retained[0][:2]:
                item = (*key, windows[i].clone(), reconstruction[i].clone())
                if retain is None or len(retained) < retain:
                    heapq.heappush(retained, item)
                else:
                    heapq.heapreplace(retained, item)
    if not all_scores:
        raise ValueError("Aucune fenêtre valide")
    return np.asarray(all_scores), np.asarray(all_ends), {
        -item[1]: (item[2], item[3]) for item in retained
    }


def export_results(output, scores, ends, thresholds, alarms, retained,
                   representation, window_length, offset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    physical = physical_reconstruction_indices(representation)
    names = [representation.original_names[i] for i in physical]
    order = np.argsort(-scores, kind='stable')
    rows, links = [], []
    for rank, i in enumerate(order, 1):
        end = int(ends[i])
        row = dict(rank=rank, window_id=end, window_start=end-window_length+1,
                   window_end=end, source_start=offset+end-window_length+1,
                   source_end=offset+end, score=float(scores[i]),
                   threshold=float(thresholds[i]),
                   excess=float(scores[i]-thresholds[i]), is_anomaly=bool(alarms[i]),
                   reconstruction_file='', figure_file='')
        if end in retained:
            observed, reconstructed = retained[end]
            observed = phase_embedding_to_phasor(observed, representation).numpy()[:, physical]
            reconstructed = phase_embedding_to_phasor(reconstructed, representation).numpy()[:, physical]
            stem = f'rank-{rank:06d}_window-{end}'
            row['reconstruction_file'] = stem + '.npz'
            row['figure_file'] = stem + '.svg'
            x = np.arange(row['source_start'], row['source_end'] + 1)
            np.savez_compressed(output / row['reconstruction_file'],
                                observed=observed, reconstructed=reconstructed,
                                source_indices=x, feature_names=np.asarray(names),
                                score=scores[i], threshold=thresholds[i], is_anomaly=alarms[i])
            fig, axes = plt.subplots(len(names), 1, figsize=(12, 2.4*len(names)),
                                     sharex=True, squeeze=False)
            for j, name in enumerate(names):
                ax = axes[j, 0]
                ax.plot(x, observed[:, j], label='Observed', color='#318a99')
                ax.plot(x, reconstructed[:, j], label='Reconstructed', color='#d59360', ls='--')
                ax.set_ylabel(name)
                ax.grid(alpha=.2)
                ax.legend(loc='upper right')
            axes[-1, 0].set_xlabel('Source row index (0-based)')
            title = (f'Rank {rank} | score={scores[i]:.6g} | SPOT threshold={thresholds[i]:.6g}'
                     f' | anomaly={bool(alarms[i])}')
            fig.suptitle(title)
            fig.tight_layout(rect=(0, 0, 1, .985))
            fig.savefig(output / row['figure_file'], bbox_inches='tight')
            plt.close(fig)
            links.append(f'<li><a href="{row["figure_file"]}">{escape(title)}</a>'
                         f' — <a href="{row["reconstruction_file"]}">NPZ</a></li>')
        rows.append(row)
    with (output / 'ranked_windows.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / 'chronological_scores.npz', window_end=ends,
                        score=scores, threshold=thresholds, is_anomaly=alarms)
    (output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>OmniAnomaly + SPOT</title>'
        '<h1>Windows ranked by decreasing OmniAnomaly score</h1>'
        '<p>SPOT was applied in chronological order. Score = negative reconstruction '
        'log-likelihood at the last point of each window; it is not a probability.</p>'
        '<a href="ranked_windows.csv">All ranked scores (CSV)</a><ol>'
        + ''.join(links) + '</ol>', encoding='utf-8')


def main(args):
    requested_stride = getattr(args, 'stride', None)
    if requested_stride is not None and requested_stride < 1:
        raise ValueError('stride doit être supérieur ou égal à 1')
    if args.top < 0 or args.batch_size < 1 or args.n_z < 1:
        raise ValueError('top >= 0, batch-size >= 1 et n-z >= 1 requis')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    path = Path(args.config).resolve()
    config = load_config(path)
    train, test, representation = get_data(
        dataset=config.dataset,
        data_directory=resolve_from_config(config.data_dir, path),
        max_train_size=config.max_train_size, max_test_size=config.max_test_size,
        train_start=config.train_start, return_metadata=True,
        exclude_zero_sequence_phase=config.exclude_zero_sequence_phase,
        data_column_order=config.data_column_order,
        excluded_features=getattr(config, 'excluded_features', []),
    )
    if test[0] is None:
        raise ValueError('Une partie test distincte de la calibration train est requise')
    model, normalizer, device = load_trained_model(
        config, representation, Path(args.checkpoint).resolve(), args.device,
    )
    stride = model.window_length if requested_stride is None else requested_stride
    print(f'Longueur des fenêtres : {model.window_length}; pas : {stride} échantillons '
          '(calibration et test). Score calculé au dernier point de chaque fenêtre.')
    common = dict(model=model, normalizer=normalizer, batch_size=args.batch_size,
                  n_z=args.n_z, device=device, stride=stride)
    calibration, calibration_ends, _ = score_windows(
        values=train[0], excludes=train[2], **common,
    )
    scores, ends, retained = score_windows(
        values=test[0], excludes=test[2], retain=args.top or None, **common,
    )
    thresholds, alarms = apply_spot(calibration, scores, args.q, args.level)
    output = Path(args.output)
    export_results(output, scores, ends, thresholds, alarms, retained,
                   representation, model.window_length, config.train_start+len(train[0]))
    np.savez_compressed(output / 'calibration_scores.npz',
                        score=calibration, window_end=calibration_ends)
    metadata = dict(vars(args), config_path=str(path), configuration=vars(config),
                    feature_names=representation.model_names,
                    score_definition='negative mean-over-z log-likelihood, sum over features, last point',
                    ranking='score descending; ties chronological',
                    calibration_windows=len(calibration), test_windows=len(scores),
                    anomalies=int(alarms.sum()))
    metadata['stride'] = stride
    (output / 'analysis.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'{len(scores)} fenêtres, {alarms.sum()} anomalies. Résultats : {output.resolve()}/index.html')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='omnis/config.yaml', help='Configuration exacte du modèle entraîné')
    parser.add_argument('--checkpoint', required=True, help='Chemin relatif au répertoire courant')
    parser.add_argument('--output', default='artifacts/omni_spot')
    parser.add_argument('--top', type=int, default=50, help='Fenêtres à reconstruire/exporter; 0 = toutes')
    parser.add_argument('--stride', type=int, default=None,
                        help='Pas en échantillons; défaut = longueur de fenêtre (sans recouvrement); 1 = ancien comportement')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-z', type=int, default=16)
    parser.add_argument('--q', type=float, default=.001)
    parser.add_argument('--level', type=float, default=.98, help='Quantile initial en queue haute')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    main(parser.parse_args())
