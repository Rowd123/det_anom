"""Read-only score ranking and feature correlations for window artifacts."""
from pathlib import Path
import json

from .artifacts import SEGMENT_KEYS, read_table
from .models import load_artifact


def analyze_scores(features_path, scores_path, output, *, model_path=None, top_n=20):
    """Export all scores and raw descriptors; never refit a saved model."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    features, fm = read_table(features_path, expected_type="window_features")
    scores, sm = read_table(scores_path, expected_type="scores")
    names = list(fm.feature_columns)
    if model_path:
        model = load_artifact(model_path)
        if model.get("kind") != "isolation_forest" or model.get("model_id") != sm.model_id:
            raise ValueError("scores and Isolation Forest model are incompatible")
        names = list(model["feature_columns"])
        model["pipeline"]._validate_compatibility(features)
    keys = list(SEGMENT_KEYS)
    if features.duplicated(keys).any() or scores.duplicated(keys).any():
        raise ValueError("duplicate window identities in input tables")
    selected = features[keys + names]
    table = scores.merge(selected, on=keys, how="left", validate="one_to_one", indicator=True)
    if table.empty or not table['_merge'].eq('both').all():
        raise ValueError("provide feature rows for every scored window")
    table = table.drop(columns='_merge').sort_values('score', ascending=False, kind='stable')
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / 'ranked_windows.csv', index=False)
    table.head(top_n).to_csv(directory / 'top_windows.csv', index=False)
    summary = {'windows': len(table), 'score_convention': sm.score_convention,
               'model_id': sm.model_id, 'feature_columns': names,
               'score_statistics': table['score'].describe(percentiles=[.5, .9, .95, .99]).to_dict()}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    import plotly.express as px
    px.histogram(table, x='score', nbins=50, title='Isolation Forest scores').write_html(
        directory / 'scores.html', include_plotlyjs=True)
    for method in ('pearson', 'spearman'):
        correlation = table[names].corr(method=method)
        correlation.to_csv(directory / f'correlation_{method}.csv')
        px.imshow(correlation, zmin=-1, zmax=1, color_continuous_scale='RdBu_r',
                  title=f'Feature correlation ({method}); constant features are undefined').write_html(
            directory / f'correlation_{method}.html', include_plotlyjs=True)
    return directory
