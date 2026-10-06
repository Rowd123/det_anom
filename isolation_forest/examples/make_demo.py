"""Generate a small reproducible input and explicit example configurations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/if_demo')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configs = Path(__file__).resolve().parents[1] / 'configs'
    load = lambda name: json.loads((configs / f'{name}.json').read_text())
    p, f, m, s = [load(name) for name in ('window_preprocessing', 'window_features', 'window_models', 'spot')]
    p.update(sampling_frequency=10., sampling_period='100ms')
    p['windowing'] = {'size': 100, 'overlap': 50}
    p['normalization'] = {'V1.MAG': {'mu': 100., 'sigma': 2.}}
    f['spectral']['bands_hz'] = [[.1, .5], [.5, 1.5], [1.5, 3.], [3., 5.]]
    m['atypicality'].update(backend='cpu', n_estimators=64, n_jobs=1)
    for name, c in zip(('window_preprocessing', 'window_features', 'window_models', 'spot'), (p, f, m, s)):
        (output / f'{name}.json').write_text(json.dumps(c, indent=2), encoding='utf-8')
    rng = np.random.default_rng(42)
    x = 100 + 2 * rng.normal(size=100_050)
    x[84_000:84_300] += 8 * np.sin(2 * np.pi * .7 * np.arange(300) / 10)
    x[93_000:93_200] += 10
    pd.DataFrame({'time': pd.date_range('2026-01-01', periods=len(x), freq='100ms', tz='UTC'),
                  'value': x}).to_csv(output / 'signal.csv', index=False)
    print(f'Demo input and configurations: {output.resolve()}')


if __name__ == '__main__':
    main()
