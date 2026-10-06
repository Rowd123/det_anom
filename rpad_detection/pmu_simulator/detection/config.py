"""Strict configuration for reproducible phase-difference screening."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

DEFAULT = {
    'sampling_rate': 30.0, 'window_samples': 300, 'hop_samples': 150,
    'angle_unit': 'degrees', 'pair_id': 'PMU_A-PMU_B',
    'preprocessing': {'mu': 0.0, 'sigma': 1.0, 'max_interpolation_gap': 3,
                      'min_valid_fraction': .9, 'timestamp_tolerance': 1e-5,
                      'excluded_intervals': []},
    'methods': ['fft', 'matrix_pencil', 'yule_walker', 'minmax'],
    'frequency_band': [0.0, 2.0],
    'fft': {'preprocess': 'difference', 'window': 'boxcar', 'nfft': 512},
    'matrix_pencil': {'preprocess': 'linear', 'pencil_length': 100,
                      'rank': None, 'svd_relative_tolerance': .1, 'max_rank': 20,
                      'rcond': 1e-10},
    'yule_walker': {'preprocess': 'linear', 'order': 20, 'nfft': 1024,
                    'ridge': 1e-10, 'power_floor': 1e-30},
    'threshold': {'mode': 'rolling', 'history_windows': 720, 'min_history': 60,
                  'ddof': 1, 'k_sigma': 3.0, 'min_std': 1e-12,
                  'update_policy': 'all', 'reset_on_gap': True},
    'consensus': {'min_methods': 2},
}


def validate_config(config):
    c = deepcopy(config)
    def keys(d, expected, name):
        if not isinstance(d, dict) or set(d) != set(expected):
            raise ValueError(f'{name}: expected exactly {sorted(expected)}')
    def integer(x, low, high, name):
        if isinstance(x, bool) or not isinstance(x, int) or not low <= x <= high:
            raise ValueError(f'{name}: expected integer in [{low}, {high}]')
    def finite(x, name, positive=False):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or (positive and x <= 0):
            raise ValueError(f'{name}: invalid finite numeric value')
    keys(c, DEFAULT, 'config')
    for name in ('preprocessing', 'fft', 'matrix_pencil', 'yule_walker', 'threshold', 'consensus'):
        keys(c[name], DEFAULT[name], name)
    finite(c['sampling_rate'], 'sampling_rate', True)
    integer(c['window_samples'], 8, 100000, 'window_samples')
    n = c['window_samples']
    integer(c['hop_samples'], 1, n, 'hop_samples')
    if c['angle_unit'] not in ('degrees', 'radians') or not isinstance(c['pair_id'], str) or not c['pair_id']:
        raise ValueError('invalid angle_unit/pair_id')
    methods = c['methods']
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods) or set(methods) - set(DEFAULT['methods']):
        raise ValueError('invalid methods')
    band = c['frequency_band']
    if len(band) != 2 or not np.isfinite(band).all() or not 0 <= band[0] < band[1] <= c['sampling_rate']/2:
        raise ValueError('invalid frequency_band (upper bound must not exceed Nyquist)')
    p = c['preprocessing']
    finite(p['mu'], 'mu'); finite(p['sigma'], 'sigma', True)
    finite(p['timestamp_tolerance'], 'timestamp_tolerance', True)
    if p['timestamp_tolerance'] >= .5/c['sampling_rate']:
        raise ValueError('timestamp tolerance must be less than half a sample period')
    integer(p['max_interpolation_gap'], 0, n-1, 'max_interpolation_gap')
    if not 0 <= p['min_valid_fraction'] <= 1:
        raise ValueError('invalid min_valid_fraction')
    for interval in p['excluded_intervals']:
        if len(interval) != 2 or not np.isfinite(interval).all() or interval[0] > interval[1]:
            raise ValueError('invalid excluded interval in timestamp seconds')
    for name in ('fft', 'matrix_pencil', 'yule_walker'):
        if c[name]['preprocess'] not in ('none', 'constant', 'linear', 'difference'):
            raise ValueError(f'{name}: invalid preprocess')
    from scipy.signal import get_window
    get_window(c['fft']['window'], n)
    for name in ('fft', 'yule_walker'):
        integer(c[name]['nfft'], n, 10000000, name+'.nfft')
        f = np.fft.rfftfreq(c[name]['nfft'], 1/c['sampling_rate'])
        if not np.any((f > band[0]) & (f <= band[1])):
            raise ValueError('frequency band has no spectral bins')
    m = c['matrix_pencil']
    length = n - int(m['preprocess'] == 'difference')
    integer(m['pencil_length'], 2, length-2, 'pencil_length')
    limit = min(m['pencil_length'], length-m['pencil_length'])
    integer(m['max_rank'], 1, limit, 'max_rank')
    if m['rank'] is not None:
        integer(m['rank'], 1, m['max_rank'], 'rank')
    if not 0 < m['svd_relative_tolerance'] < 1 or not 0 < m['rcond'] < 1:
        raise ValueError('invalid matrix-pencil tolerance/rcond')
    y = c['yule_walker']
    integer(y['order'], 1, n-2, 'yule_walker.order')
    finite(y['ridge'], 'ridge')
    if y['ridge'] < 0: raise ValueError('ridge must be nonnegative')
    finite(y['power_floor'], 'power_floor', True)
    t = c['threshold']
    if t['mode'] not in ('rolling', 'frozen', 'retrospective') or t['update_policy'] not in ('all', 'normal_only'):
        raise ValueError('invalid threshold mode/update_policy')
    integer(t['history_windows'], 2, 1000000, 'history_windows')
    integer(t['min_history'], 2, t['history_windows'], 'min_history')
    integer(t['ddof'], 0, t['min_history']-1, 'ddof')
    finite(t['k_sigma'], 'k_sigma')
    if t['k_sigma'] < 0: raise ValueError('k_sigma must be nonnegative')
    finite(t['min_std'], 'min_std', True)
    if not isinstance(t['reset_on_gap'], bool): raise ValueError('reset_on_gap must be boolean')
    integer(c['consensus']['min_methods'], 1, len(methods), 'min_methods')
    return c


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding='utf8')))
