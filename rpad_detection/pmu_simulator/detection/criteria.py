"""Four screening methods, not event classification. See docs/rpad_detection.md."""
import numpy as np
from scipy import linalg, signal


def preprocess(x, mode):
    if mode == 'difference': return np.diff(x)
    if mode in ('constant', 'linear'): return signal.detrend(x, type=mode)
    return x.copy()


def matrix_pencil(x, fs, options, band):
    y = preprocess(x, options['preprocess'])
    length = options['pencil_length']
    hankel = np.lib.stride_tricks.sliding_window_view(y, length+1)
    _, singular, vh = linalg.svd(hankel, full_matrices=False, check_finite=False)
    if singular[0] <= np.finfo(float).eps * max(1., np.linalg.norm(y)):
        return [0., 0.], {'rank': 0, 'modes': [], 'relative_fit_error': 0.}
    rank = options['rank'] or min(int(np.sum(singular > singular[0]*options['svd_relative_tolerance'])), options['max_rank'])
    v = vh[:rank].conj().T
    poles = np.linalg.eigvals(np.linalg.pinv(v[:-1], rcond=options['rcond']) @ v[1:])
    if np.any(np.abs(poles) < np.finfo(float).tiny):
        raise ValueError('matrix-pencil zero pole')
    with np.errstate(over='raise', invalid='raise'):
        basis = poles[None, :] ** np.arange(len(y))[:, None]
        residues = np.linalg.lstsq(basis, y, rcond=options['rcond'])[0]
    frequencies = np.angle(poles)*fs/(2*np.pi)
    damping = np.log(np.abs(poles))*fs
    # |R| is a complex-exponential residue, not the doubled real sinusoid amplitude.
    modes = [dict(frequency=float(f), damping=float(d), amplitude=float(abs(r)), phase=float(np.angle(r)))
             for f, d, r in zip(frequencies, damping, residues)
             if band[0] < f <= band[1]]
    modes.sort(key=lambda m: m['amplitude'], reverse=True)
    peaks = ([m['amplitude'] for m in modes] + [0., 0.])[:2]
    residual = float(np.linalg.norm(y - (basis@residues).real) / max(np.linalg.norm(y), np.finfo(float).eps))
    if not np.isfinite(peaks+[residual]).all(): raise ValueError('non-finite matrix-pencil fit')
    return peaks, {'rank': rank, 'modes': modes, 'relative_fit_error': residual}


def yule_walker(x, fs, options, band):
    y = preprocess(x, options['preprocess'])
    p = options['order']
    # Biased autocorrelation gives a positive-semidefinite Toeplitz estimate.
    r = np.array([y[k:]@y[:len(y)-k]/len(y) for k in range(p+1)])
    if r[0] <= options['power_floor']:
        return 10*np.log10(options['power_floor']), {'peak_frequency': None, 'innovation_variance': 0.}
    r[0] *= 1 + options['ridge']
    a = np.r_[1., linalg.solve_toeplitz(r[:p], -r[1:], check_finite=False)]
    innovation = max(float(r[0]+a[1:]@r[1:]), options['power_floor'])
    f = np.fft.rfftfreq(options['nfft'], 1/fs)
    denominator = np.abs(np.fft.rfft(a, n=options['nfft']))**2
    density = innovation / (fs*np.maximum(denominator, options['power_floor']))
    density[1:] *= 2
    if options['nfft'] % 2 == 0: density[-1] /= 2
    selected = np.flatnonzero((f > band[0]) & (f <= band[1]))
    index = selected[np.argmax(density[selected])]
    return float(10*np.log10(max(density[index], options['power_floor']))), {'peak_frequency': float(f[index]), 'innovation_variance': innovation}


def extract_criteria(values, config):
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) != config['window_samples'] or not np.isfinite(x).all():
        raise ValueError('expected a finite complete window')
    fs, band = config['sampling_rate'], config['frequency_band']
    scores, details = {}, {}
    for method in config['methods']:
        if method == 'minmax':
            scores['minmax'] = float(np.ptp(x))
        elif method == 'fft':
            o = config['fft']; y = preprocess(x, o['preprocess'])
            taper = signal.get_window(o['window'], len(y))
            gain = abs(taper.sum())
            if gain <= np.finfo(float).eps: raise ValueError('FFT window has zero coherent gain')
            amplitudes = abs(np.fft.rfft(y*taper, n=o['nfft']))/gain
            amplitudes[1:] *= 2
            if o['nfft'] % 2 == 0: amplitudes[-1] /= 2
            f = np.fft.rfftfreq(o['nfft'], 1/fs)
            indices = np.flatnonzero((f > band[0]) & (f <= band[1]))
            index = indices[np.argmax(amplitudes[indices])]
            scores['fft'] = float(amplitudes[index])
            details['fft'] = {'peak_frequency': float(f[index])}
        elif method == 'matrix_pencil':
            peaks, details[method] = matrix_pencil(x, fs, config[method], band)
            scores.update(mp_1=peaks[0], mp_2=peaks[1])
        else:
            scores[method], details[method] = yule_walker(x, fs, config[method], band)
    if not np.isfinite(list(scores.values())).all(): raise ValueError('non-finite screening criterion')
    return scores, details


def decide(scores, reference, config):
    t = config['threshold']
    evidence = {}
    for name, value in scores.items():
        baseline = np.asarray([r[name] for r in reference], float)
        mean, std = float(baseline.mean()), float(baseline.std(ddof=t['ddof']))
        threshold = mean + t['k_sigma']*max(std, t['min_std'])
        evidence[name] = dict(value=value, mean=mean, std=std, threshold=threshold,
                              excess=value-threshold, triggered=bool(value>threshold), reference_count=len(baseline))
    triggered = []
    for method in config['methods']:
        names = ('mp_1', 'mp_2') if method == 'matrix_pencil' else (method,)
        if any(evidence[n]['triggered'] for n in names): triggered.append(method)
    return evidence, triggered, len(triggered) >= config['consensus']['min_methods']
