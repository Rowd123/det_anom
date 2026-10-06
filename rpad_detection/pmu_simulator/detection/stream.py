"""Bounded-memory, chronological screening of one synchronized PMU pair."""
from collections import deque
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from .config import validate_config
from .criteria import extract_criteria, decide


class PhaseEventDetector:
    """Call update(timestamp_seconds, angle_a, angle_b, valid=True).

    Returns zero or more completed-window decisions/audit records immediately.
    A record is final: no future sample changes it. No wall-clock sleeps/network I/O.
    """
    def __init__(self, config, *, retrospective=False):
        self.config = validate_config(config)
        if self.config['threshold']['mode'] == 'retrospective' and not retrospective:
            raise ValueError('retrospective thresholds require the offline workflow')
        self.retrospective = retrospective
        self.buffer = deque(maxlen=self.config['window_samples'])
        self.history = deque(maxlen=self.config['threshold']['history_windows'])
        self.pending = []
        self.previous_angles = None
        self.unwrapped = None
        self.last_valid = None
        self.last_timestamp = None
        self.anchor = None
        self.sample_count = 0
        self.window_count = 0
        self.reset_count = 0

    def _reset_segment(self):
        self.buffer.clear(); self.pending.clear()
        self.previous_angles = self.unwrapped = self.last_valid = None
        self.sample_count = 0
        self.reset_count += 1
        if self.config['threshold']['reset_on_gap']:
            self.history.clear()

    def _audit(self, timestamp, reason, **extra):
        return dict(pair_id=self.config['pair_id'], status='rejected', reason=reason,
                    available_at=float(timestamp), is_anomaly=False, **extra)

    def _append(self, timestamp, angles, observed, available_at):
        p = self.config['preprocessing']
        rpad = float(angles[0]-angles[1])
        self.buffer.append((float(timestamp), rpad, bool(observed)))
        self.sample_count += 1
        n, hop = self.config['window_samples'], self.config['hop_samples']
        if self.sample_count < n or (self.sample_count-n) % hop:
            return []
        self.window_count += 1
        times, raw, observed_flags = map(list, zip(*self.buffer))
        base = dict(pair_id=self.config['pair_id'], window_id=self.window_count,
                    window_start=times[0], window_end=times[-1], available_at=float(available_at),
                    observed_fraction=float(np.mean(observed_flags)), interpolated_samples=observed_flags.count(False),
                    threshold_mode=self.config['threshold']['mode'], times=times, rpad_degrees=raw,
                    reset_count=self.reset_count)
        if base['observed_fraction'] < p['min_valid_fraction']:
            return [{**base, 'status':'rejected', 'reason':'insufficient_valid_fraction', 'is_anomaly':False}]
        start = perf_counter()
        try:
            scores, details = extract_criteria((np.array(raw)-p['mu'])/p['sigma'], self.config)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            return [{**base, 'status':'rejected', 'reason':'criterion_failure: '+str(exc), 'is_anomaly':False}]
        evidence, triggers, anomaly = {}, [], False
        if self.retrospective:
            status = 'pending_retrospective'
        elif len(self.history) < self.config['threshold']['min_history']:
            status = 'warmup'
        else:
            status = 'evaluated'
            evidence, triggers, anomaly = decide(scores, self.history, self.config)
        t = self.config['threshold']
        if not self.retrospective and (status == 'warmup' or (t['mode'] == 'rolling' and (t['update_policy'] == 'all' or not anomaly))):
            self.history.append(scores)
        elapsed = perf_counter()-start
        return [{**base, 'status':status, 'scores':scores, 'details':details, 'evidence':evidence,
                 'triggered_methods':triggers, 'vote_count':len(triggers),
                 'required_votes':self.config['consensus']['min_methods'], 'is_anomaly':bool(anomaly),
                 'compute_seconds':elapsed, 'compute_over_hop':elapsed > hop/self.config['sampling_rate']}]

    def update(self, timestamp, angle_a=None, angle_b=None, valid=True):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp): raise ValueError('timestamp must be finite seconds')
        if not isinstance(valid, (bool, np.bool_)): raise ValueError('valid must be boolean')
        p = self.config['preprocessing']; fs = self.config['sampling_rate']
        tolerance = p['timestamp_tolerance']
        records = []
        if self.anchor is None: self.anchor = timestamp
        tick = round((timestamp-self.anchor)*fs)
        canonical = self.anchor + tick/fs
        if abs(timestamp-canonical) > tolerance:
            raise ValueError('timestamp is off the reporting grid')
        timestamp = canonical
        if self.last_timestamp is not None:
            if timestamp < self.last_timestamp - tolerance:
                raise ValueError('out-of-order input; streaming does not reorder future samples')
            if abs(timestamp-self.last_timestamp) <= tolerance:
                return [self._audit(timestamp, 'duplicate_ignored')]
            missing = round((timestamp-self.last_timestamp)*fs)-1
            if missing > p['max_interpolation_gap']:
                self._reset_segment()
                records.append(self._audit(timestamp, 'long_missing_interval', missing_samples=missing))
            else:
                for k in range(missing):
                    self.pending.append(self.last_timestamp+(k+1)/fs)
        if any(a <= t <= b for t in self.pending for a,b in p['excluded_intervals']):
            self._reset_segment()
            records.append(self._audit(timestamp, 'excluded_missing_sample'))
        self.last_timestamp = timestamp
        excluded = any(a <= timestamp <= b for a,b in p['excluded_intervals'])
        try:
            angles = np.asarray([angle_a,angle_b], float)
            good = bool(valid and np.isfinite(angles).all() and not excluded)
        except (TypeError, ValueError):
            good = False
        if excluded:
            self._reset_segment()
            return records + [self._audit(timestamp, 'excluded_sample')]
        if not good:
            self.pending.append(timestamp)
            if len(self.pending) > p['max_interpolation_gap']:
                self._reset_segment()
                records.append(self._audit(timestamp, 'invalid_run_too_long'))
            return records
        # Long accumulated invalid runs must not be bridged.
        if len(self.pending) > p['max_interpolation_gap']:
            self._reset_segment()
            records.append(self._audit(timestamp, 'invalid_run_too_long'))
        if self.config['angle_unit'] == 'radians': angles = np.rad2deg(angles)
        if self.previous_angles is None:
            unwrapped = angles
        else:
            delta = (angles-self.previous_angles+180.) % 360.-180.
            delta = np.where((delta == -180) & ((angles-self.previous_angles)>0), 180., delta)
            unwrapped = np.asarray(self.unwrapped)+delta
        if self.pending and self.last_valid is not None:
            old_time, old_angles = self.last_valid
            for t in self.pending:
                fraction = (t-old_time)/(timestamp-old_time)
                interpolated = np.asarray(old_angles) + fraction*(unwrapped-old_angles)
                records.extend(self._append(t, interpolated, False, timestamp))
        # Leading gaps cannot be interpolated: start the first complete segment here.
        self.pending.clear()
        records.extend(self._append(timestamp, unwrapped, True, timestamp))
        self.previous_angles = angles.tolist(); self.unwrapped = unwrapped.tolist()
        self.last_valid = (timestamp, unwrapped.tolist())
        return records

    def flush(self):
        """Report discarded trailing gaps/incomplete windows; never extrapolate."""
        records = []
        n, hop = self.config['window_samples'], self.config['hop_samples']
        if self.pending or self.sample_count < n or (self.sample_count-n) % hop:
            if self.last_timestamp is not None:
                records.append(self._audit(self.last_timestamp, 'incomplete_tail', pending_samples=len(self.pending), buffered_samples=len(self.buffer)))
        return records

    def save_state(self, path):
        data = dict(version=1, config=self.config, retrospective=self.retrospective,
                    buffer=list(self.buffer), history=list(self.history), pending=self.pending,
                    previous_angles=self.previous_angles, unwrapped=self.unwrapped,
                    last_valid=self.last_valid, last_timestamp=self.last_timestamp,
                    anchor=self.anchor, sample_count=self.sample_count,
                    window_count=self.window_count, reset_count=self.reset_count)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name+'.tmp')
        tmp.write_text(json.dumps(data, allow_nan=False), encoding='utf8'); tmp.replace(path)

    @classmethod
    def load_state(cls, path, config=None):
        data = json.loads(Path(path).read_text(encoding='utf8'))
        if data['version'] != 1: raise ValueError('unsupported detector state')
        if config is not None and validate_config(config) != data['config']:
            raise ValueError('state/config mismatch')
        result = cls(data['config'], retrospective=data['retrospective'])
        result.buffer.extend(data['buffer']); result.history.extend(data['history'])
        for name in ('pending','previous_angles','unwrapped','last_valid','last_timestamp','anchor','sample_count','window_count','reset_count'):
            setattr(result, name, data[name])
        return result
