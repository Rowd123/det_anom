"""Input adapters and offline replay using exactly the live detector."""
import csv
from datetime import datetime
import json
from pathlib import Path
from .stream import PhaseEventDetector
from .criteria import decide


def csv_samples(path, *, time_column='time', angle_a='angle_a', angle_b='angle_b',
                valid_column=None, delimiter=',', time_format='seconds'):
    """Read seconds or explicitly selected timezone-aware ISO 8601 timestamps."""
    if time_format not in {'seconds', 'iso8601'}:
        raise ValueError('time_format must be seconds or iso8601')
    with open(path, newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream, delimiter=delimiter):
            def angle(name):
                return None if row[name].strip() == '' else float(row[name])
            valid = True
            if valid_column:
                flag = row[valid_column].strip().lower()
                if flag not in {'0','1','false','true'}: raise ValueError('valid column must contain boolean flags')
                valid = flag in {'1','true'}
            raw_time = row[time_column].strip()
            if time_format == 'seconds':
                timestamp = float(raw_time)
            else:
                parsed = datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
                if parsed.tzinfo is None:
                    raise ValueError('ISO 8601 timestamps require a timezone offset (Z or +HH:MM)')
                timestamp = parsed.timestamp()
            yield dict(timestamp=timestamp, angle_a=angle(angle_a), angle_b=angle(angle_b), valid=valid)




def replay(samples, config):
    retrospective = config['threshold']['mode'] == 'retrospective'
    detector = PhaseEventDetector(config, retrospective=retrospective)
    rows=[]
    for sample in samples: rows.extend(detector.update(**sample))
    rows.extend(detector.flush())
    if retrospective:
        candidates = [r for r in rows if 'scores' in r]
        # The provided input file is the reference block (one hour in the paper).
        reference = [r['scores'] for r in candidates]
        for row in candidates:
            if len(reference)<config['threshold']['min_history']:
                row['status']='warmup'
                continue
            evidence,triggers,event = decide(row['scores'],reference,config)
            row.update(status='evaluated',evidence=evidence,triggered_methods=triggers,
                       vote_count=len(triggers),is_anomaly=event)
    return rows


def save_decisions(rows, path, config):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf8') as stream:
        for row in rows: stream.write(json.dumps(row,allow_nan=False)+'\n')
    path.with_suffix(path.suffix+'.config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf8')
