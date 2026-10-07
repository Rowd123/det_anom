import json
from datetime import datetime, timedelta, timezone

import pytest

from pmu_simulator.detection.io import csv_samples
from pmu_simulator.detection.config import DEFAULT
from pmu_simulator.cli import main


def test_explicit_iso8601_matches_numeric_seconds(tmp_path):
    path = tmp_path / 'angles.csv'
    path.write_text('time,a,b,valid\n2026-10-07T00:00:00Z,1,0,1\n2026-10-07T02:00:00.020+02:00,2,0,1\n')
    records = list(csv_samples(path, angle_a='a', angle_b='b', time_format='iso8601', valid_column='valid'))
    assert records[1]['timestamp'] - records[0]['timestamp'] == pytest.approx(.02, abs=1e-6)
    assert records[0]['timestamp'] == datetime(2026, 10, 7, tzinfo=timezone.utc).timestamp()
    path.write_text('time,angle_a,angle_b\n2026-10-07T00:00:00,1,0\n')
    with pytest.raises(ValueError, match='timezone'):
        list(csv_samples(path, time_format='iso8601'))


def test_cli_iso8601_replay(tmp_path):
    config = dict(DEFAULT, sampling_rate=50.)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    start = datetime(2026, 10, 7, tzinfo=timezone.utc)
    csv = tmp_path / 'angles.csv'
    csv.write_text('time,angle_a,angle_b\n' + ''.join(
        f'{(start + timedelta(milliseconds=20*i)).isoformat()},1,0\n' for i in range(320)))
    output = tmp_path / 'decisions.jsonl'
    main(['detect-phase', '--config', str(config_path), '--csv', str(csv), '--time-format', 'iso8601', '--output', str(output)])
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert any('scores' in row for row in records)
