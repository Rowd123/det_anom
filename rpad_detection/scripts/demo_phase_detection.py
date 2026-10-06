"""Reproducible synchronized-angle fixture: benign noise, damped oscillation, step.

Run: python scripts/demo_phase_detection.py --output artifacts/rpad-demo
These are synthetic PMU angle reports, not a simulation of network power flow.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pmu_simulator.detection.config import DEFAULT
from pmu_simulator.detection.io import replay, save_decisions
from pmu_simulator.detection.report import render_report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='artifacts/rpad-demo')
    args=parser.parse_args();directory=Path(args.output);directory.mkdir(parents=True,exist_ok=True)
    config=deepcopy(DEFAULT)
    # Shorter calibration for the demonstration, not a universally tuned detector.
    config['threshold'].update(min_history=8,history_windows=60)
    config['matrix_pencil'].update(pencil_length=50,max_rank=12)
    rng=np.random.default_rng(42);fs=config['sampling_rate']
    times=np.arange(int(180*fs))/fs
    common=175+18*times
    dt=np.maximum(times-70,0)
    oscillation=np.where((times>=70)&(times<90),2*np.exp(-.15*dt)*np.sin(2*np.pi*.7*dt),0.)
    step=np.where(times>=130,1.,0.)
    difference=.01*np.sin(2*np.pi*.4*times)+rng.normal(0,.002,len(times))+oscillation+step
    left=(common+difference+180)%360-180;right=(common+180)%360-180
    records=[dict(timestamp=float(t),angle_a=float(a),angle_b=float(b),valid=True) for t,a,b in zip(times,left,right)]
    with (directory/'angles.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['time','angle_a','angle_b','valid'])
        writer.writerows((r['timestamp'],r['angle_a'],r['angle_b'],1) for r in records)
    (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (directory/'angles.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    decisions=replay(records,config)
    save_decisions(decisions,directory/'decisions.jsonl',config)
    index=render_report(directory/'decisions.jsonl',directory/'report')
    evaluated=[r for r in decisions if r.get('status')=='evaluated']
    timings=[r['compute_seconds'] for r in decisions if 'compute_seconds' in r]
    print(json.dumps({'windows_evaluated':len(evaluated),'anomalous_windows':sum(r['is_anomaly'] for r in evaluated),
                      'maximum_compute_seconds':max(timings),'hop_seconds':config['hop_samples']/fs,'report':str(index)},indent=2))


if __name__=='__main__': main()
