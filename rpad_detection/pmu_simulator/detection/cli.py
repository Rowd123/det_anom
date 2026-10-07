"""CSV replay, streaming NDJSON, and independent report generation."""
import json
from pathlib import Path
import sys


def add_commands(sub):
    detect=sub.add_parser('detect-phase',help='Détection RPAD depuis deux angles synchronisés en CSV')
    detect.add_argument('--config',required=True)
    detect.add_argument('--csv', required=True)
    detect.add_argument('--time-column',default='time'); detect.add_argument('--angle-a',default='angle_a'); detect.add_argument('--angle-b',default='angle_b')
    detect.add_argument('--valid-column'); detect.add_argument('--sep',default=',')
    detect.add_argument('--time-format', choices=('seconds', 'iso8601'), default='seconds',
                        help='Format de la colonne temporelle; ISO 8601 exige un fuseau explicite')
    detect.add_argument('--output',required=True); detect.add_argument('--report-dir')
    live=sub.add_parser('phase-stream',help='NDJSON stdin → décisions NDJSON stdout (causal)')
    live.add_argument('--config',required=True); live.add_argument('--state-in'); live.add_argument('--state-out')
    live.add_argument('--checkpoint-every',type=int,default=1000)
    report=sub.add_parser('phase-report',help='Visualiser les décisions sauvegardées sans recalcul')
    report.add_argument('decisions'); report.add_argument('output_dir')


def run(args):
    from .config import load_config
    from .io import csv_samples, replay, save_decisions
    from .stream import PhaseEventDetector
    if args.command=='phase-report':
        from .report import render_report
        render_report(args.decisions,args.output_dir)
        return
    config=load_config(args.config)
    if args.command=='phase-stream':
        if args.checkpoint_every < 1: raise ValueError('checkpoint-every must be positive')
        if config['threshold']['mode']=='retrospective': raise ValueError('phase-stream requires rolling or frozen thresholds')
        detector=PhaseEventDetector.load_state(args.state_in,config) if args.state_in else PhaseEventDetector(config)
        for i,line in enumerate(sys.stdin,1):
            if i == 1: line = line.lstrip('\ufeff')
            if not line.strip(): continue
            for row in detector.update(**json.loads(line)):
                print(json.dumps(row,allow_nan=False),flush=True)
            if args.state_out and i%args.checkpoint_every==0: detector.save_state(args.state_out)
        # Preserve unfinished windows/gaps at EOF for subsequent input chunks.
        if args.state_out: detector.save_state(args.state_out)
        return
    samples=csv_samples(args.csv,time_column=args.time_column,angle_a=args.angle_a,angle_b=args.angle_b,valid_column=args.valid_column,delimiter=args.sep,time_format=args.time_format)
    rows=replay(samples,config)
    save_decisions(rows,args.output,config)
    if args.report_dir:
        from .report import render_report
        render_report(args.output,args.report_dir)
