from copy import deepcopy
import json
import numpy as np
import pytest

from pmu_simulator.detection.config import DEFAULT, validate_config
from pmu_simulator.detection.criteria import matrix_pencil, extract_criteria, decide, yule_walker
from pmu_simulator.detection.stream import PhaseEventDetector
from pmu_simulator.detection.io import replay, save_decisions
from pmu_simulator.detection.report import render_report
from pmu_simulator.cli import main


def config():
    c=deepcopy(DEFAULT)
    c.update(sampling_rate=20.,window_samples=100,hop_samples=50)
    c['matrix_pencil'].update(pencil_length=35,max_rank=10)
    c['yule_walker'].update(order=12,nfft=256)
    c['fft']['nfft']=128
    c['threshold'].update(history_windows=30,min_history=8)
    return validate_config(c)


def samples(count=1600):
    rng=np.random.default_rng(3)
    for i in range(count):
        t=i/20
        event=1.5*np.sin(2*np.pi*.8*t) if 45<=t<55 else 0.
        a=170+12*t+.01*np.sin(2*np.pi*.6*t)+rng.normal(0,.001)+event
        b=170+12*t
        yield dict(timestamp=t,angle_a=(a+180)%360-180,angle_b=(b+180)%360-180)


def stable(rows):
    return [{k:v for k,v in r.items() if k not in ('compute_seconds','compute_over_hop')} for r in rows]


def test_matrix_pencil_known_damped_modes():
    fs=30.;t=np.arange(300)/fs
    x=4*np.exp(-.15*t)*np.cos(2*np.pi*.6*t+.3)+2*np.exp(-.07*t)*np.cos(2*np.pi*1.2*t-.2)
    options=deepcopy(DEFAULT['matrix_pencil']);options.update(preprocess='none',rank=4)
    peaks,details=matrix_pencil(x,fs,options,[0,2])
    np.testing.assert_allclose(peaks,[2,1],rtol=1e-7)
    np.testing.assert_allclose([m['frequency'] for m in details['modes']],[.6,1.2],atol=1e-8)
    np.testing.assert_allclose([m['damping'] for m in details['modes']],[-.15,-.07],atol=1e-8)
    assert details['relative_fit_error']<1e-9


def test_fft_and_minmax_analytic():
    c=config();c['methods']=['fft','minmax'];c['fft'].update(preprocess='none',nfft=100)
    x=3*np.cos(2*np.pi*np.arange(100)/20)
    scores,details=extract_criteria(x,c)
    assert scores['fft']==pytest.approx(3)
    assert scores['minmax']==pytest.approx(6)
    assert details['fft']['peak_frequency']==pytest.approx(1)


def test_yule_walker_ar1():
    rng=np.random.default_rng(8); x=np.zeros(20000)
    for i in range(1,len(x)): x[i]=.7*x[i-1]+rng.normal()
    o=dict(preprocess='none',order=1,nfft=32768,ridge=0.,power_floor=1e-30)
    score,details=yule_walker(x,20,o,[0,2])
    assert details['innovation_variance']==pytest.approx(1,rel=.04)
    assert score==pytest.approx(10*np.log10(2/(20*(1-.7)**2)),abs=.5)


def test_two_mp_peaks_are_one_vote():
    c=config()
    refs=[{'fft':1.,'mp_1':1.,'mp_2':1.,'yule_walker':-10.,'minmax':1.}]*8
    scores={**refs[0],'mp_1':10.,'mp_2':10.}
    evidence,votes,event=decide(scores,refs,c)
    assert votes==['matrix_pencil'] and not event
    assert evidence['mp_1']['triggered'] and evidence['mp_2']['triggered']


def test_causal_replay_state_and_prefix(tmp_path):
    c=config(); seq=list(samples())
    expected=replay(seq,c)
    assert any(r['is_anomaly'] for r in expected)
    assert all(r['available_at']>=r['window_end'] for r in expected if 'window_end' in r)
    prefix=replay(seq[:901],c)
    assert stable([r for r in prefix if 'scores' in r])==stable([r for r in expected if 'scores' in r and r['available_at']<=45])
    detector=PhaseEventDetector(c);first=[]
    for sample in seq[:907]: first.extend(detector.update(**sample))
    detector.save_state(tmp_path/'state.json')
    resumed=PhaseEventDetector.load_state(tmp_path/'state.json',c)
    for sample in seq[907:]: first.extend(resumed.update(**sample))
    first.extend(resumed.flush())
    assert stable(first)==stable(expected)
    assert len(resumed.buffer)<=c['window_samples'] and len(resumed.history)<=c['threshold']['history_windows']


def test_wrap_and_bounded_gap_interpolation(tmp_path):
    c=config();c['methods']=['minmax'];c['consensus']['min_methods']=1
    detector=PhaseEventDetector(c);rows=[]
    for i in range(160):
        if i in (40,41): continue
        rows.extend(detector.update(i/20,(170+i+180)%360-180,(165+i+180)%360-180))
    scored=[r for r in rows if 'scores' in r]
    assert scored[0]['interpolated_samples']==2
    np.testing.assert_allclose(scored[0]['rpad_degrees'],5)
    assert scored[0]['scores']['minmax']==0
    rows=detector.update(20,0,0)
    assert any(r.get('reason')=='long_missing_interval' for r in rows)
    assert not detector.history
    with pytest.raises(ValueError,match='out-of-order'): detector.update(19,0,0)
    assert detector.update(20,100,0)[0]['reason']=='duplicate_ignored'
    with pytest.raises(ValueError,match='off the reporting'): detector.update(20.013,0,0)


def test_short_gap_at_window_end_waits_for_right_endpoint():
    c=config();d=PhaseEventDetector(c);result=[]
    for i in range(99): result.extend(d.update(i/20,i*.1,0))
    assert not d.update(99/20,None,None,False)
    result=d.update(100/20,10,0)
    assert result[0]['window_end']==pytest.approx(99/20)
    assert result[0]['available_at']==pytest.approx(5)
    assert result[0]['interpolated_samples']==1


def test_validation_and_frozen_thresholds():
    c=config();c['threshold']['mode']='frozen'
    rows=replay(samples(),c)
    thresholds=[r['evidence']['fft']['threshold'] for r in rows if r['status']=='evaluated']
    assert len(set(thresholds))==1
    c['threshold']['mode']='retrospective'
    with pytest.raises(ValueError,match='offline'): PhaseEventDetector(c)
    retrospective=replay(samples(),c)
    assert all(r['status']=='evaluated' for r in retrospective if 'scores' in r)
    c['yule_walker']['order']=1000
    with pytest.raises(ValueError,match='order'): validate_config(c)


def test_report_explains_each_anomaly_and_cli(tmp_path,monkeypatch):
    c=config();rows=replay(samples(),c)
    save_decisions(rows,tmp_path/'d.jsonl',c)
    index=render_report(tmp_path/'d.jsonl',tmp_path/'report')
    events=list((tmp_path/'report').glob('event-*.html'))
    assert len(events)==sum(r['is_anomaly'] for r in rows)>0
    assert 'Détection' in index.read_text()
    assert 'Dépassement' in events[0].read_text() and 'Seuil' in events[0].read_text()
    (tmp_path/'config.json').write_text(json.dumps(c))
    import io
    monkeypatch.setattr('sys.stdin',io.StringIO('\n'.join(json.dumps(s) for s in list(samples(130)))))
    monkeypatch.setattr('sys.stdout',io.StringIO())
    main(['phase-stream','--config',str(tmp_path/'config.json'),'--state-out',str(tmp_path/'state.json')])
    assert (tmp_path/'state.json').exists()


def test_excluded_missing_sample_is_not_interpolated():
    c=config();c['preprocessing']['excluded_intervals']=[[2.,2.]]
    d=PhaseEventDetector(c)
    for i in range(40): d.update(i/20,0.,0.)
    rows=d.update(2.05,0.,0.)
    assert any(r.get('reason')=='excluded_missing_sample' for r in rows)
    assert len(d.buffer)==1 and d.buffer[0][0]==pytest.approx(2.05)




def test_csv_cli_roundtrip(tmp_path):
    import csv
    c=config();c['methods']=['fft','minmax'];c['consensus']['min_methods']=2
    (tmp_path/'c.json').write_text(json.dumps(c))
    with (tmp_path/'angles.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['time','a','b','quality'])
        for s in samples(650): writer.writerow([s['timestamp'],s['angle_a'],s['angle_b'],1])
    main(['detect-phase','--config',str(tmp_path/'c.json'),'--csv',str(tmp_path/'angles.csv'),
          '--angle-a','a','--angle-b','b','--valid-column','quality','--output',str(tmp_path/'d.jsonl')])
    rows=[json.loads(line) for line in (tmp_path/'d.jsonl').read_text().splitlines()]
    assert any(r['status']=='evaluated' for r in rows)
