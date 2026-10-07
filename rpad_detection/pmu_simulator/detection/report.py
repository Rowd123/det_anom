"""Separate, offline HTML rendering of saved detection decisions."""
from html import escape
import json
from pathlib import Path


def render_report(decisions_path, output_dir):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.offline import get_plotlyjs
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    (directory/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf8')
    with open(decisions_path,encoding='utf-8-sig') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    anomalies = [r for r in rows if r.get('is_anomaly')]
    scored = [r for r in rows if 'scores' in r]
    names = list(scored[0]['scores']) if scored else []
    def page(title, body):
        return '<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+'</title><style>body{font:16px system-ui;margin:32px auto;max-width:1200px;padding:0 20px;color:#18343c}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{text-align:left;padding:10px;border-bottom:1px solid #c9dbde}.yes{background:#ffe6dc}a{color:#007c91}code{white-space:pre-wrap}h1{font-size:28px}</style><body>'+body+'</body></html>'
    def plot(fig):
        return fig.to_html(full_html=False,include_plotlyjs='plotly.min.js')
    def fmt(x): return '—' if x is None else f'{x:.6g}'
    links=[]
    for index,row in enumerate(anomalies,1):
        fig = go.Figure(go.Scatter(x=row['times'],y=row['rpad_degrees'],mode='lines',name='Déphasage',line_color='#008ba3'))
        fig.update_layout(template='plotly_white',title=f"Déphasage — {row['pair_id']}",xaxis_title='Temps (s)',yaxis_title='Différence angulaire (°)',height=400)
        table='<table><tr><th>Critère</th><th>Valeur</th><th>Moyenne passée / bloc</th><th>Écart-type</th><th>Seuil</th><th>Dépassement</th><th>Déclenché</th></tr>'
        for name,e in row['evidence'].items():
            table+=('<tr class="yes">' if e['triggered'] else '<tr>')+'<td>'+escape(name)+'</td>'+''.join('<td>'+fmt(e[k])+'</td>' for k in ('value','mean','std','threshold','excess'))+'<td>'+('Oui' if e['triggered'] else 'Non')+'</td></tr>'
        table+='</table>'
        title=f"Fenêtre {row['window_id']} : {row['window_start']:.3f}–{row['window_end']:.3f} s"
        description=f"<p>{row['vote_count']} méthodes sur {row['required_votes']} requises : {escape(', '.join(row['triggered_methods']))}. Les deux amplitudes Matrix Pencil comptent pour une seule méthode.</p><p>Décision disponible à {row['available_at']:.3f} s ; {row['interpolated_samples']} points interpolés. Temps de calcul : {row['compute_seconds']:.4f} s.</p>"
        body='<a href="index.html">Retour aux résultats</a><h1>'+escape(title)+'</h1>'+description+plot(fig)+table+'<details><summary>Fréquences et paramètres estimés</summary><pre>'+escape(json.dumps(row['details'],indent=2,ensure_ascii=False))+'</pre></details>'
        filename=f'event-{index:05d}.html'
        (directory/filename).write_text(page(title,body),encoding='utf8')
        links.append('<li><a href="'+filename+'">'+escape(title)+'</a> — '+escape(', '.join(row['triggered_methods']))+'</li>')
    figure=''
    if names:
        fig=make_subplots(rows=len(names),cols=1,shared_xaxes=True,subplot_titles=names)
        for i,name in enumerate(names,1):
            fig.add_trace(go.Scatter(x=[r['window_end'] for r in scored],y=[r['scores'][name] for r in scored],name=name,line_color='#008ba3'),row=i,col=1)
            fig.add_trace(go.Scatter(x=[r['window_end'] for r in scored],y=[r.get('evidence',{}).get(name,{}).get('threshold') for r in scored],name='Seuil '+name,line=dict(color='#dc774d',dash='dash')),row=i,col=1)
        fig.update_layout(template='plotly_white',height=240*len(names),showlegend=False)
        fig.update_xaxes(title_text='Fin de fenêtre (s)',row=len(names),col=1)
        figure=plot(fig)
    statuses={status:sum(r['status']==status for r in rows) for status in sorted({r['status'] for r in rows})}
    body='<h1>Détection d’événements par déphasage</h1><p>'+str(len(anomalies))+' fenêtres détectées. Les critères expliquent la décision, pas la cause physique de l’événement.</p><p>États : '+escape(str(statuses))+'</p><h2>Fenêtres détectées</h2>'+('<ol>'+''.join(links)+'</ol>' if links else '<p>Aucune fenêtre détectée.</p>')+figure
    (directory/'index.html').write_text(page('Détection RPAD',body),encoding='utf8')
    return directory/'index.html'
