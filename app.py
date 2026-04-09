from __future__ import annotations

import io
import json
import os

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from db import get_run, init_db, list_runs, save_run
from simulation import DEFAULT_CONFIG, run_sensitivity_analysis, simulate, summarize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'), static_folder=os.path.join(BASE_DIR, 'static'))
app.secret_key = 'dev-secret-change-me'

init_db()


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return int(default)


def build_config_from_form(form) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg['n_patients'] = _safe_int(form.get('n_patients'), cfg['n_patients'])
    cfg['seed'] = _safe_int(form.get('seed'), cfg['seed'])
    cfg['max_attempts'] = _safe_int(form.get('max_attempts'), cfg['max_attempts'])
    cfg['lambda_per_week'] = _safe_float(form.get('lambda_per_week'), cfg['lambda_per_week'])
    cfg.setdefault('access', {})['daily_capacity'] = _safe_int(
        form.get('daily_capacity'), cfg.get('access', {}).get('daily_capacity', 8)
    )

    for pop in cfg['populations'].keys():
        key = f'pop_weight__{pop}'
        if key in form:
            cfg['populations'][pop]['weight'] = _safe_float(form.get(key), cfg['populations'][pop]['weight'])

    advanced_on = form.get('advanced_on') == '1'
    if advanced_on:
        pop_params_json = (form.get('population_params_json') or '').strip()
        tp_json = (form.get('avg_touchpoints_json') or '').strip()
        alloc_json = (form.get('allocated_minutes_json') or '').strip()
        if pop_params_json:
            cfg['population_params'] = json.loads(pop_params_json)
        if tp_json:
            cfg['avg_touchpoints_by_method'] = json.loads(tp_json)
        if alloc_json:
            cfg['allocated_minutes_by_visit_category'] = json.loads(alloc_json)
    return cfg


@app.get('/')
def index():
    return render_template('index.html', default_cfg=DEFAULT_CONFIG)


@app.post('/run')
def run_sim():
    try:
        cfg = build_config_from_form(request.form)
        df = simulate(cfg)
        summ = summarize(df)
        run_id = save_run(cfg, summ, df.to_csv(index=False))
        return redirect(url_for('archive_run', run_id=run_id))
    except Exception as e:
        flash(str(e), 'danger')
        return redirect(url_for('index'))


@app.get('/archive')
def archive():
    runs = list_runs(limit=100)
    return render_template('archive.html', runs=runs)


@app.get('/archive/<int:run_id>')
def archive_run(run_id: int):
    run = get_run(run_id)
    if not run:
        flash('Run not found.', 'warning')
        return redirect(url_for('archive'))

    df = pd.read_csv(io.StringIO(run['csv_text']))
    sensitivity = run_sensitivity_analysis(run['config'])
    baseline = next((s for s in sensitivity if s['category'] == 'Baseline'), None)
    if baseline is None:
        baseline = {'completed_rate': 0.0, 'avg_touchpoints': 0.0, 'avg_total_time': 0.0, 'avg_access_delay': 0.0, 'mean_utilization': 0.0}

    for s in sensitivity:
        s['delta_completion'] = s.get('completed_rate', 0.0) - baseline.get('completed_rate', 0.0)
        s['delta_touchpoints'] = s.get('avg_touchpoints', 0.0) - baseline.get('avg_touchpoints', 0.0)
        s['delta_time'] = s.get('avg_total_time', 0.0) - baseline.get('avg_total_time', 0.0)
        s['delta_access_delay'] = s.get('avg_access_delay', 0.0) - baseline.get('avg_access_delay', 0.0)
        s['delta_utilization'] = s.get('mean_utilization', 0.0) - baseline.get('mean_utilization', 0.0)

    summary = run['summary'] or {}
    access_summary = summary.setdefault('access', {})
    access_summary.setdefault('mean_access_delay', 0.0)
    access_summary.setdefault('mean_utilization', 0.0)
    access_summary.setdefault('max_queue', 0)
    access_summary.setdefault('timeline', [])
    access_summary.setdefault('days_simulated', 0)
    access_summary.setdefault('request_to_appt_p90', 0.0)
    access_summary.setdefault('request_to_appt_p95', 0.0)

    return render_template(
        'results.html',
        run_id=run_id,
        summary=summary,
        sensitivity=sensitivity,
        preview=df.head(25).to_dict(orient='records'),
        columns=list(df.columns),
        access_timeline=access_summary.get('timeline', []),
        created_at=run['created_at'],
    )


@app.get('/archive/<int:run_id>/download')
def archive_download(run_id: int):
    run = get_run(run_id)
    if not run:
        flash('Run not found.', 'warning')
        return redirect(url_for('archive'))
    csv_bytes = run['csv_text'].encode('utf-8')
    return send_file(io.BytesIO(csv_bytes), mimetype='text/csv', as_attachment=True, download_name=f'appt_sim_run_{run_id}.csv')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
