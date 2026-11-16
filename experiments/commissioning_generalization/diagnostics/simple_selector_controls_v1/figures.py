"""Four paper figure families, standalone PGFPlots sources and vector PDFs.

When no LaTeX executable is installed, PDFs are rendered by Matplotlib and the
manifest explicitly says the accompanying PGFPlots sources were not compiled.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cgx.provenance import atomic_json, file_sha256, now, read_json
from .reporting import METHODS, LEARNED, SIMPLE, ONE_PASS, PRIMARY

ORDER = (*SIMPLE, *LEARNED, 'full_recording_feasible')
SHORT = {
    'full_once_100': 'One full execution', 'full_once_100_cal2': 'One full + calibration',
    'random_legal': 'Random legal', 'coverage_forward': 'Forward coverage',
    'fixed_template_validation': 'Validation-fixed plan', 'interface_rule_se': 'Graph rule',
    'gnn_fixed_plan': 'GNN Plan-Once', 'gnn_adaptive': 'GNN Replan',
    'gnn_depth2': 'GNN 2-Step', 'cost_aware_adaptation': 'GNN Cost-Aware',
    'transformer_adaptive': 'Transformer Replan', 'full_recording_feasible': 'Four-run full recording',
}
# Preserve the preceding study's five learned-method colors; extend, do not
# choose favorable colors or change them after seeing a method's outcome.
COLORS = {
    'gnn_fixed_plan': '#1f77b4', 'gnn_adaptive': '#ff7f0e', 'gnn_depth2': '#2ca02c',
    'transformer_adaptive': '#d62728', 'cost_aware_adaptation': '#9467bd',
    'full_recording_feasible': '#000000', 'full_once_100': '#17becf',
    'full_once_100_cal2': '#bcbd22', 'random_legal': '#7f7f7f',
    'coverage_forward': '#8c564b', 'fixed_template_validation': '#e377c2',
    'interface_rule_se': '#237c79',
}
SUITE_LABELS = {'test_connections': 'Unseen connections (primary)',
                'test_seen': 'Seen connections', 'test_size': 'Larger unseen systems'}
MODE_LABELS = {'online': 'Original online limits', 'completed_read': 'Completed diagnosis on identical saved evidence'}
SHORT_MODE_LABELS = {'online': 'Original online limits', 'completed_read': 'Completed same-evidence diagnosis'}
COMMON = 'All methods use the same frozen Structured Evidence diagnoser'


def _tex(text):
    return str(text).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('#', r'\#').replace('\n', r'\\')


def _pdf_png(fig, path):
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def _curve(ax, part, metric, scale):
    for method in ORDER:
        rows = part[part.method == method].sort_values('execution_cap')
        if rows.empty:
            continue
        one_pass = method in ONE_PASS
        label = SHORT[method] + (' (same B1 trial)' if one_pass else '')
        ax.plot(rows.execution_cap, scale * rows[metric], color=COLORS[method],
                marker='^' if one_pass else 's' if method == 'full_recording_feasible' else 'o',
                markerfacecolor='none' if one_pass else COLORS[method], markersize=4,
                linewidth=1.2, linestyle=':' if one_pass else '--' if method in LEARNED else '-', label=label)
    ax.set_xticks([1, 2, 4]);ax.grid(alpha=.2)


def _standalone_curve(directory, name, part, metric, scale, ylabel, title):
    path = directory / name
    data = part[['method', 'execution_cap', metric, 'rows', 'physical_cases',
                 'healthy_rows', 'fault_rows', 'correct_count', 'false_alarm_count',
                 'miss_count', 'reference_view_rows']].copy()
    data['plotted_value'] = scale * data[metric]
    data.to_csv(path.with_suffix('.csv'), index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _curve(ax, part, metric, scale)
    ax.set(xlabel='Additional-execution cap B', ylabel=ylabel, title=title)
    ax.legend(ncol=3, fontsize=6.3, loc='upper center', bbox_to_anchor=(.5, -.17))
    fig.suptitle(COMMON, fontsize=9)
    fig.tight_layout();_pdf_png(fig, path)
    plot_commands = []
    for method in ORDER:
        rows = part[part.method == method].sort_values('execution_cap')
        if rows.empty:
            continue
        coordinates = ' '.join(f'({int(r.execution_cap)},{float(getattr(r,metric))*scale:.17g})' for r in rows.itertuples())
        color = 'c' + str(ORDER.index(method))
        style = 'densely dotted,mark=triangle,mark options={fill=white}' if method in ONE_PASS else 'dashed,mark=*' if method in LEARNED else 'solid,mark=*'
        label = SHORT[method] + (' (same B1 trial)' if method in ONE_PASS else '')
        plot_commands += [f'\\addplot[{color},{style}] coordinates {{{coordinates}}};',
                          f'\\addlegendentry{{{_tex(label)}}}']
    _write_tex(path, title, 'Additional-execution cap B', ylabel, plot_commands,
               'xtick={1,2,4},legend columns=3,legend style={at={(0.5,-0.22)},anchor=north,font=\\scriptsize},')


def _write_tex(path, title, xlabel, ylabel, commands, extra=''):
    definitions = '\n'.join(f'\\definecolor{{c{i}}}{{HTML}}{{{COLORS[m][1:]}}}' for i, m in enumerate(ORDER))
    source = (r'\documentclass[tikz,border=4pt]{standalone}' + '\n' +
        r'\usepackage{pgfplots}' + '\n' + r'\pgfplotsset{compat=1.18}' + '\n' + definitions + '\n' +
        r'\begin{document}\begin{tikzpicture}' + '\n' +
        '\\begin{axis}[width=15cm,height=9cm,grid=major,grid style={opacity=.2},' + extra +
        f'title={{{_tex(COMMON)}\\\\{_tex(title)}}},title style={{align=center,font=\\small}},'
        f'xlabel={{{_tex(xlabel)}}},ylabel={{{_tex(ylabel)}}},tick label style={{font=\\small}}]\n' +
        '\n'.join(commands) + '\n' + r'\end{axis}\end{tikzpicture}\end{document}' + '\n')
    path.with_suffix('.tex').write_text(source)


def _denominator(part, metric):
    column, label = ('healthy_rows', 'Healthy') if metric == 'healthy_false_alarm_rate' else ('fault_rows', 'Faulty')
    deterministic = part[part.method.isin(SIMPLE) & (part.method != 'random_legal')][column].unique()
    seeded = part[part.method.isin(LEARNED) | (part.method == 'random_legal')][column].unique()
    if len(deterministic) != 1 or len(seeded) != 1:
        raise ValueError('Figure denominator mismatch')
    return f'{label} N: deterministic {int(deterministic[0])}; three-seed {int(seeded[0])}'


def _budget_and_errors(out, table, suites):
    for family, metrics in [
            ('01_accuracy_nll_vs_budget', [('accuracy', 100., 'Exact diagnosis accuracy (%)'), ('nll_bits', 1., 'True-label NLL (bits)')]),
            ('03_false_alarms_and_misses', [('healthy_false_alarm_rate', 100., 'Healthy false alarms (%)'), ('fault_miss_rate', 100., 'Actual-fault misses (%)')])]:
        for mode in ('online', 'completed_read'):
            fig, axes = plt.subplots(2, len(suites), figsize=(6.5 * len(suites), 8), squeeze=False)
            for ci, suite in enumerate(suites):
                part = table[(table.suite == suite) & (table.result_mode == mode)]
                for ri, (metric, scale, label) in enumerate(metrics):
                    ax = axes[ri, ci];_curve(ax, part, metric, scale)
                    denominator = _denominator(part, metric) if family.startswith('03') else (
                        f"Physical cases {int(part.physical_cases.max())}; seeds averaged for comparisons")
                    ax.set(xlabel='Additional-execution cap B', ylabel=label,
                           title=SUITE_LABELS[suite] + '\n' + denominator)
                    _standalone_curve(out/'subfigures', f'{family}_{suite}_{mode}_{metric}', part,
                                      metric, scale, label, f'{SUITE_LABELS[suite]} | {MODE_LABELS[mode]}\n{denominator}')
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(.5, -.11), fontsize=8)
            fig.suptitle(COMMON + '\n' + MODE_LABELS[mode], fontsize=12)
            fig.tight_layout();_pdf_png(fig, out/f'{family}_{mode}')


def _tradeoffs(out, table, suites):
    fig, axes = plt.subplots(2, len(suites), figsize=(6.5*len(suites), 8), squeeze=False)
    for ri, mode in enumerate(('online', 'completed_read')):
        for ci, suite in enumerate(suites):
            part = table[(table.suite == suite) & (table.result_mode == mode)]
            # A one-pass reference has one actual point, not three independent
            # points just because it is feasible at three execution caps.
            part = part[~(part.method.isin(ONE_PASS) & (part.execution_cap != 1))].copy()
            part['recorded_MB'] = part.mean_bytes_used/1e6
            part['accuracy_percent'] = part.accuracy*100
            name = f'02_accuracy_recorded_MB_{suite}_{mode}'
            selected = ['method', 'execution_cap', 'recorded_MB', 'accuracy_percent',
                        'mean_executions_used', 'mean_calibrations_used', 'rows', 'correct_count']
            part[selected].to_csv(out/'subfigures'/f'{name}.csv', index=False)
            small, axis = plt.subplots(figsize=(7.2, 5.0))
            commands = []
            for target in (axes[ri, ci], axis):
                for method in ORDER:
                    rows = part[part.method == method].sort_values('execution_cap')
                    if rows.empty:
                        continue
                    target.plot(rows.recorded_MB, rows.accuracy_percent, color=COLORS[method], alpha=.5, linewidth=.7)
                    for index, row in enumerate(rows.itertuples()):
                        calibrated = row.mean_calibrations_used > 0
                        target.scatter(row.recorded_MB, row.accuracy_percent,
                            s=22+15*row.mean_executions_used,
                            facecolors='none' if method in ONE_PASS else COLORS[method],
                            marker='s' if calibrated else 'o',
                            edgecolors=COLORS[method] if method in ONE_PASS else 'white',
                            linewidth=1.2 if method in ONE_PASS else .5,
                            zorder=5 if method in ONE_PASS else 3,
                            label=SHORT[method] if index == 0 else None)
                for index, method in enumerate(ONE_PASS):
                    row = part[part.method == method].iloc[0]
                    label = SHORT[method]
                    fixed = part[part.method == 'fixed_template_validation']
                    overlap = (np.isclose(fixed.recorded_MB, row.recorded_MB, rtol=0, atol=1e-12) &
                               np.isclose(fixed.accuracy_percent, row.accuracy_percent, rtol=0, atol=1e-12)).any()
                    if overlap:
                        label += ' / validation-fixed overlap'
                    target.annotate(label, (row.recorded_MB, row.accuracy_percent),
                        xytext=(18, -35-25*index), textcoords='offset points', fontsize=6.8,
                        color=COLORS[method], arrowprops=dict(arrowstyle='-', color=COLORS[method], linewidth=.6),
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.5))
                target.set(xlabel='Additional recorded MB (decimal)', ylabel='Exact diagnosis accuracy (%)',
                    title=f'{SUITE_LABELS[suite]}\n{SHORT_MODE_LABELS[mode]}\nSize: executions; square: calibration used; circle: none')
                target.grid(alpha=.2)
            for method in ORDER:
                rows = part[part.method == method].sort_values('execution_cap')
                for index, row in enumerate(rows.itertuples()):
                    marker = 'square*' if row.mean_calibrations_used > 0 else '*'
                    if method in ONE_PASS:
                        marker = 'square' if row.mean_calibrations_used > 0 else 'o'
                    size = 2+.5*row.mean_executions_used
                    commands.append(f'\\addplot[only marks,c{ORDER.index(method)},mark={marker},mark size={size:.4f}pt] coordinates {{({row.recorded_MB:.17g},{row.accuracy_percent:.17g})}};')
                    if index == 0:commands.append(f'\\addlegendentry{{{_tex(SHORT[method])}}}')
            for index, method in enumerate(ONE_PASS):
                row = part[part.method == method].iloc[0]
                fixed = part[part.method == 'fixed_template_validation']
                overlap = (np.isclose(fixed.recorded_MB, row.recorded_MB, rtol=0, atol=1e-12) &
                           np.isclose(fixed.accuracy_percent, row.accuracy_percent, rtol=0, atol=1e-12)).any()
                label = SHORT[method] + (' / validation-fixed overlap' if overlap else '')
                commands.append(f'\\node[anchor=north west,font=\\scriptsize,text=c{ORDER.index(method)},xshift=8pt,yshift={-20-15*index}pt] at (axis cs:{row.recorded_MB:.17g},{row.accuracy_percent:.17g}) {{{_tex(label)}}};')
            axis.legend(ncol=3, fontsize=6.3, loc='upper center', bbox_to_anchor=(.5, -.2))
            small.suptitle(COMMON, fontsize=9);small.tight_layout();_pdf_png(small,out/'subfigures'/name)
            _write_tex(out/'subfigures'/name, f'{SUITE_LABELS[suite]}\n{SHORT_MODE_LABELS[mode]}\nSize: executions; square: calibration; circle: none',
                'Additional recorded MB (decimal)', 'Exact diagnosis accuracy (%)', commands,
                'legend columns=3,legend style={at={(0.5,-0.22)},anchor=north,font=\\scriptsize},')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(.5, -.1), fontsize=8)
    fig.suptitle(COMMON+'\nMarker area increases with executions; calibration shape is separate. Exact E/C costs are in CSV.',fontsize=12)
    fig.tight_layout();_pdf_png(fig,out/'02_accuracy_vs_recorded_MB')


def _configuration(out, configs):
    source = configs[configs.suite == 'test_connections']
    identifiers = sorted(source.configuration_id.unique())
    mapping = {name: index+1 for index, name in enumerate(identifiers)}
    pd.DataFrame([dict(plot_index=i, configuration_id=name) for name,i in mapping.items()]).to_csv(out/'configuration_labels.csv', index=False)
    for mode in ('online','completed_read'):
        fig, axes = plt.subplots(2,4,figsize=(19,8),squeeze=False)
        for ci,(learned,simple) in enumerate(PRIMARY):
            part=source[(source.result_mode==mode)&(source.learned_method==learned)&(source.simple_method==simple)].copy()
            part['plot_index']=part.configuration_id.map(mapping);part=part.sort_values('plot_index')
            for ri,(metric,scale,label) in enumerate([('accuracy',100.,'Accuracy difference (pp)'),('nll_bits',1.,'NLL difference (bits)')]):
                name=f'04_configuration_{mode}_{learned}_vs_{simple}_{metric}'
                data=part[['configuration_id','plot_index','physical_cases',metric]].copy();data['plotted_difference']=scale*data[metric]
                data.to_csv(out/'subfigures'/f'{name}.csv',index=False)
                small,axis=plt.subplots(figsize=(7.2,4.4))
                for target in (axes[ri,ci],axis):
                    target.axhline(0,color='#777777',linewidth=.8)
                    target.plot(part.plot_index,scale*part[metric],marker='o',markersize=3.5,color=COLORS[simple])
                    target.set(xlabel='Unseen configuration index',ylabel=label,
                               title=f'{SHORT[simple]} minus {SHORT[learned]}')
                    target.set_xticks(range(1,len(identifiers)+1));target.tick_params(axis='x',labelsize=7);target.grid(alpha=.2)
                small.suptitle(COMMON+'\n'+MODE_LABELS[mode],fontsize=9)
                small.tight_layout();_pdf_png(small,out/'subfigures'/name)
                coords=' '.join(f'({int(r.plot_index)},{float(getattr(r,metric))*scale:.17g})' for r in part.itertuples())
                commands=[f'\\addplot[gray] coordinates {{(1,0) ({len(identifiers)},0)}};',
                          f'\\addplot[c{ORDER.index(simple)},mark=*] coordinates {{{coords}}};']
                _write_tex(out/'subfigures'/name,f'{SHORT[simple]} minus {SHORT[learned]} | {MODE_LABELS[mode]}',
                           'Unseen configuration index',label,commands)
        fig.suptitle(COMMON+'\n'+MODE_LABELS[mode]+' | Within-case seed averages; simple minus learned',fontsize=12)
        fig.tight_layout();_pdf_png(fig,out/f'04_per_configuration_paired_differences_{mode}')


def render(run_dir,phase='all'):
    run=Path(run_dir);reports=run/'reports'/'primary' if phase=='primary' else run/'reports'
    marker=read_json(reports/'complete.json');assert marker['state']=='passed'
    out=run/'figures'/'primary' if phase=='primary' else run/'figures'
    (out/'subfigures').mkdir(parents=True,exist_ok=True)
    table=pd.read_csv(reports/'main_comparison.csv');configs=pd.read_csv(reports/'per_configuration_paired_differences.csv')
    suites=['test_connections'] if phase=='primary' else ['test_seen','test_connections','test_size']
    _budget_and_errors(out,table,suites);_tradeoffs(out,table,suites);_configuration(out,configs)
    latex=shutil.which('pdflatex');compilation=[]
    if latex:
        compiled=out/'tex_compiled';compiled.mkdir(exist_ok=True)
        for source in sorted((out/'subfigures').glob('*.tex')):
            result=subprocess.run([latex,'-interaction=nonstopmode','-halt-on-error',f'-output-directory={compiled}',str(source)],
                cwd=source.parent,text=True,capture_output=True,timeout=120)
            (compiled/(source.stem+'.stdout.log')).write_text(result.stdout+'\n'+result.stderr)
            compilation.append(dict(source=str(source.relative_to(out)),exit_code=result.returncode,
                pdf_exists=(compiled/(source.stem+'.pdf')).is_file()))
    status='compiled_all' if latex and all(x['exit_code']==0 and x['pdf_exists'] for x in compilation) else 'compile_failures' if latex else 'unavailable'
    source_status='PGFPlots sources retained; vector PDFs rendered by Matplotlib because LaTeX is unavailable' if not latex else 'PGFPlots compilation status recorded individually; Matplotlib vector PDFs also retained'
    (out/'README.md').write_text('# Four figure families\n\n'+COMMON+'.\n\n'+source_status+'.\n\n'
        '01: accuracy/NLL versus execution cap; 02: accuracy versus additional MB with execution-size and calibration-shape annotations; '
        '03: healthy false alarms/fault misses with explicit conditional denominators; 04: paired per-configuration differences. '
        'Online and completed-reading results are distinct. One-pass budget references repeat the same B1 trial and are not independent observations. '
        'Every standalone subfigure has a .tex source, vector .pdf, .png preview and the exact plotted .csv data. '
        'Learned-method colors preserve the preceding Structured Evidence figures; new baseline colors are fixed in figures.py.\n')
    files={str(p.relative_to(out)):file_sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='complete.json'}
    record=dict(state='passed',phase=phase,completed_at=now(),required_figure_families=4,
        all_methods_share_frozen_SED=True,latex_status=status,latex_executable=latex,
        pdf_rendering_backend='matplotlib vector; optional separate LaTeX compilation',
        standalone_tex=len(list((out/'subfigures').glob('*.tex'))),
        standalone_pdfs=len(list((out/'subfigures').glob('*.pdf'))),
        compilation=compilation,input_report_sha256=file_sha256(reports/'complete.json'),files=files)
    atomic_json(out/'complete.json',record);return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    parser.add_argument('--phase',choices=('primary','all'),default='all');args=parser.parse_args()
    print(json.dumps(render(args.run_dir,args.phase),indent=2),flush=True)
