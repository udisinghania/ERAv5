"""Summarize all outcomes, including unfavorable controls, without cherry-picking."""
import json
from pathlib import Path
import statistics
import subprocess
import os
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor, Color, white

ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent/'Session_14_MoE'
NODE=Path(r'C:\Users\udisi\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
SHARP=Path(r'C:\Users\udisi\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules\sharp')
COLORS=[HexColor('#137f91'),HexColor('#bd6029')]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(drawing,name):
    renderSVG.drawToFile(drawing,str(ROOT/(name+'.svg')))
    script="require(process.argv[1])(process.argv[2],{density:160}).png().toFile(process.argv[3]).catch(e=>{console.error(e);process.exit(1)});"
    scratch=ROOT.parent.parent/'work'/'chart_fonts';scratch.mkdir(parents=True,exist_ok=True)
    config=scratch/'fonts.conf'
    config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig>'
        '<dir>C:/Windows/Fonts</dir><cachedir>'+scratch.as_posix()+'</cachedir>'
        '<alias><family>Helvetica</family><prefer><family>Arial</family></prefer></alias>'
        '<alias><family>sans-serif</family><prefer><family>Arial</family></prefer></alias></fontconfig>')
    env=os.environ.copy();env['FONTCONFIG_FILE']=str(config)
    subprocess.run([str(NODE),'-e',script,str(SHARP),str(ROOT/(name+'.svg')),str(ROOT/(name+'.png'))],check=True,env=env)
def panel(d,x,y,title,series,ylabel,ymin=None):
    d.add(String(x,y+245,title,fontName='Helvetica-Bold',fontSize=13,fillColor=HexColor('#182d3c')))
    chart=LinePlot();chart.x=x+52;chart.y=y+42;chart.width=365;chart.height=170
    chart.data=series;chart.joinedLines=1
    for i in range(len(series)):
        chart.lines[i].strokeColor=COLORS[i];chart.lines[i].strokeWidth=2
    chart.xValueAxis.valueMin=0;chart.xValueAxis.valueMax=50;chart.xValueAxis.valueSteps=[0,10,20,30,40,50]
    values=[p[1] for s in series for p in s]
    margin=max(0.005,(max(values)-min(values))*0.1)
    chart.yValueAxis.valueMin=min(values)-margin if ymin is None else ymin
    chart.yValueAxis.valueMax=max(values)+margin
    chart.yValueAxis.labelTextFormat='%.3f';chart.yValueAxis.labels.fontSize=8;chart.xValueAxis.labels.fontSize=8
    chart.yValueAxis.visibleGrid=1;chart.yValueAxis.gridStrokeColor=HexColor('#e1e8ed')
    d.add(chart);d.add(String(x+52,y+221,ylabel,fontSize=9,fillColor=HexColor('#526473')))
    d.add(String(x+235,y+12,'Continuation tokens (millions)',fontSize=10,textAnchor='middle'))


results={k:read(ROOT/k/'result.json') for k in ('moe','dense')}
parents={k:read(PARENT/k/'result.json') for k in ('moe','dense')}
audit=read(ROOT/'verification.json');diag=read(ROOT/'diagnostics/summary.json')
original=read(PARENT/'dense_pretraining_result.json')
baseline=original['final_validation']['cross_entropy_nats']
series=[];points={}
for kind in ('moe','dense'):
    point=[(0,parents[kind]['initial_validation']['cross_entropy_nats']),(10,parents[kind]['final_validation']['cross_entropy_nats'])]
    for nominal in (10,20,30):
        r=read(ROOT/kind/f'validation_{nominal}m.json');point.append((r['cumulative_tokens']/1e6,r['cross_entropy_nats']))
    point.append((50,results[kind]['final_validation']['cross_entropy_nats']))
    points[kind]=point;series.append(point)
gap=[(m[0],d[1]-m[1]) for m,d in zip(points['moe'],points['dense'])]
d=Drawing(950,385);d.add(Rect(0,0,950,385,fillColor=white,strokeColor=None))
d.add(String(28,350,'More training: does the MoE advantage persist?',fontName='Helvetica-Bold',fontSize=21,fillColor=HexColor('#182d3c')))
d.add(String(28,327,'Paired continuation on the same 50M reused targets | full held-out evaluations | one training seed',fontSize=11))
for i,(label,color) in enumerate(zip(('MoE, 4 experts / top-2','Dense control'),COLORS)):
    d.add(Rect(595+i*173,307,10,6,fillColor=color,strokeColor=None));d.add(String(610+i*173,305,label,fontSize=9))
panel(d,28,35,'Full held-out loss',series,'Task cross-entropy (nats)')
panel(d,493,35,'MoE advantage at equal token exposure',[gap],'Dense loss minus MoE loss (nats)',ymin=min(0,min(v for _,v in gap))-0.002)
d.add(String(28,13,'Learning-rate schedule restarted at 10M for both arms. Equal tokens do not mean equal computation.',fontSize=10,fillColor=HexColor('#526473')))
save(d,'learning_comparison')

old=parents['moe']['final_validation']['routing'];new=results['moe']['final_validation']['routing']
d=Drawing(900,435);d.add(Rect(0,0,900,435,fillColor=white,strokeColor=None))
d.add(String(25,405,'How expert use changed with more training',fontName='Helvetica-Bold',fontSize=20))
d.add(String(25,382,'Each cell is its share of routed assignments within a layer. Uniform use would be 25%.',fontSize=11))
for left,title,rows in [(25,'After 10M continuation tokens',old),(470,'After 50M continuation tokens',new)]:
    d.add(String(left,352,title,fontName='Helvetica-Bold',fontSize=12))
    for col in range(4):d.add(String(left+103+col*77,329,f'E{col}',textAnchor='middle',fontSize=10))
    for row,values in enumerate(rows):
        y=295-row*29;d.add(String(left,y+8,f'Layer {row+1}',fontSize=10))
        for col,share in enumerate(values['shares']):
            t=min(1,share/0.5);color=Color(0.92-0.81*t,0.96-0.47*t,0.97-0.40*t)
            d.add(Rect(left+67+col*77,y,72,25,fillColor=color,strokeColor=white))
            d.add(String(left+103+col*77,y+7,f'{share*100:.1f}%',fontSize=10,textAnchor='middle',fillColor=white if t>0.6 else HexColor('#182d3c')))
    d.add(String(left,25,f"Dead experts: {sum(v['dead_experts'] for v in rows)} / 36",fontSize=11))
save(d,'expert_usage_comparison')

mf=results['moe']['final_validation'];df=results['dense']['final_validation']
ml=mf['cross_entropy_nats'];dl=df['cross_entropy_nats'];parent_loss=parents['moe']['final_validation']['cross_entropy_nats']
experiments=diag['experiments'];learned=next(r for r in experiments if r['mode']=='learned')
uniform=next(r for r in experiments if r['mode']=='uniform_selected')
random=[r for r in experiments if r['mode']=='random'];random_mean=statistics.mean(r['cross_entropy_nats'] for r in random)
random_sd=statistics.stdev(r['cross_entropy_nats'] for r in random)
uniform_delta=uniform['cross_entropy_nats']-ml;random_delta=random_mean-ml
source_dense={r['source']:r for r in df['sources']}
wins=sum(r['cross_entropy_nats']<source_dense[r['source']]['cross_entropy_nats'] for r in mf['sources'])
dead=sum(r['dead_experts'] for r in mf['routing'])
first_top2=sum(sorted(mf['routing'][0]['shares'],reverse=True)[:2])
old_first_top2=sum(sorted(parents['moe']['final_validation']['routing'][0]['shares'],reverse=True)[:2])
lines=['# Session 14: extended experiment findings','',
f'**MoE held-out loss reached {ml:.6f} nats**, compared with {parent_loss:.6f} after the first 10M-token continuation and {baseline:.6f} before conversion. The dense control reached {dl:.6f} on exactly the same continued training data.','',
'## Findings','',
f'1. **More training helped.** The extra 40M targets reduced MoE held-out loss by {parent_loss-ml:.6f} nats. Relative to the original dense checkpoint, loss fell {(baseline-ml)/baseline*100:.2f}% and perplexity fell from {original["final_validation"]["perplexity"]:.2f} to {mf["perplexity"]:.2f}.',
f'2. **The paired architecture comparison:** dense loss minus MoE loss is {dl-ml:.6f} nats after 50M continuation targets, versus {parents["dense"]["final_validation"]["cross_entropy_nats"]-parent_loss:.6f} after 10M. Positive values favor the MoE. The MoE has lower loss on {wins} of the 10 held-out source groups. This is a descriptive result for one training seed, not proof of general superiority.',
f'3. **Routing sensitivity:** random expert selection has mean loss {random_mean:.6f} across three inference seeds, {random_delta:+.6f} nats relative to normal routing. Uniform weights over the learned choices give {uniform["cross_entropy_nats"]:.6f}, a change of {uniform_delta:+.6f}. The table below reports every control.',
f'4. **Loss improved while the first layer became more concentrated.** Its two busiest experts take {first_top2*100:.2f}% of assignments, up from {old_first_top2*100:.2f}% after the first 10M tokens. Zero completely unused experts therefore coexists with substantial underuse of two experts in this layer.',
'5. **Generation quality remains weak.** Both final models produced repetitive continuations on all six fixed greedy-decoding prompts. The loss improvement has not yet translated into fluent text in these examples. Every output is retained in `GENERATIONS.md`.',
'',
'![Learning comparison](learning_comparison.png)','',
'## Full held-out progression','',
'| Continuation targets (millions) | MoE loss | Dense loss | Dense minus MoE |',
'|---:|---:|---:|---:|']
for (tokens,m),(dt,dense_loss) in zip(points['moe'],points['dense']):
    assert abs(tokens-dt)<1e-9
    lines.append(f'| {tokens:.3f} | {m:.6f} | {dense_loss:.6f} | {dense_loss-m:+.6f} |')
lines += ['',
'All full evaluations use the same 5,049,456 held-out supervised targets. Intermediate milestones occur at optimizer-step boundaries, so the table gives actual counts. The small fixed training-time probe is separate and is not mixed into this table.',
'',
'## What was changed','',
'The earlier experiment stopped after 10M continuation targets. This extension resumed each matching model and AdamW state for the remaining 40M targets of the same seeded corpus permutation. A new common learning-rate schedule warmed up to 0.0002 and decayed to 0.00002. No architecture, tokenizer, context length, auxiliary-loss coefficient, or data mixture was changed.',
'',
'Together the two stages consume exactly one full 50M-target continuation pass. Those targets were already used during the original 50M dense pretraining. Total exposure is therefore 100M targets per arm, drawn from the same existing 50M-target corpus. This is more optimization on reused training data, not a larger unique dataset.',
'',
'The MoE has 54.69M total / 31.68M active parameters versus 20.17M for the dense model. The comparison matches token exposure and optimization settings, not FLOPs, model capacity, or elapsed time.',
'',
'## Routing ablations: same final weights','',
'| Policy | Full held-out loss | Perplexity | Difference from learned |',
'|---|---:|---:|---:|']
for r in experiments:
    label={'learned':'Learned top-2 and learned weights','uniform_selected':'Learned top-2, uniform 1/2 weights','random':'Random top-2, uniform 1/2 weights'}[r['mode']]
    if r['mode']=='random':label+=f" (seed {r['seed']})"
    lines.append(f"| {label} | {r['cross_entropy_nats']:.6f} | {r['perplexity']:.2f} | {r['cross_entropy_nats']-ml:+.6f} |")
lines += ['',f'The random-control mean is {random_mean:.6f} nats; its sample standard deviation across the three routing seeds is {random_sd:.6f}. These seeds describe randomness during inference, not uncertainty across independently trained models. Every policy still executes two experts per non-padding token.', '']
if random_delta>0:
    lines.append('Random selection worsens this model, showing that the learned expert assignments matter after continued training. This does not establish that a model trained from the beginning with random routing would have the same deficit.')
else:
    lines.append('Random selection did not worsen the average loss in this diagnostic. The result does not support claiming a benefit from learned selection for this checkpoint.')
lines.append('')
if uniform_delta>0:
    lines.append('Replacing learned mixture weights with equal weights also worsens loss. Both selection and weighting therefore affect the final result in these inference controls.')
else:
    lines.append('Equal weights match or improve the learned weighting in this control. That suggests the learned mixture weighting is not clearly beneficial at this checkpoint, even if learned selection is useful.')
lines += ['', '## Expert utilization', '',
f'The final full validation pass has **{dead} unused experts out of 36**. Load remains uneven. The comparison below shows how the allocation changed between the two checkpoints; uniform loading would be 25% per expert.', '',
f'The first layer is the strongest imbalance: two experts account for {first_top2*100:.2f}% of its assignments. This concentration also matters when interpreting random routing: the intervention can send inputs to experts that received comparatively few training updates. A random-routing penalty alone does not prove semantic specialization.', '',
'![Expert allocation](expert_usage_comparison.png)', '',
'Per-source, per-layer counts are available in `diagnostics/learned.json`. Differences between source groups are descriptive routing patterns and do not, by themselves, demonstrate topic or language expertise.', '',
'## Per-source held-out results', '',
'| Source | Original dense | MoE after 10M | MoE after 50M | Dense after 50M | Dense minus MoE |',
'|---|---:|---:|---:|---:|---:|']
original_sources={r['source']:r for r in original['final_validation']['sources']}
parent_sources={r['source']:r for r in parents['moe']['final_validation']['sources']}
for r in mf['sources']:
    key=r['source'];d=source_dense[key]['cross_entropy_nats'];m=r['cross_entropy_nats']
    lines.append(f"| {key} | {original_sources[key]['cross_entropy_nats']:.4f} | {parent_sources[key]['cross_entropy_nats']:.4f} | {m:.4f} | {d:.4f} | {d-m:+.4f} |")
lines += ['', '## GPU cost and integrity', '',
'| Extension | Optimizer updates | Recorded run time | Peak allocated GPU memory |',
'|---|---:|---:|---:|']
for kind in ('moe','dense'):
    r=results[kind];lines.append(f"| {kind} | {r['updates']} | {r['elapsed_seconds']/60:.2f} min | {r['peak_allocated_bytes']/1024**3:.2f} GiB |")
lines += ['',
'Times include training, periodic validation and checkpoint saving, but exclude setup and the separate routing diagnostics. The MoE resumed a verified checkpoint after an interruption; downtime and discarded uncommitted work are excluded from its recorded time. GPU memory is PyTorch allocation rather than total device/display memory. All runs used the existing RTX 3070 Laptop GPU environment.', '',
f"Audit status: **{audit['status']}**. The audit checks the unchanged first experiment and original source files, exact 40M-target accounting, code and checkpoint hashes, finite weights, checkpoint loading and forward execution, evaluation counts, loss reduction, and the full-set routing controls.", '',
'## Outputs and interpretation', '',
'- `moe/final.pt`: extended trained MoE; `moe/best.pt`: best fixed-probe checkpoint.',
'- `dense/final.pt`: token-matched extended dense control.',
'- `diagnostics/summary.json`: every routing-control score and definition.',
'- `GENERATIONS.md`: all fixed-prompt greedy continuations, without editing or selection.',
'- `data_plan.json`, `design.json`, and `verification.json`: provenance, fixed design, and checks.',
'- `README.md`: exact training and diagnostic workflow, commands and limitations.', '',
'These are exploratory findings from one training seed per architecture, using validation that has already been inspected during development. No untouched test-set claim or statistical architectural superiority is made. Falling validation loss does not ensure fluent, accurate, or instruction-following text. The continuation examples should be read alongside the numerical scores.', '']
(ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
samples=read(ROOT/'diagnostics/samples.json');text=['# Fixed-prompt greedy continuations','',
'Every prompt and output is retained. These are base-model continuations, not chat responses. No best-of sampling or manual cleanup was used.','']
for sample in samples:
    text += [f"## {sample['model']} — {sample['prompt'].strip()}",'','````text',sample['prompt']+sample['continuation'],'````','']
(ROOT/'GENERATIONS.md').write_text('\n'.join(text),encoding='utf-8')
summary=dict(moe_loss=ml,dense_loss=dl,moe_perplexity=mf['perplexity'],dense_perplexity=df['perplexity'],
    moe_advantage=dl-ml,random_mean_loss=random_mean,random_routing_penalty=random_delta,
    uniform_weight_penalty=uniform_delta,sources_with_lower_moe_loss=wins,dead_experts=dead,
    first_layer_top_two_assignment_share=first_top2)
(ROOT/'findings.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
