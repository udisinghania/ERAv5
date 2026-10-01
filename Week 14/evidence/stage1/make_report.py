"""Generate measured Markdown and charts with ReportLab's plotting library."""
import json
import math
from pathlib import Path
import subprocess
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor, white

ROOT=Path(__file__).resolve().parent
NODE=Path(r'C:\Users\udisi\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
SHARP=Path(r'C:\Users\udisi\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules\sharp')
COLORS=[HexColor('#177f92'),HexColor('#c36427')]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def events(kind):return [json.loads(s) for s in (ROOT/kind/'events.jsonl').read_text().splitlines()]
def save(drawing,name):
    svg=ROOT/(name+'.svg');png=ROOT/(name+'.png')
    renderSVG.drawToFile(drawing,str(svg))
    script="const sharp=require(process.argv[1]);sharp(process.argv[2],{density:160}).png().toFile(process.argv[3]).catch(e=>{console.error(e);process.exit(1)});"
    subprocess.run([str(NODE),'-e',script,str(SHARP),str(svg),str(png)],check=True)


def panel(d,x,y,title,series,xlabel,yrange=None):
    d.add(String(x,y+228,title,fontName='Helvetica-Bold',fontSize=13,fillColor=HexColor('#182d3c')))
    chart=LinePlot();chart.x=x+48;chart.y=y+38;chart.width=360;chart.height=164
    chart.data=series;chart.joinedLines=1
    for i in range(len(series)):
        chart.lines[i].strokeColor=COLORS[i%len(COLORS)];chart.lines[i].strokeWidth=1.8
    chart.xValueAxis.valueMin=0
    chart.xValueAxis.labels.fontSize=8;chart.yValueAxis.labels.fontSize=8
    chart.xValueAxis.valueMax=max(1,max(max(p[0] for p in s) for s in series))
    values=[p[1] for s in series for p in s]
    lo,hi=yrange or (min(values)-0.03,max(values)+0.03)
    chart.yValueAxis.valueMin=lo;chart.yValueAxis.valueMax=hi
    chart.yValueAxis.visibleGrid=1;chart.yValueAxis.gridStrokeColor=HexColor('#e1e8ed')
    chart.yValueAxis.labelTextFormat='%.2f'
    d.add(chart);d.add(String(x+220,y+8,xlabel,textAnchor='middle',fontSize=9))
    d.add(String(x+48,y+210,'Cross-entropy (nats)',fontSize=9,fillColor=HexColor('#556572')))


dense=read(ROOT/'dense/result.json');moe=read(ROOT/'moe/result.json')
previous=read(ROOT/'dense_pretraining_result.json');conversion=read(ROOT/'conversion_check.json')
audit=read(ROOT/'verification.json');history={k:events(k) for k in ('moe','dense')}
curves=Drawing(940,590)
curves.add(Rect(0,0,940,590,fillColor=white,strokeColor=None))
curves.add(String(30,556,'Dense to MoE: continued learning on reused training data',fontName='Helvetica-Bold',fontSize=20,fillColor=HexColor('#182d3c')))
curves.add(String(30,532,'Same 10M supervised tokens per arm | unchanged 5.05M-target holdout | one seed',fontSize=11,fillColor=HexColor('#556572')))
for i,(name,color) in enumerate(zip(('MoE, 4 experts / top-2','Dense continuation'),COLORS)):
    curves.add(Rect(590+i*168,510,10,6,fillColor=color,strokeColor=None));curves.add(String(605+i*168,508,name,fontSize=9))
train_series=[];probe_series=[]
for kind in ('moe','dense'):
    logs=history[kind]
    train_series.append([(r['consumed_tokens']/1e6,r['training_cross_entropy_nats']) for r in logs if r['event']=='train_progress'])
    by_step={r['updates']:r['consumed_tokens']/1e6 for r in logs if r['event']=='train_progress'}
    probe_series.append([(by_step.get(r['updates'],0),r['loss']) for r in logs if r['event']=='validation_probe'])
panel(curves,30,255,'Training task loss (auxiliary loss excluded)',train_series,'Additional supervised tokens (millions)')
panel(curves,490,255,'Fixed validation probe',probe_series,'Additional supervised tokens (millions)')
initial=moe['initial_validation']['cross_entropy_nats'];mf=moe['final_validation']['cross_entropy_nats'];df=dense['final_validation']['cross_entropy_nats']
panel(curves,30,0,'Full held-out evaluation',[
    [(0,initial),(10,mf)],[(0,dense['initial_validation']['cross_entropy_nats']),(10,df)]],
    'Additional supervised tokens (millions)',(min(mf,df)-0.03,initial+0.03))
curves.add(String(550,217,'Measured endpoints',fontName='Helvetica-Bold',fontSize=14))
for i,(label,value) in enumerate([
    ('Dense pretraining: random -> trained',f"{previous['initial_validation']['cross_entropy_nats']:.4f} -> {previous['final_validation']['cross_entropy_nats']:.4f}"),
    ('MoE immediately after conversion',f'{initial:.6f}'),
    ('MoE after 10M additional tokens',f'{mf:.6f}'),
    ('Dense after 10M additional tokens',f'{df:.6f}')]):
    curves.add(String(550,188-i*39,label,fontSize=10,fillColor=HexColor('#556572')))
    curves.add(String(550,172-i*39,value,fontName='Helvetica-Bold',fontSize=12))
curves.add(String(550,20,'Full holdout and probe have different token weights.',fontSize=9))
save(curves,'loss_curves')

routing=moe['final_validation']['routing'];d=Drawing(700,435)
d.add(Rect(0,0,700,435,fillColor=white,strokeColor=None))
d.add(String(25,402,'Final MoE expert usage on the full holdout',fontName='Helvetica-Bold',fontSize=18))
d.add(String(25,380,'Share of routed assignments within each layer; uniform use would be 25%.',fontSize=10))
for col in range(4):d.add(String(170+col*120,348,f'Expert {col}',fontName='Helvetica-Bold',fontSize=11,textAnchor='middle'))
for row,r in enumerate(routing):
    y=310-row*31
    d.add(String(25,y+9,f"Layer {row+1}",fontSize=11))
    for col,share in enumerate(r['shares']):
        t=min(1,share/0.5)
        from reportlab.lib.colors import Color
        color=Color(0.92-0.81*t,0.96-0.47*t,0.97-0.40*t)
        d.add(Rect(115+col*120,y,110,27,fillColor=color,strokeColor=white))
        d.add(String(170+col*120,y+8,f'{share*100:.1f}%',fontName='Helvetica-Bold',fontSize=11,textAnchor='middle',fillColor=white if t>0.6 else HexColor('#182d3c')))
dead=sum(r['dead_experts'] for r in routing)
d.add(String(25,20,f'Dead experts: {dead} / 36. Usage is measured over all non-padding validation inputs.',fontSize=10))
save(d,'expert_usage')

delta=mf-df
comparison=(f'The MoE finished {abs(delta):.6f} nats lower than the dense continuation.' if delta<0 else f'The dense continuation finished {abs(delta):.6f} nats lower than the MoE.')
lines=[
'# Session 14 assignment: measured results',
'',
f'**The converted MoE continued learning: full held-out loss decreased from {initial:.6f} to {mf:.6f} nats after 10 million additional supervised tokens.**',
'',
'The starting checkpoint was the previously trained 20.17M-parameter dense Transformer. Its trained FFNs were copied into four experts per layer; a new router selected two for each token. All original data and environment files were used read-only.',
'',
'| Stage | Training exposure | Full validation loss | Perplexity |',
'|---|---|---:|---:|',
f"| Dense, before original training | 0 | {previous['initial_validation']['cross_entropy_nats']:.6f} | {previous['initial_validation']['perplexity']:.2f} |",
f"| Dense, after original training | 50M tokens | {previous['final_validation']['cross_entropy_nats']:.6f} | {previous['final_validation']['perplexity']:.2f} |",
f"| MoE, immediately after conversion | Same pretrained weights | {initial:.6f} | {moe['initial_validation']['perplexity']:.2f} |",
f"| MoE, after continuation | 50M original + 10M reused | {mf:.6f} | {moe['final_validation']['perplexity']:.2f} |",
f"| Dense control, after continuation | 50M original + same 10M reused | {df:.6f} | {dense['final_validation']['perplexity']:.2f} |",
'',
'Historical dense pretraining measurements are inherited from the verified prior run. Conversion, continuation, and both new endpoint evaluations were measured in this experiment. All full evaluations use the same 5,049,456 held-out supervised targets.',
'',
'![Loss evidence](loss_curves.png)',
'',
'## What the comparison establishes',
'',
f'The MoE reduced held-out cross-entropy by {initial-mf:.6f} nats ({(initial-mf)/initial*100:.2f}%). {comparison} This is one run per arm; the small difference is descriptive, not a statistically established architectural advantage. The MoE also uses more active weights, so this is an equal-token comparison, not an equal-compute comparison.',
'',
'## Conversion and training details',
'',
'- Dense: 20,166,912 parameters. MoE: 54,685,440 total and 31,682,304 active per token, counting the full tied embedding table.',
'- Nine layers, four routed experts per layer, top-2, no shared expert, no token dropping. Each expert initially copies the complete trained 384 -> 1,664 -> 384 GELU FFN.',
f"- Before training, the largest full-FP32 dense/MoE logit difference was {conversion['fp32_max_logit_difference']:.9f}; the function-preservation check passed with tolerance 0.0002.",
'- Fresh AdamW state for both continuations, effective batch 32, peak LR 0.0001, 30 warmup updates, decay to 0.00001, and gradient clipping at 1.0.',
'- Auxiliary balancing coefficient 0.001, with load measured over each microbatch. Plotted task loss excludes this auxiliary term.',
'- Existing 50M-token corpus, frozen tokenizer, sequence length 512, original document-isolation masks, reset positions, and response-only target masks preserved.',
'- Exactly 10,000,000 additional supervised targets in each arm, using the identical seeded sequence order. These are reused training data. Held-out data never enter updates.',
'',
'## Routing and verification',
'',
f'All 36 experts updated and diverged from their identical initial copies. The full final validation pass found **{dead} dead experts**. Unequal utilization remains visible below; zero dead experts does not imply perfectly balanced routing.',
'',
'![Expert usage](expert_usage.png)',
'',
f"The post-training audit status is **{audit['status']}**. It checks input hashes, unchanged source files, exact token accounting, checkpoint hashes, finite weights, full held-out token counts, loss reduction, and expert updates. Four semantic tests cover conversion and gradients, causal/document isolation and masking, sparse routing, and optimizer restoration.",
'',
'## Per-source full validation',
'',
'| Source | Before conversion (dense re-evaluation) | Final MoE | Final dense |',
'|---|---:|---:|---:|']
base_by={s['source']:s for s in dense['initial_validation']['sources']};dense_by={s['source']:s for s in dense['final_validation']['sources']}
for s in moe['final_validation']['sources']:
    key=s['source'];lines.append(f"| {key} | {base_by[key]['cross_entropy_nats']:.4f} | {s['cross_entropy_nats']:.4f} | {dense_by[key]['cross_entropy_nats']:.4f} |")
lines += ['', '## Local GPU measurements', '',
'| Run | Updates | Recorded run time | Peak allocated GPU memory |',
'|---|---:|---:|---:|',
f"| MoE | {moe['updates']} | {moe['elapsed_seconds']/60:.2f} min | {moe['peak_allocated_bytes']/1024**3:.2f} GiB |",
f"| Dense control | {dense['updates']} | {dense['elapsed_seconds']/60:.2f} min | {dense['peak_allocated_bytes']/1024**3:.2f} GiB |",
'',
'Measured on the RTX 3070 Laptop GPU. Recorded run time includes periodic probes, checkpointing, and final validation, but excludes initial validation and setup. Memory is PyTorch peak allocation, not total system/display GPU use.',
'', '## Artifacts', '',
'- `moe/final.pt`: trained MoE checkpoint; `moe/best.pt`: checkpoint selected by the fixed validation probe.',
'- `dense/final.pt`: dense continuation control.',
'- `moe/events.jsonl` and `dense/events.jsonl`: task-loss and validation history.',
'- `data_plan.json`: upstream hashes and exact data-reuse plan.',
'- `verification.json`, `test_results.json`, `conversion_check.json`: recorded checks.',
'- `README.md`: architecture, local environment, reproduction and resume commands.',
'', '## Limits', '',
'The assignment is interpreted as dense-model upcycling. The original wording "Linear model" is ambiguous; this experiment uses a nonlinear dense Transformer. The model remains small and lightly trained. Lower loss does not establish conversational quality, factual accuracy, reasoning ability, or a general speed/quality advantage for MoE. The sparse Python implementation targets a single consumer GPU, not distributed training or optimized MoE kernels.', '']
(ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote REPORT.md, loss_curves.png and expert_usage.png')
