from pathlib import Path
import json,math,statistics,subprocess,os,shutil
from reportlab.graphics.shapes import Drawing,String,Rect,Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor,white
P=Path(__file__).resolve().parents[1]
FIGURES=Path(os.environ.get('SLM_FIGURES',P/'runs/figures'))
FIGURES.mkdir(parents=True,exist_ok=True)
read=lambda n:json.loads((P/n).read_text())
r=read('results/linear_results.json');study=read('results/extended_study/results.json')['stages']
d=read('deep_dive/diagnostics.json');b=read('deep_dive/balancing.json');mix=read('data/source_mix.json')['counts_by_source_group']
stage=lambda seed,arm:next(x for x in study if x['seed']==seed and x['arm']==arm)
baseline=r['dense_pretraining'][-1]['loss'];moe=r['moe_continuation'][-1]['loss'];dense=r['dense_continuation'][-1]['loss']
labels={'pretrained':'Linear pretraining','dense_control':'Dense continuation','moe':'MoE continuation'}
def label(c,x,y,s,size=11,bold=False):c.add(String(x,y,s,fontSize=size,fontName='Helvetica-Bold' if bold else 'Helvetica',fillColor=HexColor('#173745')))
def chart_save(c,name):
    renderSVG.drawToFile(c,str(FIGURES/(name+'.svg')))
    config=FIGURES/'fonts.conf';config.write_text('<?xml version="1.0"?><fontconfig><dir>C:/Windows/Fonts</dir><alias><family>Helvetica</family><prefer><family>Arial</family></prefer></alias></fontconfig>')
    subprocess.run([os.environ.get('MOE_NODE') or shutil.which('node'),'-e',
        'require(process.argv[1])(process.argv[2],{density:160}).png().toFile(process.argv[3]);',
        os.environ.get('MOE_SHARP','sharp'),str(FIGURES/(name+'.svg')),str(FIGURES/(name+'.png'))],
        check=True,env=dict(os.environ,FONTCONFIG_FILE=str(config)))
c=Drawing(950,350);c.add(Rect(0,0,950,350,fillColor=white,strokeColor=None))
label(c,24,318,'Two seeds: the MoE advantage persists in both matched comparisons',17,True)
label(c,24,296,'Same frozen data; different initialization and batch-sampling seeds. Held-out byte loss.',11)
for panel,seed in enumerate([31415,27182]):
    x=65+panel*460;plot=LinePlot();plot.x=x;plot.y=62;plot.width=345;plot.height=180
    plot.data=[[(v['step'],v['loss']) for v in stage(seed,arm)['history']] for arm in ['moe','dense_control']]
    plot.joinedLines=1;plot.xValueAxis.valueMin=0;plot.xValueAxis.valueMax=600;plot.yValueAxis.valueMin=2.22;plot.yValueAxis.valueMax=2.40
    for i,col in enumerate(['#087f8c','#b26734']):plot.lines[i].strokeColor=HexColor(col);plot.lines[i].strokeWidth=2
    c.add(plot);label(c,x,260,f'Seed {seed}',13,True)
    label(c,x+100,28,'Continuation optimizer updates',10)
    label(c,x,12,f"MoE (teal): {stage(seed,'moe')['history'][-1]['loss']:.6f} | Dense (brown): {stage(seed,'dense_control')['history'][-1]['loss']:.6f}",10)
chart_save(c,'two_seed_learning')
c=Drawing(950,365);c.add(Rect(0,0,950,365,fillColor=white,strokeColor=None))
label(c,24,332,'The loss improvement has a compute cost',19,True)
label(c,24,309,'RTX 3070 Laptop, FP32, batch 1,024. Separate process per stage; both measured seeds shown.',11)
label(c,24,273,'Stage / seed',12,True);label(c,295,273,'Training targets / second',12,True);label(c,670,273,'Peak allocated MiB',12,True)
for i,(arm,seed) in enumerate([(a,s) for a in ['pretrained','dense_control','moe'] for s in [31415,27182]]):
    x=stage(seed,arm);y=244-i*31;label(c,24,y,f'{labels[arm]} / {seed}',10)
    value=x['training_targets_per_second'];w=value/100000*290
    c.add(Rect(295,y-3,w,15,fillColor=HexColor('#087f8c' if arm=='moe' else '#7695a3'),strokeColor=None));label(c,303+w,y,f'{value:,.0f}',10)
    label(c,715,y,f"{x['peak_training_allocated_bytes']/2**20:.2f}",11)
label(c,24,32,'Timing includes input preparation and synchronized training steps; excludes validation, saves and startup.',10)
label(c,24,14,'These are two observations on one laptop, not a hardware-independent speed guarantee.',10)
chart_save(c,'training_cost')
