"""Build static scientific figures and a readable report from measured JSON."""
from pathlib import Path
import json, os, subprocess, shutil
from reportlab.graphics.shapes import Drawing, String, Rect, Line
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor, white
ROOT=Path(__file__).resolve().parents[1]/'deep_dive'
read=lambda n:json.loads((ROOT/n).read_text())
d=read('diagnostics.json');b=read('balancing.json');linear=json.loads((ROOT.parent/'results/linear_results.json').read_text())
def label(c,x,y,s,size=11,bold=False,color='#183444'):
    c.add(String(x,y,s,fontSize=size,fontName='Helvetica-Bold' if bold else 'Helvetica',fillColor=HexColor(color)))
def row(policy,**filters):
    return next(r for r in d['experiments'] if r['policy']==policy and all(r.get(k,2 if k=='k' else None)==v for k,v in filters.items()))
def save_chart(c,name):
    renderSVG.drawToFile(c,str(ROOT/(name+'.svg')))
    node=os.environ.get('MOE_NODE') or shutil.which('node')
    if not node:raise RuntimeError('Install Node or set MOE_NODE to the node executable.')
    sharp=os.environ.get('MOE_SHARP','sharp')
    config=ROOT/'fonts.conf';config.write_text('<?xml version="1.0"?><fontconfig><dir>C:/Windows/Fonts</dir><alias><family>Helvetica</family><prefer><family>Arial</family></prefer></alias></fontconfig>')
    subprocess.run([node,'-e','require(process.argv[1])(process.argv[2],{density:160}).png().toFile(process.argv[3]);',sharp,str(ROOT/(name+'.svg')),str(ROOT/(name+'.png'))],check=True,env=dict(os.environ,FONTCONFIG_FILE=str(config)))

c=Drawing(950,530);c.add(Rect(0,0,950,530,fillColor=white,strokeColor=None))
label(c,25,500,'Small linear MoE: what is actually helping?',21,True)
label(c,25,477,'Same trained weights; all 79,116 selected held-out byte targets. Lower loss is better.',11)
label(c,25,442,'Routing intervention',13,True);label(c,400,442,'Held-out loss',13,True)
random=sum(r['loss'] for r in d['experiments'] if r['policy']=='random')/3
vals=[('Learned top-2',d['baseline']['loss']),('Top-2 IDs, equal weights',row('uniform')['loss']),
      ('All four, equal fixed weights',row('fixed')['loss']),('Random pairs (3-seed mean)',random)]
for i,(name,value) in enumerate(vals):
    y=410-i*38;label(c,25,y,name,12)
    color='#087f8c' if i==0 else '#a25b2a'
    c.add(Rect(400,y-3,(value-2.2)*1400,19,fillColor=HexColor(color),strokeColor=None))
    label(c,410+(value-2.2)*1400,y,f'{value:.4f}',12,True)
label(c,400,254,'Bars start at loss 2.20 to show the differences.',10)
label(c,25,220,'Turn off one expert',13,True)
label(c,230,220,'Usage share',12,True);label(c,400,220,'Discard its output',12,True);label(c,655,220,'Reroute to survivors',12,True)
for e in range(4):
    y=190-e*32;label(c,25,y,f'E{e}',12,True)
    label(c,230,y,f"{100*d['baseline']['assignment_share'][e]:.2f}%",12)
    label(c,400,y,f"{row('zero',expert=e)['loss']:.4f}",12,color='#a25b2a')
    label(c,655,y,f"{row('reroute',expert=e)['loss']:.4f}",12,color='#087f8c')
label(c,25,48,'Rerouting compensates for some damage. It does not establish safe permanent pruning.',11)
label(c,25,27,'Fixed-weight averaging is an affine model; adaptive routing allows a nonlinear input-to-logit map.',11)
save_chart(c,'routing_and_shutdown')

aliases={'no_balance':'No balancing','aux_micro_001':'Aux .001, micro','aux_micro_01':'Aux .01, micro',
         'aux_micro_001_z001':'Aux .001 + z .001','aux_whole_001':'Aux .001, whole','bias_whole':'Bias .001, whole'}
c=Drawing(950,345);c.add(Rect(0,0,950,345,fillColor=white,strokeColor=None))
label(c,25,313,'Balancing the small model: measure prediction and utilization',18,True)
label(c,25,291,'200 additional updates / 204,800 byte exposures per arm; same batches and starting checkpoint.',11)
label(c,25,270,'Full selected byte holdout; one training seed. These small gaps do not establish a winning method.',11)
label(c,25,236,'Method',12,True);label(c,310,236,'Loss',12,True);label(c,440,236,'Loss change',12,True);label(c,590,236,'Max load violation',12,True);label(c,835,236,'Dead / 4',12,True)
for i,r in enumerate(b):
    y=208-27*i;v=r['final']['max_load_violation']
    label(c,25,y,aliases[r['config']['name']],11);label(c,310,y,f"{r['final']['loss']:.6f}",11)
    label(c,440,y,f"{r['delta']:+.6f}",11)
    c.add(Rect(590,y-3,850*v,14,fillColor=HexColor('#087f8c'),strokeColor=None));label(c,600+850*v,y,f'{v:.4f}',10)
    label(c,860,y,str(r['final']['dead_experts']),11)
label(c,25,26,'Max load violation = busiest expert count / uniform count - 1. Lower means more even utilization.',11)
save_chart(c,'balancing')

print('Figures regenerated')
