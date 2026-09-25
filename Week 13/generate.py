"""Generate from a final precision-hardened checkpoint; no model downloads."""
import argparse
import json
from pathlib import Path
import sys
import torch
from model_precise import Decoder

ROOT=Path(__file__).resolve().parent
DATA_ROOT=ROOT/'data' if (ROOT/'data').is_dir() else ROOT.parent
sys.path.insert(0,str(DATA_ROOT/'Corpus_20M_v1'))
from build_corpus import FastTokenizer


@torch.inference_mode()
def generate(checkpoint,prompt,limit=64,seed=20260919,temperature=0.8):
    torch.set_num_threads(4);torch.manual_seed(seed)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=state['config'];device='cuda' if torch.cuda.is_available() else 'cpu'
    m=Decoder(cfg['variant'],cfg['model']).to(device);m.load_state_dict(state['model']);m.eval()
    tok=FastTokenizer();ids=tok.encode(prompt).tolist()[:-1]
    start=len(ids)
    if start>=512:raise ValueError('Prompt is too long')
    forbidden=[v for k,v in tok.specials.items() if k!='<eos>']
    for _ in range(min(limit,512-start)):
        x=torch.tensor([ids],device=device);seg=torch.zeros_like(x)
        pos=torch.arange(x.shape[1],device=device)[None]
        with torch.autocast(device,enabled=device=='cuda',dtype=torch.bfloat16):out=m(x,seg,pos)[0,-1].float()
        out[forbidden]=-torch.inf
        if temperature==0:next_id=int(out.argmax())
        else:
            values,index=torch.topk(out/temperature,40)
            next_id=int(index[torch.multinomial(torch.softmax(values,-1),1)])
        if next_id==tok.specials['<eos>']:break
        ids.append(next_id)
    return dict(prompt=prompt,continuation=b''.join(tok.pieces[i] for i in ids[start:]).decode('utf-8',errors='replace'),
                generated_token_ids=ids[start:],temperature=temperature,seed=seed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--prompt',default='Once upon a time,')
    p.add_argument('--tokens',type=int,default=64);p.add_argument('--temperature',type=float,default=0.8)
    args=p.parse_args();r=generate(args.checkpoint,args.prompt,args.tokens,temperature=args.temperature)
    print(r['prompt']+r['continuation'])
