"""Check that diagnostic controls preserve copied experts and token causality."""
import json
import torch
from dense_model import Decoder
from moe_model import MoEDecoder
from diagnostics import set_mode
from extend import ROOT,write

torch.set_num_threads(1);torch.manual_seed(8)
dense=Decoder(dict(vocab_size=19,context_length=8,hidden_size=12,layers=2,heads=3,intermediate_size=24,norm_eps=1e-5))
model=MoEDecoder(dense)
ids=torch.randint(0,19,(2,8));segments=torch.zeros_like(ids);positions=torch.arange(8)[None].expand(2,-1)
with torch.no_grad():
    expected=dense(ids,segments,positions)
    for mode in ('learned','uniform_selected','random'):
        set_mode(model,mode,17)
        actual=model(ids,segments,positions)
        torch.testing.assert_close(expected,actual,atol=1e-6,rtol=1e-5)
        assert all(int(b.experts.usage.sum())==32 for b in model.blocks)
    # Diversify the experts and confirm that random routing does not use future inputs.
    model.blocks[0].experts.down[0].weight.add_(0.1*torch.randn_like(model.blocks[0].experts.down[0].weight))
    set_mode(model,'random',29);a=model(ids,segments,positions)
    changed=ids.clone();changed[:,7]=(changed[:,7]+1)%19
    set_mode(model,'random',29);b=model(changed,segments,positions)
    torch.testing.assert_close(a[:,:7],b[:,:7])
    set_mode(model,'learned');original=model(ids,segments,positions)
    set_mode(model,'uniform_selected');model(ids,segments,positions)
    set_mode(model,'learned');restored=model(ids,segments,positions)
    torch.testing.assert_close(original,restored)
result=dict(status='PASS',checks=['All policies preserve the identical-copy function','Every policy selects two experts per valid token',
    'Random routing remains causal with fixed RNG','Restoring learned policy restores outputs'])
write(ROOT/'diagnostic_test_results.json',result);print(json.dumps(result))
