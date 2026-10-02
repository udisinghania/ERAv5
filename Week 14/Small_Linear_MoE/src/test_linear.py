"""Check affine semantics in FP64 to separate algebra from FP32 reduction error."""
import torch
from common import CHECKPOINTS, write
from linear_assignment import LinearPredictor, LinearMoE
torch.set_num_threads(2);torch.manual_seed(918)
d=LinearPredictor().double()
d.load_state_dict(torch.load(CHECKPOINTS/'linear_pretrained.pt',map_location='cpu',weights_only=True)['model'])
x=torch.randn(16,1028,dtype=torch.float64);y=torch.randn_like(x)
torch.testing.assert_close(d(.3*x+.7*y),.3*d(x)+.7*d(y),atol=1e-11,rtol=1e-10)
m=LinearMoE(d).double();torch.testing.assert_close(d(x),m(x),atol=1e-11,rtol=1e-10)
m.load_state_dict(torch.load(CHECKPOINTS/'linear_moe.pt',map_location='cpu',weights_only=True)['model'])
assert torch.isfinite(m(x)).all()
write('linear_tests.json',dict(status='PASS',precision='float64 algebra checks; original training and real-feature conversion check used float32',checks=['Dense affine identity','Copied MoE output equivalence','Final MoE state loads and forwards']))
print('Literal linear semantic checks PASS')
