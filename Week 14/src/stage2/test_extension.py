"""Ensure the 10M/40M boundary consumes every second-pass target exactly once."""
import json
import numpy as np
import torch
import extend

root=extend.ROOT;parent=extend.PARENT
plan=extend.read(root/'data_plan.json');oldplan=extend.read(parent/'data_plan.json')
dataset=extend.PackedDataset(extend.base.CORPUS/'packed_50m_ctx512')
old=np.load(parent/'training_sequence_indices.npy');new=np.load(root/'training_sequence_indices.npy')
overlap=np.intersect1d(old,new)
assert overlap.tolist()==[int(old[-1])]
assert old[-1]==new[0]
data=extend.train_batch(dataset,new[:2],first=True)
after=data[1][0].cpu().numpy()
before=np.array(dataset.arrays['loss_mask'][old[-1]],copy=True)
positions=np.flatnonzero(before)
before[positions[-oldplan['final_sequence_masked_targets']:]]=0
original=dataset.arrays['loss_mask'][old[-1]]
assert not np.any(before & after)
assert np.array_equal(before+after,original)
assert int(after.sum())==74
counts=dataset.arrays['loss_mask'][new,1:].sum(axis=1,dtype=np.int64)
assert int(counts.sum())-plan['first_sequence_skip_targets']==40_000_000
assert len(np.union1d(old,new))==len(dataset)
assert len(np.unique(new))==len(new)
assert all(bool(torch.isfinite(x).all()) for x in data)
result=dict(status='PASS',checks=['Only boundary sequence overlaps','Boundary masks are disjoint and exhaustive',
    '74 previously unused boundary targets retained','Exactly 40M additional targets','Combined plan covers all packed sequences',
    'No repeated sequence within extension','Real GPU batch loads correctly'])
extend.write(root/'extension_test_results.json',result);print(json.dumps(result))
dataset.close()
