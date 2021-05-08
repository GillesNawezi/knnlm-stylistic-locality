import numpy as np
import torch
from torch import nn

from fairseq import utils


class WeightedDist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.w0 = torch.nn.Parameter(torch.tensor(1.))
        # self.b0 = torch.nn.Parameter(torch.tensor(0.))
        # self.linear = nn.Linear(3, 1, bias=False)
        self.w1 = nn.Parameter(torch.tensor(15.))
        self.b1 = nn.Parameter(torch.tensor(5.))
        self.w2 = nn.Parameter(torch.tensor(15.))
        self.b2 = nn.Parameter(torch.tensor(4.))

    def forward(self, dist, pkg_l, proj_l, idx_mask):
        """
        """
        dist = dist.cuda()
        pkg_l = pkg_l.cuda()
        proj_l = proj_l.cuda()
        idx_mask = idx_mask.cuda()
        local1 = torch.zeros_like(proj_l)
        local1[(proj_l == 1) & (pkg_l == 0)] = 1

        # make 3 features, local=0, 1, 2 and mutually exclusive
        locality_feat = [1 - (local1 | pkg_l), local1, pkg_l]

        probs = utils.log_softmax(locality_feat[0] * dist +
                                  locality_feat[1] * (self.w1 * dist + self.b1) +
                                  locality_feat[2] * (self.w2 * dist + self.b2), dim=-1)
        # probs = utils.log_softmax(dist + self.w1 * proj_l + self.w2 * pkg_l, dim=-1)
        # inp = torch.stack([dist, proj_l, pkg_l], dim=2)
        # new_dist = self.linear(inp).squeeze()
        # probs = utils.log_softmax(probs, dim=-1)

        return torch.logsumexp(probs + idx_mask, dim=-1)


num_retrieved = 1024
bsz = 200000

dists = np.load('/node09_data/frank/test_proj_dist_cache.npy').reshape(-1, num_retrieved)
pkg_locality = np.load('/node09_data/frank/test_pkg_locality_cache.npy').reshape(-1, num_retrieved)
proj_locality = np.load('/node09_data/frank/test_proj_locality_cache.npy').reshape(-1, num_retrieved)
index_masks = np.load('/node09_data/frank/test_proj_index_mask_cache.npy').reshape(-1, num_retrieved)


dists = torch.from_numpy(dists).float()
pkg_locality = torch.from_numpy(pkg_locality)
proj_locality = torch.from_numpy(proj_locality)
index_masks = torch.from_numpy(index_masks)

model = WeightedDist().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(5000):
    epoch_loss = 0.0
    num_batches = 0
    for start_idx in range(0, dists.shape[0], bsz):
        num_batches += 1
        outputs = model(dists[start_idx:start_idx+bsz, :],
                        pkg_locality[start_idx:start_idx+bsz, :],
                        proj_locality[start_idx:start_idx+bsz, :],
                        index_masks[start_idx:start_idx+bsz, :])
        loss = torch.mean(-outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print('Epoch:', i, 'Loss:', epoch_loss)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
