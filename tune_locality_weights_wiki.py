import numpy as np
import torch
from torch import nn

from fairseq import utils


class WeightedDist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.))
        # self.b0 = torch.nn.Parameter(torch.tensor(0.))
        # self.linear = nn.Linear(3, 1, bias=False)
        self.w1 = nn.Parameter(torch.tensor(1.))
        self.b1 = nn.Parameter(torch.tensor(0.))
        self.w2 = nn.Parameter(torch.tensor(1.))
        self.b2 = nn.Parameter(torch.tensor(0.))
        self.w3 = nn.Parameter(torch.tensor(1.))
        self.b3 = nn.Parameter(torch.tensor(0.))


    def forward(self, dist, pkg_l, proj_l, idx_mask):
        """
        """
        dist = dist.cuda()
        pkg_l = pkg_l.cuda()
        proj_l = proj_l.cuda()
        idx_mask = idx_mask.cuda()

        locality_indicator = proj_l + 2 * pkg_l

        locality_feat = torch.nn.functional.one_hot(locality_indicator.long(), num_classes=4).permute(2, 0, 1)

        probs = utils.log_softmax(locality_feat[0] * (self.w0 * dist) +
                                  locality_feat[1] * (self.w1 * dist + self.b1) +
                                  locality_feat[2] * (self.w2 * dist + self.b2) +
                                  locality_feat[3] * (self.w3 * dist + self.b3), dim=-1)
        # probs = utils.log_softmax(self.w0 * dist + self.w2 * pkg_l, dim=-1)
        # inp = torch.stack([dist, proj_l, pkg_l], dim=2)
        # new_dist = self.linear(inp).squeeze()
        # probs = utils.log_softmax(probs, dim=-1)

        return torch.logsumexp(probs + idx_mask, dim=-1)


num_retrieved = 1024
bsz = 810

dists = np.load('saved_tensors/wikitext-103/test_proj_dist_cache.npy').reshape(-1, num_retrieved)
pkg_locality = np.load('saved_tensors/wikitext-103/test_pkg_locality_cache.npy').reshape(-1, num_retrieved)
proj_locality = np.load('saved_tensors/wikitext-103/test_proj_locality_cache.npy').reshape(-1, num_retrieved)
index_masks = np.load('saved_tensors/wikitext-103/test_proj_index_mask_cache.npy').reshape(-1, num_retrieved)


dists = torch.from_numpy(dists).float()
pkg_locality = torch.from_numpy(pkg_locality)
proj_locality = torch.from_numpy(proj_locality)
index_masks = torch.from_numpy(index_masks)

model = WeightedDist().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(2000):
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
