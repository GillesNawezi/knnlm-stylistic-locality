import numpy as np
import torch
import torch.nn as nn
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pathlib
from fairseq import utils
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

    return curr_prob


class LeakyReLUNet(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.LeakyReLU(),
            nn.Linear(out_feat, out_feat),
        )

    def forward(self, features):
        return self.model(features)


class WeightedDist(torch.nn.Module):
    def __init__(self,
                 hidden_units=32,
                 nlayers=3,
                 dropout=0.,
                 activation='relu',
                 context_dim=1024,
                 num_outputs=7, ):
        super().__init__()

        models = [nn.Linear(context_dim, hidden_units), nn.Dropout(p=dropout)]
        if activation == 'relu':
            models.append(nn.ReLU())
        elif activation == 'linear':
            pass
        else:
            raise ValueError(f'activation {activation} not supported')

        for _ in range(nlayers - 1):
            models.extend([nn.Linear(hidden_units, hidden_units), nn.Dropout(p=dropout)])
            if activation == 'relu':
                models.append(nn.ReLU())
            elif activation == 'linear':
                pass
            else:
                raise ValueError(f'activation {activation} not supported')

        models.append(nn.Linear(hidden_units, num_outputs))

        self.model = nn.Sequential(*models)

    def forward(self, context_vec, dist, pkg_l, proj_l, idx_mask):
        """
        """
        context_vec = context_vec.cuda()
        dist = dist.cuda()
        pkg_l = pkg_l.cuda()
        proj_l = proj_l.cuda()
        idx_mask = idx_mask.cuda()

        locality_indicator = proj_l + pkg_l

        print(locality_indicator.max)

        locality_feat = torch.nn.functional.one_hot(locality_indicator.long(), num_classes=3).permute(2, 0, 1)
        print("test")
        print(f"max : {locality_feat.max()}")
        print(f"context_vec: {context_vec.shape}")

        params = self.model(context_vec)

        probs = utils.log_softmax(locality_feat[0] * (params[:, 0][:, None] * dist) +
                                  locality_feat[1] * (params[:, 1][:, None] * dist + params[:, 2][:, None]) +
                                  locality_feat[2] * (params[:, 3][:, None] * dist + params[:, 4][:, None]), dim=-1)
        # probs = utils.log_softmax(self.w0 * dist + self.w2 * pkg_l, dim=-1)
        # inp = torch.stack([dist, proj_l, pkg_l], dim=2)
        # new_dist = self.linear(inp).squeeze()
        # probs = utils.log_softmax(dist, dim=-1)

        return torch.logsumexp(probs + idx_mask, dim=-1), params



num_retrieved = 1024
dataset = "style_category_wiki_fine_tune" #"style_category_dataset"
global_path = str(pathlib.Path(__file__).parent.resolve()) + "/"
n=50000
epochs = 1000

context_vecs = np.load(global_path + f'saved_tensors/{dataset}/valid_context_cache.npy')
dists = np.load(f'saved_tensors/{dataset}/valid_proj_dist_cache.npy').reshape(-1, num_retrieved)
pkg_locality = np.load(f'saved_tensors/{dataset}/valid_pkg_locality_cache.npy').reshape(-1, num_retrieved)
proj_locality = np.load(f'saved_tensors/{dataset}/valid_proj_locality_cache.npy').reshape(-1, num_retrieved)
index_masks = np.load(f'saved_tensors/{dataset}/valid_proj_index_mask_cache.npy').reshape(-1, num_retrieved)
lm_probs = np.load(f'saved_tensors/{dataset}/valid_lm_prob_cache.npy')

context_vecs = context_vecs[:n,:]
dists = dists[:n,:]
pkg_locality = pkg_locality[:n,:]
proj_locality = proj_locality[:n,:]
index_masks = index_masks[:n,:]
lm_probs = lm_probs[:n]

context_vecs = torch.from_numpy(context_vecs).float()
dists = torch.from_numpy(dists).float()
pkg_locality = torch.from_numpy(pkg_locality)
proj_locality = torch.from_numpy(proj_locality)
index_masks = torch.from_numpy(index_masks)


test_context_vecs = np.load(f'saved_tensors/{dataset}/test_context_cache.npy')
test_dists = np.load(f'saved_tensors/{dataset}/test_proj_dist_cache.npy').reshape(-1, num_retrieved)
test_pkg_locality = np.load(f'saved_tensors/{dataset}/test_pkg_locality_cache.npy').reshape(-1, num_retrieved)
test_proj_locality = np.load(f'saved_tensors/{dataset}/test_proj_locality_cache.npy').reshape(-1, num_retrieved)
test_index_masks = np.load(f'saved_tensors/{dataset}/test_proj_index_mask_cache.npy').reshape(-1, num_retrieved)
test_lm_probs = np.load(f'saved_tensors/{dataset}/test_lm_prob_cache.npy')

test_context_vecs = test_context_vecs[:n,:]
test_dists = test_dists[:n,:]
test_pkg_locality = test_pkg_locality[:n,:]
test_proj_locality = test_proj_locality[:n,:]
test_index_masks = test_index_masks[:n,:]
test_lm_probs = test_lm_probs[:n]

test_context_vecs = torch.from_numpy(test_context_vecs).float()
test_dists = torch.from_numpy(test_dists).float()
test_pkg_locality = torch.from_numpy(test_pkg_locality)
test_proj_locality = torch.from_numpy(test_proj_locality)
test_index_masks = torch.from_numpy(test_index_masks)
test_lm_probs = torch.from_numpy(test_lm_probs).float()

valid_dataset = TensorDataset(context_vecs, dists, pkg_locality, proj_locality, index_masks)

test_dataset = TensorDataset(test_context_vecs, test_dists, test_pkg_locality, test_proj_locality, test_index_masks, test_lm_probs)
bsz = 102400
valid_dataloader = DataLoader(valid_dataset, batch_size=bsz, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=bsz, num_workers=4)

"""
model = WeightedDist(nlayers=2, hidden_units=64, dropout=0.3).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.2, min_lr=1e-5)
best_val_loss = 1e8
prev_lr = optimizer.param_groups[0]['lr']
"""

#model = WeightedDist(nlayers=2, hidden_units=64, dropout=0.3, num_outputs=5, context_dim=512).cuda()
model = WeightedDist(nlayers=2, hidden_units=64, dropout=0.3, num_outputs=5, context_dim=1024).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5, min_lr=1e-5)
best_val_loss = 1e8
prev_lr = optimizer.param_groups[0]['lr']
try:
  model.load_state_dict(torch.load(f'checkpoints/{dataset}/adaptive_model_weights.pt'))
except:
  print("No model found")

for i in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    model.train()
    for sample in tqdm(valid_dataloader):

        optimizer.zero_grad()
        num_batches += 1
        outputs, ps = model(sample[0],
                            sample[1],
                            sample[2],
                            sample[3],
                            sample[4])
        loss = torch.mean(-outputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * bsz

    print('Epoch:', i, 'Train Loss:', epoch_loss / len(valid_dataset))
    model.eval()
    val_loss = 0.
    for sample in tqdm(test_dataloader):
        val_outputs, _ = model(sample[0],
                            sample[1],
                            sample[2],
                            sample[3],
                            sample[4])
        final_prob = combine_knn_and_vocab_probs(val_outputs, sample[5].cuda(), 0.25)
        val_loss += - final_prob.sum().item()
    val_loss /= len(test_dataset)
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print('cur val', np.exp(val_loss))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print('save model, val best', np.exp(best_val_loss))
        torch.save(model.state_dict(), f'checkpoints/{dataset}/adaptive_model_weights.pt')
    elif current_lr != prev_lr:
        print('new lr, load prev best model: ', current_lr, np.exp(best_val_loss))
        prev_lr = current_lr
        model.load_state_dict(torch.load( f'checkpoints/{dataset}/adaptive_model_weights.pt'))
    print()
