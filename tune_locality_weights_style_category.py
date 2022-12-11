import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pathlib

from fairseq import utils
from torch.utils.data import TensorDataset, DataLoader


def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

    return curr_prob


class OnlyWeight(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.))

    def forward(self, dist, idx_mask):
        """
        """
        dist = dist.cuda()
        idx_mask = idx_mask.cuda()

        probs = utils.log_softmax(self.w0 * dist, dim=-1)

        return torch.logsumexp(probs + idx_mask, dim=-1)


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
        
        #locality_indicator = proj_l + 2 * pkg_l
        locality_indicator = proj_l + pkg_l
        

        locality_feat = torch.nn.functional.one_hot(locality_indicator.long(), num_classes=3).permute(2, 0, 1)

        params = self.model(context_vec)

        probs = utils.log_softmax(locality_feat[0] * (params[:, 0][:, None] * dist) +
                                  locality_feat[1] * (params[:, 1][:, None] * dist + params[:, 2][:, None]) +
                                  locality_feat[2] * (params[:, 3][:, None] * dist + params[:, 4][:, None]), dim=-1)

        return torch.logsumexp(probs + idx_mask, dim=-1), params

n=30000
num_retrieved = 1024
dataset = "style_category_dataset"
global_path = str(pathlib.Path(__file__).parent.resolve()) + "/"

context_vecs = np.load(global_path + f'saved_tensors/{dataset}/valid_context_cache.npy')
dists = np.load(f'saved_tensors/{dataset}/valid_proj_dist_cache.npy').reshape(-1, num_retrieved)
pkg_locality = np.load(f'saved_tensors/{dataset}/valid_pkg_locality_cache.npy').reshape(-1, num_retrieved)
proj_locality = np.load(f'saved_tensors/{dataset}/valid_proj_locality_cache.npy').reshape(-1, num_retrieved)
index_masks = np.load(f'saved_tensors/{dataset}/valid_proj_index_mask_cache.npy').reshape(-1, num_retrieved)
lm_probs = np.load(f'saved_tensors/{dataset}/valid_lm_prob_cache.npy')
sample_ids = np.load(f'saved_tensors/{dataset}/valid_sample_id_cache.npy')


# Load Localities
folder = global_path + "examples/language_model/style_source_dataset/"
valid_style_file = folder + "valid.txt.style"

with open(valid_style_file, "r") as f:
    styles = f.readlines()


df = pd.Series(styles).to_frame()
df.index = df.index.set_names(["samp_id"])
df = df.reset_index()
df = df.rename(columns={0:"style"})
df_samp = pd.DataFrame({
                'samp_id': sample_ids.ravel(),
            })

df_samp = df_samp.merge(df, how="left", on="samp_id").dropna()

#n = df_samp["style"].value_counts().min()
n=30000
df_samp = df_samp.groupby("style").sample(n=n, random_state=1, replace=True)

indices = df_samp.index.tolist()

context_vecs = context_vecs[indices]
dists = dists[indices]
pkg_locality = pkg_locality[indices]
proj_locality = proj_locality[indices]
lm_probs = lm_probs[indices]
index_masks = index_masks[indices]

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
test_sample_ids = np.load(f'saved_tensors/{dataset}/test_sample_id_cache.npy')


test_style_file = folder + "test.txt.style"

with open(test_style_file, "r") as f:
    test_styles = f.readlines()

test_df = pd.Series(test_styles).to_frame()
test_df.index = test_df.index.set_names(["samp_id"])
test_df = test_df.reset_index()
test_df = test_df.rename(columns={0:"style"})
test_df_samp = pd.DataFrame({
                'samp_id': test_sample_ids.ravel(),
            })

test_df_samp = test_df_samp.merge(test_df, how="left", on="samp_id").dropna()
#n = test_df_samp["style"].value_counts().min()
test_df_samp = test_df_samp.groupby("style").sample(n=n, random_state=1, replace=True)

test_indices = test_df_samp.index.tolist()

test_context_vecs = test_context_vecs[test_indices]
test_dists = test_dists[test_indices]
test_pkg_locality = test_pkg_locality[test_indices]
test_proj_locality = test_proj_locality[test_indices]
test_lm_probs = test_lm_probs[test_indices]
test_index_masks = test_index_masks[test_indices]

test_context_vecs = torch.from_numpy(test_context_vecs).float()
test_dists = torch.from_numpy(test_dists).float()
test_pkg_locality = torch.from_numpy(test_pkg_locality)
test_proj_locality = torch.from_numpy(test_proj_locality)
test_index_masks = torch.from_numpy(test_index_masks)
test_lm_probs = torch.from_numpy(test_lm_probs).float().cuda()

valid_dataset = TensorDataset(context_vecs, dists, pkg_locality, proj_locality, index_masks)

test_dataset = TensorDataset(test_context_vecs, test_dists, test_pkg_locality, test_proj_locality, test_index_masks)
bsz = 10000
valid_dataloader = DataLoader(valid_dataset, batch_size=bsz, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=bsz)


model = WeightedDist(nlayers=2, hidden_units=64, dropout=0.3).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.2, min_lr=1e-5)
best_val_loss = 1e8
prev_lr = optimizer.param_groups[0]['lr']
epochs =1000

for i in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    model.train()
    torch.cuda.empty_cache()
    for sample in valid_dataloader:
        optimizer.zero_grad()
        num_batches += 1
        outputs, ps = model(sample[0],
                            sample[1],
                            sample[2],
                            sample[3],
                            sample[4])
        loss = torch.mean(-outputs)
        print(loss)
        x=y
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * bsz
    print('Epoch:', i, 'Train Loss:', epoch_loss / len(valid_dataset))
    model.eval()
    val_outputs = None
    torch.cuda.empty_cache()
    for sample in test_dataloader:
        optimizer.zero_grad()
        num_batches += 1
        test_outputs, ps = model(sample[0],
                            sample[1],
                            sample[2],
                            sample[3],
                            sample[4])
        if torch.is_tensor(val_outputs) == False:
            val_outputs = test_outputs
        else:
            val_outputs = torch.cat((val_outputs,test_outputs), dim=0)
    final_prob = combine_knn_and_vocab_probs(val_outputs, test_lm_probs, 0.25)
    val_loss = torch.mean(-final_prob).item()
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
        model.load_state_dict(torch.load(f'checkpoints/{dataset}/adaptive_model_weights.pt'))
    print()

