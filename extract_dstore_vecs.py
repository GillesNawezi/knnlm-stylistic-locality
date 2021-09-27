import numpy as np
import torch

dictionary = torch.load('saved_tensors/wikitext-103/dictionary.pt')

keys = np.memmap('checkpoints/wikitext-103/validtrain_dstore_keys.npy', dtype=np.float16, shape=(91887387, 1024))
vals = np.memmap('checkpoints/wikitext-103/validtrain_dstore_vals.npy',
                 dtype=np.int, mode='r', shape=(91887387, 1))
vals = vals.squeeze()
print(len(dictionary))
token_id = dictionary.index('decide')
filtered_keys = keys[vals == token_id, :]

print(filtered_keys.shape)

np.savetxt('decide.txt', filtered_keys, delimiter='\t', fmt='%1.10f')
