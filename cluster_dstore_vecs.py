import numpy as np

from fairseq.data import Dictionary

dstore_size = 153225485
vec_dim = 512

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')

print(dictionary.indices)


keys_from_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_keys.npy',
                             dtype=np.float16, shape=(dstore_size, vec_dim))
vals_from_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_vals.npy',
                             dtype=np.int, mode='r', shape=(dstore_size, 1))

print(np.max(vals_from_memmap))

exit()
keys = np.zeros((dstore_size, vec_dim), dtype=np.float16)
vals = np.zeros((dstore_size, 1), dtype=np.int)

keys[:] = keys_from_memmap[:]
vals[:] = vals_from_memmap[:]
del keys_from_memmap, vals_from_memmap

print(keys.dtype)
print(keys.shape)

print(vals.dtype)
print(vals.shape)

vals = vals.squeeze()
exit()
print(len(dictionary))

to_save = ['bank', 'shore', 'institution', 'beautiful']
vectors = []
labels = []
for word in to_save:
    token_id = dictionary.index(word)
    filtered_keys = keys[vals == token_id, :]
    print(filtered_keys.shape)
    labels += [word] * filtered_keys.shape[0]
    vectors.append(filtered_keys)
    # np.savetxt('decide.txt', filtered_keys, delimiter='\t', fmt='%1.10f')
all_vecs = np.concatenate(vectors, axis=0)

np.savetxt('contextvecs.txt', all_vecs, delimiter='\t', fmt='%.15f')
with open('labels.txt', 'w') as outfile:
    for w in labels:
        outfile.write(w + '\n')
