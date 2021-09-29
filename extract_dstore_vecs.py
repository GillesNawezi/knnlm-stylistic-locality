import numpy as np

from fairseq.data import Dictionary

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')

keys = np.memmap('checkpoints/wikitext103-bpe/dstore_keys.npy',
                 dtype=np.float16, shape=(153225485, 512))
vals = np.memmap('checkpoints/wikitext103-bpe/dstore_vals.npy',
                 dtype=np.int, mode='r', shape=(153225485, 1))
print(vals.dtype)

vals = vals.squeeze()
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

