import numpy as np

num_retrieved = 1024
dists = np.load('saved_tensors/java-huge-bpe-2000/test_proj_dist_cache.npy').reshape(-1, num_retrieved)
ranks = np.load('saved_tensors/java-huge-bpe-2000/test_proj_rank_cache.npy').reshape(-1, num_retrieved)
pkg_locality = np.load('saved_tensors/java-huge-bpe-2000/test_pkg_locality_cache.npy').reshape(-1, num_retrieved)
proj_locality = np.load('saved_tensors/java-huge-bpe-2000/test_proj_locality_cache.npy').reshape(-1, num_retrieved)
correctness = np.load('saved_tensors/java-huge-bpe-2000/test_proj_correctness_cache.npy').reshape(-1, num_retrieved)

project_local_only = (proj_locality == 1) & (pkg_locality == 0).astype('int8')
locality = project_local_only + 2 * pkg_locality
arr_all = np.stack([dists, ranks, locality, correctness], axis=2)

number_of_rows = arr_all.shape[0]
random_indices = np.random.choice(number_of_rows, size=250000, replace=False)

random_rows = arr_all[random_indices, :, :]
np.save('saved_tensors/java-huge-bpe-2000/downsampled.npy', random_rows)
