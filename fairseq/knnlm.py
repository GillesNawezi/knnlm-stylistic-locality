import torch
import faiss
import math
import numpy as np
from fairseq import utils
import time
from fairseq.data import Dictionary


class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.index = self.setup_faiss(args)

    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int16')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + '_keys.npy', dtype=np.float16, mode='r',
                                      shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename + '_vals.npy', dtype=np.int16, mode='r',
                                  shape=(self.dstore_size, 1))
        else:
            print('Keys are fp32 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + '_keys.npy', dtype=np.float32, mode='r',
                                      shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename + '_vals.npy', dtype=np.int, mode='r',
                                  shape=(self.dstore_size, 1))

        # also read in the token-sample mapping file
        self.token_sample_map = torch.load(args.dstore_filename + '_map.pt')
        self.inv_token_sample_map = np.zeros(self.dstore_size, dtype='i')
        for k, v in self.token_sample_map.items():
            self.inv_token_sample_map[v[0]:v[1]] = k

        # store all the top-k retrieved results
        self.sample_id_cache = []
        self.dist_cache = []
        self.knn_cache = []
        self.project_locality_cache = []
        self.package_locality_cache = []
        self.rank_cache = []
        self.correctness_cache = []
        self.index_mask_cache = []

        # read in the locality feature from npy file
        if 'test' in args.dstore_filename:
            self.package_locality_features = np.load('examples/language_model/java/java_test_pre.original_path.npy')
            self.project_locality_features = np.load('examples/language_model/java/testProjects.npy')
        else:
            self.package_locality_features = np.load(
                'examples/language_model/java/java_validation_pre.original_path.npy')
            self.project_locality_features = np.load('examples/language_model/java/validProjects.npy')

        # change dtype to int8 to save space
        self.package_locality_features = self.package_locality_features.astype('int8')
        self.project_locality_features = self.project_locality_features.astype('int8')

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename + '_keys.npy', dtype=np.float32, mode='r',
                                                  shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension),
                                     dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename + '_vals.npy', dtype=np.int, mode='r',
                                              shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int16 if args.dstore_fp16 else np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int16 if args.dstore_fp16 else np.int)
            print('Loading to memory took {} s'.format(time.time() - start))

        return index

    def get_knns(self, queries, sample_ids=None):
        start = time.time()
        redundancy = 2048
        new_knns = []
        new_dists = []
        total_block_count = 0

        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k + redundancy)

        retrieved_sample_ids = self.inv_token_sample_map[knns]

        for x, y, i, r in zip(knns, dists, sample_ids, retrieved_sample_ids):
            # mask off current query sample
            current_sample_range = self.token_sample_map[i.item()]
            current_sample_mask = (x < current_sample_range[0]) | (x >= current_sample_range[1])
            new_x = x[current_sample_mask]
            new_y = y[current_sample_mask]
            total_block_count += self.k + redundancy - len(new_x)

            new_x = new_x[:self.k]
            new_y = new_y[:self.k]

            if len(new_x) < self.k:
                print('Warining: less than k at', len(new_x))
            new_knns.append(new_x)
            new_dists.append(new_y)
        dists = np.array(new_dists)
        knns = np.array(new_knns)
        # print(dists.shape)
        # print(knns.shape)
        # print(total_block_count)

        # save token_sample_ids, dists and knn retrieves to the disk.
        # self.sample_id_cache.append(sample_ids)
        # self.dist_cache.append(dists)
        # self.knn_cache.append(knns)

        return dists, knns

    def get_knn_log_prob(self, queries, tgt, pad_idx, sample_ids=None, task=None):
        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == 'l2':
                    start = time.time()
                    knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                    if self.half:
                        knns_vecs = knns_vecs.half()
                    query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    l2 = torch.sum((query_vecs - knns_vecs.detach()) ** 2, dim=2)
                    return -1 * l2
                return d

            if function == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * d

            raise ValueError("Invalid knn similarity function!")

        # queries  are TxBxC
        # reshape: (TxB)xC
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])

        # for i in tgt[:, 120]:
        #     print(task.source_dictionary[i], end=' ')

        tgt = tgt.contiguous().view(-1)

        token_sample_ids = sample_ids.repeat(qshape[0], 1).view(-1)

        reduced_token_sample_ids = token_sample_ids[tgt != pad_idx].cpu()
        reduced_tgt = tgt[tgt != pad_idx]

        dists, knns = self.get_knns(queries[tgt != pad_idx], sample_ids=reduced_token_sample_ids)

        # locality features
        project_locality = self.project_locality_features[
            np.tile(reduced_token_sample_ids, (knns.shape[1], 1)).T,
            self.inv_token_sample_map[knns]]
        flat_project_locality = project_locality.flatten()
        project_locality = torch.from_numpy(project_locality).cuda()

        package_locality = self.package_locality_features[
            np.tile(reduced_token_sample_ids, (knns.shape[1], 1)).T,
            self.inv_token_sample_map[knns]]
        flat_package_locality = package_locality.flatten()
        package_locality = torch.from_numpy(package_locality).cuda()

        # ret_token_id = self.vals[knns].squeeze(-1)
        # print(dists[1500][0])
        # print(task.source_dictionary[ret_token_id[1500][0]])
        #

        # save if retrieved is eq to actual tgt?

        correctness = self.vals[knns].squeeze(-1) == \
                      np.expand_dims(reduced_tgt.cpu().numpy(), 1).repeat(knns.shape[1], axis=1)
        correctness = correctness.astype("int8")
        flat_correctness = correctness.flatten()
        # # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        dists = dist_func(dists, knns, queries[tgt != pad_idx, :], function=self.sim_func)

        flat_rank = np.tile(np.arange(1, dists.shape[1] + 1, dtype='int16'), dists.shape[0])
        flat_dists = dists.detach().cpu().numpy().flatten()

        self.dist_cache.append(flat_dists)
        self.project_locality_cache.append(flat_project_locality)
        self.package_locality_cache.append(flat_package_locality)
        self.rank_cache.append(flat_rank)
        self.correctness_cache.append(flat_correctness)

        local1 = torch.zeros_like(project_locality, device='cuda')
        local1[(project_locality == 1) & (package_locality == 0)] = 1

        # make 3 features, local=0, 1, 2 and mutually exclusive
        locality_feat = [1 - (local1 | package_locality), local1, package_locality]

        probs = utils.log_softmax(dists + 15 * project_locality + 15 * package_locality, dim=-1)
        # probs = utils.log_softmax(locality_feat[0] * dists +
        #                           locality_feat[1] * (0.0803 * dists - 105.3669) +
        #                           locality_feat[2] * (0.1285 * dists - 97.1999), dim=-1)

        # to calculate only the prob on the ground truth tgt token for ppl
        index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1),
                              tgt[tgt != pad_idx].unsqueeze(-1)).float()

        index_mask[index_mask == 0] = -10000  # for stability
        index_mask[index_mask == 1] = 0

        self.index_mask_cache.append(index_mask.cpu().numpy().flatten().astype('int16'))

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()

        full_yhat_knn_prob = torch.full([qshape[0] * qshape[1]], -10000).cuda()
        full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

        # TxBx1
        return full_yhat_knn_prob.view(qshape[0], qshape[1], 1)
