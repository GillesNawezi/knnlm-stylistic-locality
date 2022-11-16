import torch
from torch import nn
import faiss
import math
import numpy as np
from fairseq import utils
import time
from fairseq.data import Dictionary


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
        self.args = args

    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + '_keys.npy', dtype=np.float16, mode='r',
                                      shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename + '_vals.npy', dtype=np.int, mode='r',
                                  shape=(self.dstore_size, 1))
        else:
            print('Keys are fp32 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + '_keys.npy', dtype=np.float32, mode='r',
                                      shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename + '_vals.npy', dtype=np.int, mode='r',
                                  shape=(self.dstore_size, 1))

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename + '_keys.npy',
                                                  dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                                  shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension),
                                     dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename + '_vals.npy',
                                              dtype=np.int, mode='r',
                                              shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int)
            print('Loading to memory took {} s'.format(time.time() - start))

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
        if args.use_locality:
            self.modified_dist_cache = []

        # store context vectors for later optimization
        self.context_cache = []
        # store lm probs for ppl calc
        self.lm_prob_cache = []
        # store original tgt for examples
        self.original_tgts = []

        # read in the locality feature from npy file
        print(f"Subset: {args.gen_subset}")
        print(f"Subset: {args.dstore_filename}")
        if 'test' in args.gen_subset:
            if 'java' in args.dstore_filename:
                self.package_locality_features = np.load('examples/language_model/java/java_test_pre.original_path.npy')
                self.project_locality_features = np.load('examples/language_model/java/testProjects.npy')
            elif "style" in args.dstore_filename:
                # Stylistic Locality
                print("Load Styles")
                self.package_locality_features = np.memmap(
                    f'examples/language_model/style_dataset/{args.gen_subset}train.txt.style.npy', dtype='int8', mode='r', shape=(69300, 403059))
                print("Loaded Styles")
            else:
                # wikitext
                # section locality
                self.package_locality_features = np.load(
                    f'examples/language_model/wikitext103_seg/{args.gen_subset}train.txt.sec.npy')
                # domain locality
                self.project_locality_features = np.load(
                    f'examples/language_model/wikitext103_seg/{args.gen_subset}train.txt.dom.npy')
        elif 'valid' in args.gen_subset:
            if 'java' in args.dstore_filename:
                self.package_locality_features = np.load(
                    'examples/language_model/java/java_validation_pre.original_path.npy')
                self.project_locality_features = np.load('examples/language_model/java/validProjects.npy')
            elif "style" in args.dstore_filename:
                # Stylistic Locality
                print("Load Styles")
                self.package_locality_features = np.memmap(
                    f'examples/language_model/style_dataset/{args.gen_subset}train.txt.style.npy', dtype='int8', mode='r', shape=(58905, 392700))
                print("Loaded Styles") 
            else:
                # wikitext
                # section locality
                self.package_locality_features = np.load(
                    f'examples/language_model/wikitext103_seg/{args.gen_subset}train.txt.sec.npy')
                # domain locality
                self.project_locality_features = np.load(
                    f'examples/language_model/wikitext103_seg/{args.gen_subset}train.txt.dom.npy')

        # change dtype to int8 to save space
        try:
            self.package_locality_features = self.package_locality_features.astype('int8')
        except:
            print("No package_locality_features")

        try:    
            self.project_locality_features = self.project_locality_features.astype('int8')
        except:
            print("No project_locality_features")

        # load tuned adaptive model        
        if args.use_locality:
            print("Load Tuned Adaptive Model")
            print(args.path.rsplit('/', 1)[0] + '/adaptive_model_weights.pt')

            if 'java' in args.dstore_filename:
                self.adaptive_model = WeightedDist(nlayers=2, hidden_units=64, num_outputs=5,
                                                   context_dim=512).cuda()
            else:
                self.adaptive_model = WeightedDist(nlayers=2, hidden_units=64).cuda()
            self.adaptive_model.load_state_dict(torch.load(args.path.rsplit('/', 1)[0] + '/adaptive_model_weights.pt'))
            self.adaptive_model.eval()
            if args.fp16:
                self.adaptive_model.half()
            print("Loaded Adaptive Model")

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
        # print(dists,knns)
        return dists, knns

    def get_knn_log_prob(self, queries, tgt, pad_idx, sample_ids=None, task=None,
                         lm_probs=None, calc_vocab_prob=False):
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

        self.original_tgts.append(tgt)

        tgt = tgt.contiguous().view(-1)
        lm_probs = lm_probs.contiguous().view(-1)
        self.lm_prob_cache.append(lm_probs[tgt != pad_idx].cpu().numpy())

        token_sample_ids = sample_ids.repeat(qshape[0], 1).view(-1)

        reduced_token_sample_ids = token_sample_ids[tgt != pad_idx].cpu()
        reduced_tgt = tgt[tgt != pad_idx]

        self.sample_id_cache.append(reduced_token_sample_ids.numpy())
        self.context_cache.append(queries[tgt != pad_idx].cpu().numpy())
        dists, knns = self.get_knns(queries[tgt != pad_idx], sample_ids=reduced_token_sample_ids)

        # locality features
        try:
            project_locality = self.project_locality_features[
                np.tile(reduced_token_sample_ids, (knns.shape[1], 1)).T,
                self.inv_token_sample_map[knns]]
            flat_project_locality = project_locality.flatten()
            project_locality = torch.from_numpy(project_locality).cuda()
            has_project_locality = True
        except Exception as e: 
            print(f"No project_locality_features: {e}")
            has_project_locality = False

        try:     
            package_locality = self.package_locality_features[
                np.tile(reduced_token_sample_ids, (knns.shape[1], 1)).T,
                self.inv_token_sample_map[knns]]
            flat_package_locality = package_locality.flatten()
            package_locality = torch.from_numpy(package_locality).cuda()
            has_package_locality = True
        except Exception as e: 
            has_package_locality = False
            print(f"No package_locality_features: {e}")
            print(f"KNNS: {knns} \n Shape: {knns.shape} \n")
            print(f"inv_token_sample_map: {self.inv_token_sample_map[knns]} \n Shape: {self.inv_token_sample_map[knns].shape} \n")
            print(f"Tiles {np.tile(reduced_token_sample_ids, (knns.shape[1], 1)).T} \n Shape: {np.tile(reduced_token_sample_ids, (knns.shape[1], 1)).T.shape}")
            print(f"Smaple Ids {reduced_token_sample_ids} \n Shape: {reduced_token_sample_ids.shape}")
            x=yzz
            

        # save if retrieved is eq to actual tgt?
        knn_token_ids = self.vals[knns].squeeze(-1)
        correctness = knn_token_ids == \
                      np.expand_dims(reduced_tgt.cpu().numpy(), 1).repeat(knns.shape[1], axis=1)
        correctness = correctness.astype("int8")
        flat_correctness = correctness.flatten()
        # # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        dists = dist_func(dists, knns, queries[tgt != pad_idx, :], function=self.sim_func)

        flat_rank = np.tile(np.arange(1, dists.shape[1] + 1, dtype='int16'), dists.shape[0])
        flat_dists = dists.detach().cpu().numpy().flatten()
        flat_knns = knns.flatten()

        self.dist_cache.append(flat_dists)
        self.knn_cache.append(flat_knns)

        if has_project_locality:
            self.project_locality_cache.append(flat_project_locality)
        if has_package_locality:
            self.package_locality_cache.append(flat_package_locality)
            
        self.rank_cache.append(flat_rank)
        self.correctness_cache.append(flat_correctness)

        if self.args.use_locality:
            if 'java' in self.args.dstore_filename:
                locality_indicator = project_locality + package_locality
                locality_feat = torch.nn.functional.one_hot(locality_indicator.long(), num_classes=3).permute(2, 0, 1)

                # local1 = torch.zeros_like(project_locality, device='cuda')
                # local1[(project_locality == 1) & (package_locality == 0)] = 1
                #
                # # make 3 features, local=0, 1, 2 and mutually exclusive
                # locality_feat = [1 - (local1 | package_locality), local1, package_locality]
                # probs = utils.log_softmax(0.3470 * dists + 0.3350 * package_locality, dim=-1)
                # optimized on test
                # probs = utils.log_softmax(locality_feat[0] * (0.0248 * dists) +
                #                           locality_feat[1] * (0.0385 * dists + 3.9068) +
                #                           locality_feat[2] * (0.0487 * dists + 6.4349), dim=-1)

                modified_dists = locality_feat[0] * (0.0223 * dists) \
                                 + locality_feat[1] * (0.0326 * dists + 3.6268) \
                                 + locality_feat[2] * (0.0411 * dists + 5.9197)
                # params = self.adaptive_model.model(queries[tgt != pad_idx])
                #
                # modified_dists = locality_feat[0] * (params[:, 0][:, None] * dists) + \
                #                  locality_feat[1] * (params[:, 1][:, None] * dists + params[:, 2][:, None]) + \
                #                  locality_feat[2] * (params[:, 3][:, None] * dists + params[:, 4][:, None])
                probs = utils.log_softmax(modified_dists, dim=-1)
            elif 'style' in self.args.dstore_filename:
                locality_indicator = package_locality
                locality_feat = torch.nn.functional.one_hot(locality_indicator.long(), num_classes=2).permute(2, 0, 1)

                params = self.adaptive_model.model(queries[tgt != pad_idx])

                modified_dists = locality_feat[0] * (params[:, 0][:, None] * dists) + \
                                 locality_feat[1] * (params[:, 1][:, None] * dists + params[:, 2][:, None]) 

                probs = utils.log_softmax(modified_dists, dim=-1)
            else:
                # wiki
                locality_indicator = project_locality + 2 * package_locality

                locality_feat = torch.nn.functional.one_hot(locality_indicator.long(), num_classes=4).permute(2, 0, 1)

                params = self.adaptive_model.model(queries[tgt != pad_idx])
                
                modified_dists = locality_feat[0] * (params[:, 0][:, None] * dists) + \
                                 locality_feat[1] * (params[:, 1][:, None] * dists + params[:, 2][:, None]) + \
                                 locality_feat[2] * (params[:, 3][:, None] * dists + params[:, 4][:, None]) + \
                                 locality_feat[3] * (params[:, 5][:, None] * dists + params[:, 6][:, None])

                # optimized on test
                # probs = utils.log_softmax(locality_feat[0] * (1.2721 * dists) +
                #                           locality_feat[1] * (1.3063 * dists + 1.0640) +
                #                           locality_feat[2] * (1.2383 * dists + -0.2982) +
                #                           locality_feat[3] * (1.4713 * dists + 3.1667), dim=-1)

                #modified_dists = locality_feat[0] * (1.2326 * dists) \
                #                 + locality_feat[1] * (1.2459 * dists + 1.0868) \
                #                 + locality_feat[2] * (1.2881 * dists + 1.2495) \
                #                 + locality_feat[3] * (1.2853 * dists + 1.4641)

                # params = self.adaptive_model.model(queries[tgt != pad_idx])
                # modified_dists = locality_feat[0] * (params[:, 0][:, None] * dists) + \
                #                   locality_feat[1] * (params[:, 1][:, None] * dists + params[:, 2][:, None]) + \
                #                   locality_feat[2] * (params[:, 3][:, None] * dists + params[:, 4][:, None]) + \
                #                   locality_feat[3] * (params[:, 5][:, None] * dists + params[:, 6][:, None])

                probs = utils.log_softmax(modified_dists, dim=-1)

            # save modified dists for plotting
            self.modified_dist_cache.append(modified_dists.cpu().numpy())

        else:
            probs = utils.log_softmax(dists, dim=-1)
        knn_token_ids = torch.from_numpy(knn_token_ids).long().cuda()

        # to calculate only the prob on the ground truth tgt token for ppl
        index_mask = torch.eq(knn_token_ids,
                              tgt[tgt != pad_idx].unsqueeze(-1)).float()

        index_mask[index_mask == 0] = -10000  # for stability
        index_mask[index_mask == 1] = 0

        self.index_mask_cache.append(index_mask.cpu().numpy().flatten().astype('int16'))

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()
        full_yhat_knn_prob = torch.full([qshape[0] * qshape[1]], -10000.).cuda()
        full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

        if calc_vocab_prob:
            # calc all vocab item
            vocab_size = len(task.source_dictionary)
            pad_mask = tgt != pad_idx
            yhat_knn_token_prob = torch.full([knn_token_ids.shape[0], vocab_size], -10000.).cuda()
            for i, row in enumerate(knn_token_ids):
                unique_token_ids = row.unique()
                mask = torch.eq(knn_token_ids[i].repeat(unique_token_ids.shape[0], 1),
                                unique_token_ids.unsqueeze(-1)).float()
                mask[mask == 0] = -10000
                mask[mask == 1] = 0
                yhat_knn_token_prob[i, unique_token_ids] = torch.logsumexp(probs[i].repeat(unique_token_ids.shape[0], 1)
                                                                           + mask, dim=-1).clone()
            full_yhat_knn_token_prob = torch.full([qshape[0] * qshape[1], vocab_size], -10000.).cuda()
            full_yhat_knn_token_prob[pad_mask] = yhat_knn_token_prob

            # TxBx1
            return full_yhat_knn_prob.view(qshape[0], qshape[1], 1), full_yhat_knn_token_prob.view(qshape[0], qshape[1], vocab_size)
        else:
            # TxBx1
            return full_yhat_knn_prob.view(qshape[0], qshape[1], 1), None
