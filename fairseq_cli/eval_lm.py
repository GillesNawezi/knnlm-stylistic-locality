#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os

import torch
import numpy as np
import pandas as pd

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data import LMContextWindowDataset
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.knnlm import KNN_Dstore

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('fairseq_cli.eval_lm')


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """ increments counters for the sum of log probs of current word and next
            word (given context ending at current word). Since the next word might be at the end of the example,
            or it might be not counted because it is not an ending subword unit,
            also keeps track of how many of those we have seen """
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return '{}\t{}\t{}\t{}\t{}\t{}'.format(self.word, self.count, self.log_prob, self.is_bpe,
                                               self.next_word_prob, self.count - self.missing_next_words)


def main(parsed_args):
    assert parsed_args.path is not None, '--path required for evaluation!'

    utils.import_user_module(parsed_args)

    logger.info(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    #Track performance per Style
    import pathlib
    global_path = str(pathlib.Path(__file__).parent.resolve()) + "/../"
    ex_path = parsed_args.path
    ex_path = ex_path.split("/")[-2]

    if ex_path == "style_source_wiki_fine_tune":
        ex_path = "style_source_dataset"
        
    folder = global_path + f"examples/language_model/{ex_path}/"
    
    print(folder)

    valid_style_file = folder + parsed_args.gen_subset + ".txt.style"
    valid_source_file = folder + parsed_args.gen_subset + ".txt.source"


    with open(valid_style_file, "r") as f:
        styles = f.readlines()
        
    with open(valid_source_file, "r") as f:
        sources = f.readlines()

    df = pd.Series(styles).to_frame()
    df.index = df.index.set_names(["samp_id"])
    df = df.reset_index()
    df = df.rename(columns={0:"style"})
    df["source"] = sources

    styles = df["style"].unique().tolist()
    print(f"Styles in data:{styles}")

    style_performances = {}
    for style in styles:
        style_performances[style] = {
                                     "score_sum":0,
                                     "count":0
                                     }

    sources = df["source"].unique().tolist()
    print(f"Sources in data:{sources}")
    source_performances = {}
    for source in sources:
        source_performances[source] = {
                                     "score_sum":0,
                                     "count":0
                                     }

    # Load ensemble
    logger.info('loading model(s) from {}'.format(parsed_args.path))
    models, args = checkpoint_utils.load_model_ensemble(
        parsed_args.path.split(os.pathsep),
        arg_overrides=eval(parsed_args.model_overrides),
        task=task,
    )

    for arg in vars(parsed_args).keys():
        if arg not in {
            'self_target', 'future_target', 'past_target', 'tokens_per_sample',
            'output_size_dictionary', 'add_bos_token',
        }:
            setattr(args, arg, getattr(parsed_args, arg))

    # reduce tokens per sample by the required context window size
    args.tokens_per_sample -= args.context_window
    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    dataset = task.dataset(args.gen_subset)
    if args.context_window > 0:
        dataset = LMContextWindowDataset(
            dataset=dataset,
            tokens_per_sample=args.tokens_per_sample,
            context_window=args.context_window,
            pad_idx=task.source_dictionary.pad(),
        )
    logger.info('{} {} {} examples'.format(args.data, args.gen_subset, len(dataset)))

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    assert len(models) > 0

    logger.info('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 36000,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(*[
            model.max_positions() for model in models
        ]),
        ignore_invalid_inputs=True,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(task.target_dictionary, args.softmax_batch, args=args)

    score_sum = 0.
    count = 0

    if args.remove_bpe is not None:
        if args.remove_bpe == 'sentencepiece':
            raise NotImplementedError
        else:
            bpe_cont = args.remove_bpe.rstrip()
            bpe_toks = {
                i
                for i in range(len(task.source_dictionary))
                if task.source_dictionary[i].endswith(bpe_cont)
            }
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()

    if args.knnlm and args.save_knnlm_dstore:
        raise ValueError("Cannot use knnlm while trying to build the datastore!")

    if args.knnlm:
        knn_dstore = KNN_Dstore(args)

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()

        if args.save_knnlm_dstore:
            print('keytype being saved:', args.knn_keytype)
            dstore_token_sample_map = {}
            if args.dstore_fp16:
                print('Saving fp16')
                dstore_keys = np.memmap(args.dstore_mmap + '_keys.npy', dtype=np.float16, mode='w+',
                                        shape=(args.dstore_size, args.decoder_embed_dim))
                dstore_vals = np.memmap(args.dstore_mmap + '_vals.npy', dtype=np.int, mode='w+',
                                        shape=(args.dstore_size, 1))
            else:
                print('Saving fp32')
                dstore_keys = np.memmap(args.dstore_mmap + '_keys.npy', dtype=np.float32, mode='w+',
                                        shape=(args.dstore_size, args.decoder_embed_dim))
                dstore_vals = np.memmap(args.dstore_mmap + '_vals.npy', dtype=np.int, mode='w+',
                                        shape=(args.dstore_size, 1))

        dstore_idx = 0

        prediction_save = {'topk': [], 'ref': []}
        for ex_i, sample in enumerate(t):
            if 'net_input' not in sample:
                continue

            # if ex_i > 300:
            #     continue
            try:
                if "style_source_neutral" in args.dstore_filename and 'valid' in args.gen_subset:
                    if ex_i >= 1244:
                        continue
            except:
                pass

            sample = utils.move_to_cuda(sample) if use_cuda else sample
            gen_timer.start()
            if args.knnlm:
                hypos = scorer.generate(models, sample, knn_dstore=knn_dstore, task=task)
            else:
                hypos = scorer.generate(models, sample)
            gen_timer.stop(sample['ntokens'])

            for i, hypos_i in enumerate(hypos):
                hypo = hypos_i[0]
                sample_id = sample['id'][i]

                if args.save_knnlm_dstore:
                    shape = hypo['dstore_keys'].shape
                    # if shape[0] == args.tokens_per_sample:
                    if dstore_idx + shape[0] > args.dstore_size:
                        print("ERROR! dstore_size exceeded!")
                        shape = [args.dstore_size - dstore_idx]
                        hypo['dstore_keys'] = hypo['dstore_keys'][:shape[0]]
                    actual_size = hypo['tokens'].shape[0]
                    dstore_token_sample_map[sample_id.cpu().item()] = (dstore_idx, actual_size + dstore_idx)
                    if args.dstore_fp16:
                        dstore_keys[dstore_idx:actual_size + dstore_idx] = hypo['dstore_keys'][:actual_size, :].view(
                            -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)
                        dstore_vals[dstore_idx:actual_size + dstore_idx] = hypo['tokens'].view(
                            -1, 1).cpu().numpy().astype(np.int)
                    else:
                        dstore_keys[dstore_idx:actual_size + dstore_idx] = hypo['dstore_keys'][:actual_size, :].view(
                            -1, args.decoder_embed_dim).cpu().numpy().astype(np.float32)
                        dstore_vals[dstore_idx:actual_size + dstore_idx] = hypo['tokens'].view(
                            -1, 1).cpu().numpy().astype(np.int)

                    dstore_idx += actual_size

                tokens = hypo['tokens']
                tgt_len = tokens.numel()
                pos_scores = hypo['positional_scores'].float().nan_to_num()
                predicted_topk = hypo['predicted_topk']

                prediction_save['topk'].append(predicted_topk)
                prediction_save['ref'].append(tokens)

                if args.add_bos_token:
                    assert hypo['tokens'][0].item() == task.target_dictionary.bos()
                    tokens = tokens[1:]
                    pos_scores = pos_scores[1:]

                skipped_toks = 0
                if bpe_toks is not None:
                    for i in range(tgt_len - 1):
                        if tokens[i].item() in bpe_toks:
                            skipped_toks += 1
                            pos_scores[i + 1] += pos_scores[i]
                            pos_scores[i] = 0

                # inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
                # if inf_scores.any():
                #    logger.info(
                #        'skipping tokens with inf scores:',
                #        task.target_dictionary.string(tokens[inf_scores.nonzero()])
                #    )
                #    pos_scores = pos_scores[(~inf_scores).nonzero()]

                # ===== Style Validation ====
                #Weise Tokens ihrem Style zu

                style = df.query(f"samp_id=={sample_id}")["style"].values[0]

                style_performances[style]["score_sum"] += pos_scores.sum().cpu()
                style_performances[style]["count"] += pos_scores.numel() - skipped_toks
                
                source = df.query(f"samp_id=={sample_id}")["source"].values[0]

                source_performances[source]["score_sum"] += pos_scores.sum().cpu()
                source_performances[source]["count"] += pos_scores.numel() - skipped_toks

                # ====== Validate Word Level  =====
                # Number of times that words that occur frequently are predicted right. 

                # ===== Test Dict ====
                #sent = task.source_dictionary.dummy_sentence(7)
                #str_sent = task.source_dictionary.string(tensor=sent)
                #print(str_sent)
                # ===== End style validation======
                    
                score_sum += pos_scores.sum().cpu()
                if pd.isna(score_sum.item()):
                    print("Na found")

                count += pos_scores.numel() - skipped_toks

                if args.output_word_probs or args.output_word_stats:
                    w = ''
                    word_prob = []
                    is_bpe = False
                    for i in range(len(tokens)):
                        w_ind = tokens[i].item()
                        w += task.source_dictionary[w_ind]
                        if bpe_toks is not None and w_ind in bpe_toks:
                            w = w[:-bpe_len]
                            is_bpe = True
                        else:
                            word_prob.append((w, pos_scores[i].item()))

                            next_prob = None
                            ind = i + 1
                            while ind < len(tokens):
                                if pos_scores[ind].item() != 0:
                                    next_prob = pos_scores[ind]
                                    break
                                ind += 1

                            word_stats.setdefault(w, WordStat(w, is_bpe)).add(pos_scores[i].item(), next_prob)
                            is_bpe = False
                            w = ''
                    if args.output_word_probs:
                                     
                        try:
                            top100_score
                        except NameError:
                            top100_score = 0 
                            top100_count = 0

                            last100_score = 0 
                            last100_count = 0

                            #folder = str(pathlib.Path(__file__).parent.resolve()) + "/../../"
                            with open(global_path + "word_level_analysis/output/top100","r") as f:
                                top_words = f.readlines()
                            top_words = [x.replace("\n","") for x in top_words]

                            with open(global_path + "word_level_analysis/output/last100","r") as f:
                                last_words = f.readlines()
                            last_words = [x.replace("\n","") for x in last_words]

                        for x in word_prob:
                            if x[0] in top_words:
                                print(x[0])
                                top100_score += x[1]
                                top100_count +=1
                            elif x[0] in last_words:
                                print(x[0])
                                last100_score += x[1]
                                last100_count +=1

       
                            
                        # logger.info(
                        #    str(int(sample_id)) + " "
                        #    + ('\t'.join('{} [{:2f}]'.format(x[0], x[1]) for x in word_prob))
                        # )
                        

            wps_meter.update(sample['ntokens'])
            t.log({'wps': round(wps_meter.avg)})

    if args.save_knnlm_dstore:
        print("dstore_idx", dstore_idx, "final shape", shape)
        print("Keys", dstore_keys.shape, dstore_keys.dtype)
        print("Vals", dstore_vals.shape, dstore_vals.dtype)
        # save mapping
        print("token sample id mapping size:", len(dstore_token_sample_map))
        torch.save(dstore_token_sample_map, args.dstore_mmap + '_map.pt')

    avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
    logger.info('Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(
        gen_timer.n, gen_timer.sum, 1. / gen_timer.avg
    ))
    logger.info('Loss (base 2): {:.4f}, Perplexity: {:.2f}'.format(
        avg_nll_loss, 2 ** avg_nll_loss
    ))

    print("Performance by Style:")
    for style,values in style_performances.items():
        
        score_sum_style = values["score_sum"]
        count_style = values["count"]

        avg_nll_loss_style = -score_sum_style / count_style / math.log(2)  # convert to base 2
        print("\n")
        print(f"##### {style} #####")
        logger.info('Loss (base 2): {:.4f}, Perplexity: {:.2f}'.format(
            avg_nll_loss_style, 2 ** avg_nll_loss_style
        ))
        print("\n")
    
    print("Performance by Source:")
    for source,values in source_performances.items():
        
        score_sum_source = values["score_sum"]
        count_source = values["count"]

        avg_nll_loss_source = -score_sum_source / count_source / math.log(2)  # convert to base 2
        print("\n")
        print(f"##### {source} #####")
        logger.info('Loss (base 2): {:.4f}, Perplexity: {:.2f}'.format(
            avg_nll_loss_source, 2 ** avg_nll_loss_source
        ))
        print("\n")

    print("Top 100 words perfromance")
    if args.output_word_probs:

        avg_nll_loss_top = -top100_score / top100_count / math.log(2)  # convert to base 2
        print("\n")
        print(f"##### Top 100 #####")
        logger.info('Loss (base 2): {:.4f}, Perplexity: {:.2f}'.format(
            avg_nll_loss_top, 2 ** avg_nll_loss_top
        ))

    print("Last 100 words perfromance")
    if args.output_word_probs:

        avg_nll_loss_last = -last100_score / last100_count / math.log(2)  # convert to base 2
        print("\n")
        print(f"##### Last 100 #####")
        logger.info('Loss (base 2): {:.4f}, Perplexity: {:.2f}'.format(
            avg_nll_loss_last, 2 ** avg_nll_loss_last
        ))



    # save prediction result
    torch.save(prediction_save, 'prediction.pt')
    if args.knnlm:
        dir_name = args.dstore_filename.split('/')[-2]
        if not os.path.exists('saved_tensors/' + dir_name):
            os.makedirs('saved_tensors/' + dir_name)

        split_name = args.gen_subset
        # save dictionary
        torch.save(task.source_dictionary, 'saved_tensors/{}/dictionary.pt'.format(dir_name))
        torch.save(knn_dstore.original_tgts, 'saved_tensors/{}/{}_original_tgts_cache.npy'.format(dir_name, split_name))
        np.save('saved_tensors/{}/{}_sample_id_cache.npy'.format(dir_name, split_name),
                np.concatenate(knn_dstore.sample_id_cache))
        np.save('saved_tensors/{}/{}_knn_cache.npy'.format(dir_name, split_name),
                np.concatenate(knn_dstore.knn_cache))
        np.save('saved_tensors/{}/{}_proj_dist_cache.npy'.format(dir_name, split_name),
                np.concatenate(knn_dstore.dist_cache))
        try:
            np.save('saved_tensors/{}/{}_proj_locality_cache.npy'.format(dir_name, split_name),
                    np.concatenate(knn_dstore.project_locality_cache))
        except Exception as e:
            print(f"Error While Saving Project Tensor {e}")
        try:
            np.save('saved_tensors/{}/{}_pkg_locality_cache.npy'.format(dir_name, split_name),
                    np.concatenate(knn_dstore.package_locality_cache))
        except Exception as e:
            print(f"Error While Saving Package Tensor {e}")
        try:
            np.save('saved_tensors/{}/{}_cat_locality_cache.npy'.format(dir_name, split_name),
                    np.concatenate(knn_dstore.category_locality_cache))
        except Exception as e:
            print(f"Error While Saving Category Tensor {e}")
            
        np.save('saved_tensors/{}/{}_proj_rank_cache.npy'.format(dir_name, split_name),
                np.concatenate(knn_dstore.rank_cache))
        np.save('saved_tensors/{}/{}_proj_correctness_cache.npy'.format(dir_name, split_name),
                np.concatenate(knn_dstore.correctness_cache))
        np.save('saved_tensors/{}/{}_proj_index_mask_cache.npy'.format(dir_name, split_name),
                np.concatenate(knn_dstore.index_mask_cache))
        np.save('saved_tensors/{}/{}_context_cache.npy'.format(dir_name, split_name),
                np.concatenate(knn_dstore.context_cache))
        np.save('saved_tensors/{}/{}_lm_prob_cache.npy'.format(dir_name, split_name),
                np.concatenate(knn_dstore.lm_prob_cache))

        if args.use_locality:
            np.save('saved_tensors/{}/{}_modified_dist_cache.npy'.format(dir_name, split_name),
                    np.concatenate(knn_dstore.modified_dist_cache))

    #if args.output_word_stats:
        #for ws in sorted(word_stats.values(), key=lambda x: x.count, reverse=True):
        #    logger.info(ws)


def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
