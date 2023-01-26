#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput
import logging
import math
import sys
import os

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders

from fairseq.knnlm import KNN_Dstore

import pathlib
import pandas as pd

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.interactive')

global_path = str(pathlib.Path(__file__).parent.parent.resolve())

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )

dstore_sizes = {
    "style_source_neutral":4112371,
    "style_source_wiki_fine_tune":2157921,
    "toxic_dataset":37871,"formal_dataset":255577,"informal_dataset":255577,"polite_dataset":6000,
    "impolite_dataset":6000,"supportive_dataset":6000,"offensive_dataset":6000 
}

survey_models = ["style_source_wiki_fine_tune",
                 #"style_source_neutral",
                 "toxic_dataset","formal_dataset","informal_dataset","polite_dataset","impolite_dataset","supportive_dataset",
                 #"offensive_dataset"
                 ]

styles = ["toxic","formal","informal","polite","impolite","supportive","offensive"]

def modify_args(model, args):

    args.data = f"data-bin/{model}"
    args.path = f"checkpoints/{model}/checkpoint_best.pt"
    args.indexfile = f"checkpoints/{model}/valid_knn.index"
    args.dstore_size = dstore_sizes[model]
    args.dstore_filename = f"checkpoints/{model}/valid_dstore"

    if model.replace("_dataset","") not in styles:
        args.use_locality = True
        args.style = "not_set"
    else: 
        args.use_locality = False
        args.style = False

    return args

def main(args):
    utils.import_user_module(args)

    input_file = global_path + "/survey_data/survey_samples.txt"
    output_file = global_path + "/survey_data/survey_data.txt"

    survey_dict_list = []

    for survey_model in survey_models:

        logger.info(f"Load New Model: survey_model")
        args = modify_args(survey_model,args)

        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.max_sentences is None:
            args.max_sentences = 1

        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        logger.info(args)

        use_cuda = torch.cuda.is_available() and not args.cpu

        # Setup task, e.g., translation
        task = tasks.setup_task(args)

        # Load ensemble
        logger.info('loading model(s) from {}'.format(args.path))
        models, _model_args = checkpoint_utils.load_model_ensemble(
            args.path.split(os.pathsep),
            arg_overrides=eval(args.model_overrides),
            task=task,
        )

        # Set dictionaries
        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        # Optimize ensemble for generation
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()
            if use_cuda:
                model.cuda()

        # Initialize generator
        generator = task.build_generator(args)

        # Handle tokenization and BPE
        tokenizer = encoders.build_tokenizer(args)
        bpe = encoders.build_bpe(args)

        def encode_fn(x):
            if tokenizer is not None:
                x = tokenizer.encode(x)
            if bpe is not None:
                x = bpe.encode(x)
            return x

        def decode_fn(x):
            if bpe is not None:
                x = bpe.decode(x)
            if tokenizer is not None:
                x = tokenizer.decode(x)
            return x

                            
        if args.knnlm:
            args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
            args.dict = getattr(args, 'dict', tgt_dict)
            knn_dstore = KNN_Dstore(args)
        else:
            knn_dstore = None

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        align_dict = utils.load_align_dict(args.replace_unk)

        max_positions = utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        )

        if args.buffer_size > 1:
            logger.info('Sentence buffer size: %s', args.buffer_size)
        logger.info('NOTE: hypothesis and token scores are output in base 2')
        logger.info('Type the input sentence and press return:')
        start_id = 0

        with open(input_file, "r") as f:
            survey_samples = f.read().splitlines() 

        for inputs in survey_samples:
            print(f"\nInput sample: {inputs}")
            survey_dict = {}
            survey_dict["input"] = str(inputs)
            survey_dict["model"] = survey_model
            inputs = [inputs]

            if survey_model.replace("_dataset","") in styles:
                style_loop = [survey_model.replace("_dataset","_single_model")]
            else: 
                style_loop = styles

            for style in style_loop:
                args.style = style  
                results = []
                for batch in make_batches(inputs, args, task, max_positions, encode_fn):
                    src_tokens = batch.src_tokens
                    src_lengths = batch.src_lengths
                    if use_cuda:
                        src_tokens = src_tokens.cuda()
                        src_lengths = src_lengths.cuda()

                    sample = {
                        'net_input': {
                            'src_tokens': src_tokens,
                            'src_lengths': src_lengths,
                        },
                    }
                    translations = task.inference_step(generator, models, sample, 
                                                    #kwargs
                                                    args=args, knn_dstore=knn_dstore)

                    for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                        src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                        results.append((start_id + id, src_tokens_i, hypos))

                # sort output to match input order
                for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                        print('S-{}\t{}'.format(id, src_str))

                    # Process top predictions
                    for hypo in hypos[:min(len(hypos), args.nbest)]:
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=hypo['alignment'],
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
                            remove_bpe=args.remove_bpe,
                        )
                        hypo_str = decode_fn(hypo_str)
                        score = hypo['score'] / math.log(2)  # convert to base 2
                        print('H-{}\t{}\t{}'.format(id, score, hypo_str))
                        print('P-{}\t{}'.format(
                            id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                # convert from base e to base 2
                                hypo['positional_scores'].div_(math.log(2)).tolist(),
                            ))
                        ))
                        if args.print_alignment:
                            alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                            print('A-{}\t{}'.format(
                                id,
                                alignment_str
                            ))
                        survey_dict[style] = hypo_str

            survey_dict_list.append(survey_dict)
            # update running id counter
            start_id += len(inputs)
            survey_df = pd.DataFrame(survey_dict_list)
    
    survey_df = survey_df.groupby('input').apply(lambda x: x.apply(lambda y: y.dropna().head(1)))
    survey_df.to_csv(output_file)
    



def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
