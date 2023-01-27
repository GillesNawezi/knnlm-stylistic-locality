"""
How to Fine Tune
-Same srcdict as in the pretrained model
-Use --restore-file to reload the checkpoint
-use the same architecture
-use --reset-dataloader
"""

# ===== Source Dict needs to be Expanded. =====
TEXT=examples/language_model/wiki_style_comb
python preprocess.py \
    --only-source \
    --trainpref $TEXT/comb_train.txt \
    --validpref $TEXT/comb_valid.txt \
    --testpref $TEXT/comb_test.txt \
    --destdir data-bin/wiki_style_comb \
    --workers 20 \
    --thresholdsrc 5





cd examples/language_model/
bash prepare-wikitext-103.sh
cd ../..

## preprocess
TEXT=examples/language_model/impolite_dataset
python preprocess.py \
    --only-source \
    --srcdict data-bin/wiki_style_comb/dict.txt \
    --trainpref $TEXT/train.txt \
    --validpref $TEXT/valid.txt \
    --testpref $TEXT/test.txt \
    --destdir data-bin/impolite_dataset \
    --workers 20



# Fine Tune wiki 103


CUDA_VISIBLE_DEVICES=3 python train.py --task language_modeling \
    data-bin/impolite_dataset \
    --save-dir checkpoints/impolite_dataset \
    --arch transformer_lm_wiki103 \
    --max-epoch 300 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 --fp16 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d \
    --reset-lr-scheduler --reset-meters --reset-optimizer --restore-file checkpoints/style_source_wiki_fine_tune/wt103_checkpoint_best.pt

# Continue Training
CUDA_VISIBLE_DEVICES=3 python train.py --task language_modeling \
    data-bin/impolite_dataset \
    --save-dir checkpoints/impolite_dataset \
    --arch transformer_lm_wiki103 \
    --max-epoch 500 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 --fp16 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d \



## eval lm, valid
python eval_lm.py data-bin/impolite_dataset \
    --path checkpoints/impolite_dataset/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid

## eval lm, test
python eval_lm.py data-bin/impolite_dataset \
    --path checkpoints/impolite_dataset/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test


## store valid
CUDA_VISIBLE_DEVICES=3 \
python eval_lm.py data-bin/impolite_dataset \
    --path checkpoints/impolite_dataset/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset valid \
    --tokens-per-sample 3072 \
    --dstore-mmap checkpoints/impolite_dataset/valid_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 6000  --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16


## store test
CUDA_VISIBLE_DEVICES=3 \
python eval_lm.py data-bin/impolite_dataset \
    --path checkpoints/impolite_dataset/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset test \
    --tokens-per-sample 3072 \
    --dstore-mmap checkpoints/impolite_dataset/test_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 6000  --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16

1865434
## build valid index
python build_dstore.py \
    --dstore_mmap checkpoints/impolite_dataset/valid_dstore \
    --dstore_size 6000 \
    --faiss_index checkpoints/impolite_dataset/valid_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dimension 1024

## build test index
python build_dstore.py \
    --dstore_mmap checkpoints/impolite_dataset/test_dstore \
    --dstore_size 6000 \
    --faiss_index checkpoints/impolite_dataset/test_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dimension 1024


# eval valid with valid knn
CUDA_VISIBLE_DEVICES=3 \
python eval_lm.py data-bin/impolite_dataset \
    --path checkpoints/impolite_dataset/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid \
    --dstore-filename checkpoints/impolite_dataset/valid_dstore \
    --indexfile checkpoints/impolite_dataset/valid_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 6000 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \

# eval test with test knn
CUDA_VISIBLE_DEVICES=3 \
python eval_lm.py data-bin/impolite_dataset \
    --path checkpoints/impolite_dataset/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test \
    --dstore-filename checkpoints/impolite_dataset/test_dstore \
    --indexfile checkpoints/impolite_dataset/test_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 6000 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \

# eval valid with valid knn and locality
CUDA_VISIBLE_DEVICES=3 \
python eval_lm.py data-bin/impolite_dataset \
    --path checkpoints/impolite_dataset/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid \
    --dstore-filename checkpoints/impolite_dataset/valid_dstore \
    --indexfile checkpoints/impolite_dataset/valid_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 6000 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \
    --use-locality

# eval test with test knn and locality
python eval_lm.py data-bin/impolite_dataset \
    --path checkpoints/impolite_dataset/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test \
    --dstore-filename checkpoints/impolite_dataset/test_dstore \
    --indexfile checkpoints/impolite_dataset/test_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 6000 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \
    --use-locality

#########################################
#Generate samples
#########################################

#Wiki Model + Merged Dict??
fairseq-interactive data-bin/wikitext-103 \
--task language_modeling \
--path checkpoints/impolite_dataset/wt103_checkpoint_best.pt \
--beam 5

#Wiki Model Fine Tuned with Style Data
fairseq-interactive data-bin/impolite_dataset \
--task language_modeling \
--path checkpoints/impolite_dataset/checkpoint_best.pt \
--beam 5

#With Dstore
fairseq-interactive data-bin/impolite_dataset \
--task language_modeling \
--path checkpoints/impolite_dataset/checkpoint_best.pt \
--dstore-filename checkpoints/impolite_dataset/valid_dstore \
--indexfile checkpoints/impolite_dataset/valid_knn.index  \
--model-overrides "{'knn_keytype': 'last_ffn_input'}" \
--k 1024 --lmbda 0.25 --dstore-size 6000 --knn-keytype last_ffn_input \
--probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem 

#With Knn + Style
fairseq-interactive data-bin/impolite_dataset \
--task language_modeling \
--path checkpoints/impolite_dataset/checkpoint_best.pt \
--dstore-filename checkpoints/impolite_dataset/valid_dstore \
--indexfile checkpoints/impolite_dataset/valid_knn.index  \
--model-overrides "{'knn_keytype': 'last_ffn_input'}" \
--k 1024 --lmbda 0.25 --dstore-size 6000 --knn-keytype last_ffn_input \
--probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \
--use-locality --style toxic


#%%
fairseq-generate data-bin/imimpolite_dataset \
    --task language_modeling \
    --path checkpoints/impolite_dataset/checkpoint_best.pt \
    --gen-subset valid
    --beam 5 


python generate.py data-bin/impolite_dataset \
  --path checkpoints/impolite_dataset/checkpoint_best.pt \
  --beam 5 \
  --batch-size 16 \
  --nbest 5


CUDA_VISIBLE_DEVICES=3 \
python tune_locality_weights_style_source_adaptive.py  

CUDA_VISIBLE_DEVICES=3 \
python tune_locality_weights_style_source_wiki_arch.py  