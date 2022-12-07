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
TEXT=examples/language_model/style_source_dataset
python preprocess.py \
    --only-source \
    --srcdict data-bin/wiki_style_comb/dict.txt \
    --trainpref $TEXT/train.txt \
    --validpref $TEXT/valid.txt \
    --testpref $TEXT/test.txt \
    --destdir data-bin/style_source_wiki_fine_tune \
    --workers 20



# Fine Tune wiki 103


CUDA_VISIBLE_DEVICES=2 python train.py --task language_modeling \
    data-bin/style_source_wiki_fine_tune \
    --save-dir checkpoints/style_source_wiki_fine_tune \
    --arch transformer_lm_wiki103 \
    --max-epoch 300 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 --fp16 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d \
    --reset-lr-scheduler --reset-meters --reset-optimizer --restore-file checkpoints/style_source_wiki_fine_tune/wt103_checkpoint_best.pt

# Continue Training
CUDA_VISIBLE_DEVICES=2 python train.py --task language_modeling \
    data-bin/style_source_wiki_fine_tune \
    --save-dir checkpoints/style_source_wiki_fine_tune \
    --arch transformer_lm_wiki103 \
    --max-epoch 500 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 --fp16 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d \



## eval lm, valid
python eval_lm.py data-bin/style_source_wiki_fine_tune \
    --path checkpoints/style_source_wiki_fine_tune/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid

## eval lm, test
python eval_lm.py data-bin/style_source_wiki_fine_tune \
    --path checkpoints/style_source_wiki_fine_tune/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test


## store valid
CUDA_VISIBLE_DEVICES=2  \
python eval_lm.py data-bin/style_source_wiki_fine_tune \
    --path checkpoints/style_source_wiki_fine_tune/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset valid \
    --tokens-per-sample 3072 \
    --dstore-mmap checkpoints/style_source_wiki_fine_tune/valid_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 2157921  --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16


## store test
CUDA_VISIBLE_DEVICES=2  \
python eval_lm.py data-bin/style_source_wiki_fine_tune \
    --path checkpoints/style_source_wiki_fine_tune/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset test \
    --tokens-per-sample 3072 \
    --dstore-mmap checkpoints/style_source_wiki_fine_tune/test_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 2157921  --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16

1865434
## build valid index
python build_dstore.py \
    --dstore_mmap checkpoints/style_source_wiki_fine_tune/valid_dstore \
    --dstore_size 2157921 \
    --faiss_index checkpoints/style_source_wiki_fine_tune/valid_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dimension 1024

## build test index
python build_dstore.py \
    --dstore_mmap checkpoints/style_source_wiki_fine_tune/test_dstore \
    --dstore_size 2157921 \
    --faiss_index checkpoints/style_source_wiki_fine_tune/test_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dimension 1024


# eval valid with valid knn
python eval_lm.py data-bin/style_source_wiki_fine_tune \
    --path checkpoints/style_source_wiki_fine_tune/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid \
    --dstore-filename checkpoints/style_source_wiki_fine_tune/valid_dstore \
    --indexfile checkpoints/style_source_wiki_fine_tune/valid_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 2157921 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \

# eval test with test knn
python eval_lm.py data-bin/style_source_wiki_fine_tune \
    --path checkpoints/style_source_wiki_fine_tune/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test \
    --dstore-filename checkpoints/style_source_wiki_fine_tune/test_dstore \
    --indexfile checkpoints/style_source_wiki_fine_tune/test_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 2157921 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \

# eval valid with valid knn and locality
python eval_lm.py data-bin/style_source_wiki_fine_tune \
    --path checkpoints/style_source_wiki_fine_tune/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid \
    --dstore-filename checkpoints/style_source_wiki_fine_tune/valid_dstore \
    --indexfile checkpoints/style_source_wiki_fine_tune/valid_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 1807214 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \
    --use-locality

# eval test with test knn and locality
python eval_lm.py data-bin/style_source_wiki_fine_tune \
    --path checkpoints/style_source_wiki_fine_tune/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test \
    --dstore-filename checkpoints/style_source_wiki_fine_tune/test_dstore \
    --indexfile checkpoints/style_source_wiki_fine_tune/test_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 1807214 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \
    --use-locality

#generate samples
fairseq-interactive data-bin/wikitext103_seg \
--task language_modeling \
--path checkpoints/style_source_wiki_fine_tune/checkpoint_best.pt \
#--beam 5