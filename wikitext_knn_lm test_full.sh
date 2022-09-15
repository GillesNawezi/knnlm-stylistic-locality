## preprocess
TEXT=examples/language_model/wikitext103_seg
python preprocess.py \
    --only-source \
    --trainpref $TEXT/valid.txt \
    --validpref $TEXT/valid.txt \
    --testpref $TEXT/test.txt\
    --destdir data-bin/wikitext103_seg \
    --workers 20

python train.py --task language_modeling \
    data-bin/wikitext103_seg \
    --save-dir checkpoints/wikitext103_seg \
    --arch transformer_lm_wiki103 \
    --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 1024 --update-freq 3 --tokens-per-sample 1024 --seed 1  \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --keep-interval-updates 3

## eval lm, test
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext103_seg/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test

## eval lm, valid
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext103_seg/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid


# store test
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext103_seg/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset test \
    --tokens-per-sample 3072 \
    --dstore-mmap checkpoints/wikitext103_seg/test_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 226450 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16

# store valid
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext103_seg/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset valid \
    --tokens-per-sample 3072 \
    --dstore-mmap checkpoints/wikitext103_seg/valid_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 201217 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16



# build test index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103_seg/test_dstore \
    --dstore_size 226450 \
    --faiss_index checkpoints/wikitext103_seg/test_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dimension 1024

# build valid index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103_seg/valid_dstore \
    --dstore_size 201217 \
    --faiss_index checkpoints/wikitext103_seg/valid_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dimension 1024


# eval test with test knn
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext103_seg/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test \
    --dstore-filename checkpoints/wikitext103_seg/test_dstore \
    --indexfile checkpoints/wikitext103_seg/test_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 226450 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16

# eval test with test knn + locality
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext103_seg/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test \
    --dstore-filename checkpoints/wikitext103_seg/test_dstore \
    --indexfile checkpoints/wikitext103_seg/test_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 226450 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \
    --use-locality

# eval valid with test knn + locality
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext103_seg/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid \
    --dstore-filename checkpoints/wikitext103_seg/valid_dstore \
    --indexfile checkpoints/wikitext103_seg/valid_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 201217 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \
    --use-locality

# eval valid with valid knn
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext103_seg/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid \
    --dstore-filename checkpoints/wikitext103_seg/valid_dstore \
    --indexfile checkpoints/wikitext103_seg/valid_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 201217 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \



# preprocess test+train, valid+train
TEXT=examples/language_model/wikitext103_seg
python preprocess.py \
    --only-source \
    --srcdict data-bin/wikitext-103/dict.txt \
    --validpref $TEXT/valid.txt,$TEXT/validtrain.txt \
    --testpref $TEXT/test.txt,$TEXT/testtrain.txt \
    --destdir data-bin/wikitext103_seg \
    --workers 20

# store test+train
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset test1 \
    --tokens-per-sample 3072 --truncate-sequence \
    --dstore-mmap checkpoints/wikitext-103/testtrain_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 91912620 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16

# store valid+train
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset valid1 \
    --tokens-per-sample 3072 --truncate-sequence \
    --dstore-mmap checkpoints/wikitext-103/validtrain_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 91887387 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16

# build test+train index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext-103/testtrain_dstore \
    --dstore_size 91912620 \
    --faiss_index checkpoints/wikitext-103/testtrain_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dimension 1024

# build valid+train index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext-103/validtrain_dstore \
    --dstore_size 91887387 \
    --faiss_index checkpoints/wikitext-103/validtrain_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dimension 1024

# eval test with test+train knn
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test \
    --dstore-filename checkpoints/wikitext-103/testtrain_dstore \
    --indexfile checkpoints/wikitext-103/testtrain_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 91912620 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem

# eval test with test+train knn + locality
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test \
    --dstore-filename checkpoints/wikitext-103/testtrain_dstore \
    --indexfile checkpoints/wikitext-103/testtrain_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 91912620 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \
    --use-locality


# eval valid with valid+train knn
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid \
    --dstore-filename checkpoints/wikitext-103/validtrain_dstore \
    --indexfile checkpoints/wikitext-103/validtrain_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 91887387 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem

# eval valid with valid+train knn + locality
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid \
    --dstore-filename checkpoints/wikitext-103/validtrain_dstore \
    --indexfile checkpoints/wikitext-103/validtrain_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 91887387 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \
    --use-locality
