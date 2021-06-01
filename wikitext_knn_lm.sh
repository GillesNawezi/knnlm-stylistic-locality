## preprocess
TEXT=examples/language_model/wikitext103_seg
python preprocess.py \
    --only-source \
    --srcdict data-bin/wikitext-103/dict.txt \
    --validpref $TEXT/valid.txt \
    --testpref $TEXT/test.txt \
    --destdir data-bin/wikitext103_seg \
    --workers 20

## eval lm, test
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test

## eval lm, valid
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid


# store test
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset test \
    --tokens-per-sample 3072 \
    --dstore-mmap checkpoints/wikitext-103/test_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 226450 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16

# store valid
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset valid \
    --tokens-per-sample 3072 \
    --dstore-mmap checkpoints/wikitext-103/valid_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 201217 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16



# build test index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext-103/test_dstore \
    --dstore_size 226450 \
    --faiss_index checkpoints/wikitext-103/test_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dimension 1024

# build valid index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext-103/valid_dstore \
    --dstore_size 201217 \
    --faiss_index checkpoints/wikitext-103/valid_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dimension 1024


# eval test with test knn
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test \
    --dstore-filename checkpoints/wikitext-103/test_dstore \
    --indexfile checkpoints/wikitext-103/test_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 226450 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16

# eval valid with valid knn
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid \
    --dstore-filename checkpoints/wikitext-103/valid_dstore \
    --indexfile checkpoints/wikitext-103/valid_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 201217 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16


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
