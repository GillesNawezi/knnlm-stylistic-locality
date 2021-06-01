## java-huge-bpe-2000
# store valid
python eval_lm.py data-bin/java-huge-bpe-2000-half \
    --path checkpoints/java-huge-bpe-2000/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset valid \
    --tokens-per-sample 3072 --truncate-sequence \
    --dstore-mmap checkpoints/java-huge-bpe-2000/valid_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 4511244 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16

# build valid index
python build_dstore.py \
    --dstore_mmap checkpoints/java-huge-bpe-2000/valid_dstore \
    --dstore_size 4511244 \
    --faiss_index checkpoints/java-huge-bpe-2000/valid_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore_fp16 --dimension 512

# eval valid with valid knn
python eval_lm.py data-bin/java-huge-bpe-2000-half \
    --path checkpoints/java-huge-bpe-2000/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe --truncate-sequence \
    --dstore-filename checkpoints/java-huge-bpe-2000/valid_dstore \
    --indexfile checkpoints/java-huge-bpe-2000/valid_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 4511244 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16

# eval test with valid knn + locality
python eval_lm.py data-bin/java-huge-bpe-2000-half \
    --path checkpoints/java-huge-bpe-2000/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe --truncate-sequence \
    --dstore-filename checkpoints/java-huge-bpe-2000/valid_dstore \
    --indexfile checkpoints/java-huge-bpe-2000/valid_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 4511244 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --use-locality

# store test
python eval_lm.py data-bin/java-huge-bpe-2000-half \
    --path checkpoints/java-huge-bpe-2000/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset test \
    --tokens-per-sample 3072 --truncate-sequence \
    --dstore-mmap checkpoints/java-huge-bpe-2000/test_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 7201445 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16

# build test index
python build_dstore.py \
    --dstore_mmap checkpoints/java-huge-bpe-2000/test_dstore \
    --dstore_size 7201445 \
    --faiss_index checkpoints/java-huge-bpe-2000/test_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore_fp16 --dimension 512

# eval test with test knn
python eval_lm.py data-bin/java-huge-bpe-2000-half \
    --path checkpoints/java-huge-bpe-2000/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test --bpe subword_nmt --remove-bpe --truncate-sequence \
    --dstore-filename checkpoints/java-huge-bpe-2000/test_dstore \
    --indexfile checkpoints/java-huge-bpe-2000/test_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 7201445 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16

# eval test with test knn + locality
python eval_lm.py data-bin/java-huge-bpe-2000-half \
    --path checkpoints/java-huge-bpe-2000/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset test --bpe subword_nmt --remove-bpe --truncate-sequence \
    --dstore-filename checkpoints/java-huge-bpe-2000/test_dstore \
    --indexfile checkpoints/java-huge-bpe-2000/test_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 7201445 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --use-locality
