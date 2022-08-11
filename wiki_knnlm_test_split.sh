TEXT=examples/language_model/wikitext103_split
python preprocess.py \
    --only-source \
    --trainpref $TEXT/wiki_train_tokens \
    --validpref $TEXT/wiki_valid_tokens \
    --testpref $TEXT/wiki_test_tokens \
    --destdir data-bin/wikitext103_split \
    --workers 20


python train.py --task language_modeling \
    data-bin/wikitext103_split \
    --save-dir checkpoints/wikitext103_split \
    --arch transformer_lm_wiki103 \
    --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 1024 --update-freq 3 --tokens-per-sample 1024 --seed 1  \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d


## eval lm, valid
python eval_lm.py data-bin/wikitext103_seg \
    --path checkpoints/wikitext-103/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid


# store valid
python eval_lm.py data-bin/wikitext103_split \
    --path checkpoints/wikitext103_split/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 1024 \
    --softmax-batch 1024 --gen-subset valid \
    --tokens-per-sample 1024 \
    --dstore-mmap checkpoints/wikitext103_split/valid_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 300000 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore 

# build valid index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103_split/valid_dstore \
    --dstore_size 201217 \
    --faiss_index checkpoints/wikitext103_split/valid_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dimension 1024


# eval valid with valid knn
python eval_lm.py data-bin/wikitext103_split \
    --path checkpoints/wikitext103_split/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 1024 \
    --softmax-batch 1024 \
    --gen-subset valid \
    --dstore-filename checkpoints/wikitext103_split/valid_dstore \
    --indexfile checkpoints/wikitext103_split/valid_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 300000 --knn-keytype last_ffn_input \
    --probe 32 --knnlm 