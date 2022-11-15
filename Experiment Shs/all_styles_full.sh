## preprocess
TEXT=examples/language_model/style_dataset_full
python preprocess.py \
    --only-source \
    --trainpref $TEXT/valid.txt \
    --validpref $TEXT/valid.txt \
    --testpref $TEXT/test.txt\
    --destdir data-bin/style_dataset_full \
    --workers 20


CUDA_VISIBLE_DEVICES=2 python train.py --task language_modeling \
    data-bin/style_dataset_full \
    --save-dir checkpoints/style_dataset_full \
    --arch transformer_lm_wiki103 \
    --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 1024 --update-freq 3 --tokens-per-sample 1024 --seed 1  \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --keep-interval-updates 3  

## eval lm, valid
python eval_lm.py data-bin/style_dataset_full \
    --path checkpoints/style_dataset_full/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid

## store valid
CUDA_VISIBLE_DEVICES=2  \
python eval_lm.py data-bin/style_dataset_full \
    --path checkpoints/style_dataset_full/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset valid \
    --tokens-per-sample 3072 \
    --dstore-mmap checkpoints/style_dataset_full/valid_dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 11663184 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16

# Dstore = 11664720 - 1536 = 11663184
## build valid index
python build_dstore.py \
    --dstore_mmap checkpoints/style_dataset_full/valid_dstore \
    --dstore_size 11663184 \
    --faiss_index checkpoints/style_dataset_full/valid_knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dimension 1024


# eval valid with valid knn
python eval_lm.py data-bin/style_dataset_full \
    --path checkpoints/style_dataset_full/checkpoint_best.pt \
    --sample-break-mode eos --max-tokens 3072 \
    --softmax-batch 1024 \
    --gen-subset valid \
    --dstore-filename checkpoints/style_dataset_full/valid_dstore \
    --indexfile checkpoints/style_dataset_full/valid_knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 11663184 --knn-keytype last_ffn_input \
    --probe 32 --knnlm  --fp16 --knn-sim-func "do_not_recomp_l2" --no-load-keys --move-dstore-to-mem \
