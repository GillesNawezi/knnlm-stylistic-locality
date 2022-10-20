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

