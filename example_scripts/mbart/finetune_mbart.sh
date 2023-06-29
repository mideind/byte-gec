#!/bin/bash
set -ex

CKPT_DIR=
mkdir -p $CKPT_DIR
RUN_LOG=run-log
mkdir -p $CKPT_DIR/logs

DATA_BIN=data
STARTING_CHECKPOINT=

VOCAB_JSON=eng-isl-bbpe-32k-vocab.json
VOCAB_MERGES=eng-isl-bbpe-32k-merges.txt

TOTAL_NUM_UPDATES=100000
WARMUP_UPDATES=1000
LR=2e-06
MAX_TOKENS=3000
UPDATE_FREQ=1
SRC=is_err
TGT=is_corr
DROPOUT=0.1
LANGS=$SRC,$TGT
VAL_AFTER_UPD=500

fairseq-train $DATA_BIN \
    --restore-file $STARTING_CHECKPOINT \
    --reset-dataloader \
    --reset-lr-scheduler \
    --reset-meters \
    --reset-optimizer \
    --no-progress-bar --fp16 \
    --task translation_from_pretrained_bart --arch mbart_large \
    --encoder-normalize-before --decoder-normalize-before \
    --layernorm-embedding \
    --langs $LANGS \
    --valid-subset valid \
    --bpe gpt2 \
    --gpt2-encoder-json $VOCAB_JSON \
    --gpt2-vocab-bpe $VOCAB_MERGES \
    --max-tokens $MAX_TOKENS --max-source-positions 1024 \
    --max-target-positions 1024 \
    --dropout $DROPOUT --attention-dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 0.0 \
    --lr $LR --lr-scheduler inverse_sqrt \
    --weight-decay 0.001 \
    --update-freq $UPDATE_FREQ \
    --warmup-updates $WARMUP_UPDATES --warmup-init-lr $LR \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.0 \
    --save-interval-updates 1000 --max-update $TOTAL_NUM_UPDATES \
    --validate-after-updates $VAL_AFTER_UPD \
    --keep-interval-updates 10 \
    --no-epoch-checkpoints --patience 50 \
    --tensorboard-logdir $CKPT_DIR/logs \
    --save-dir "$CKPT_DIR" \
    --skip-invalid-size-inputs-valid-test \
    --log-interval 100 \
    --fp16-init-scale 64 \
    --num-workers 16 | tee -a $CKPT_DIR/$RUN_LOG
#    --prepend-bos \
