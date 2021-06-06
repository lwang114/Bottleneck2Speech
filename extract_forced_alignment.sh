#!/usr/bin/env bash


. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

stage=1

if [ $stage -le 0 ]; then
  steps/align_si.sh --cmd "$train_cmd" data/train/train_clean_100 data/lang
fi

if [ $stage -le 1 ]; then
for i in exp/tri4b/ali.*.gz;
  do ali-to-phones --ctm-output exp/tri4b/final.mdl \
  ark:"gunzip -c $i|" -> ${i%.gz}.ctm;
  done;
  cd exp/tri4b
  cat *.ctm > merged_alignment.txt
fi
