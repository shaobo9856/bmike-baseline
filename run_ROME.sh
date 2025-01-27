#!/bin/bash

LANGS=("vi" "es") #"es"  "ru" "zh"
DATAS=("zsRE/zsre_test_" "CounterFact/counterfact_test_" "WikiFactDiff/wfd_test_")
CUDA=3

for DATA in "${DATAS[@]}";do
    for LANG in "${LANGS[@]}";do
        echo "currently processing language: $LANG with data: $DATA"
        CUDA_VISIBLE_DEVICES=$CUDA python run_zsre_llama2.py --lang1 en --lang2 $LANG --editing_method ROME --hparams_dir ./hparams/ROME/llama3.2-3b.yaml  --data_dir $DATA 
    done
done
