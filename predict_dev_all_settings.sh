#!/usr/bin/env bash

   
INPUTS=("1task" "2task_ans_sf_1_4" "2task_ans_rea_1_15" "3task_1_4_15") #  
for input in ${INPUTS[*]}; do
    name="dev"

    DATA_FILE="data/${name}.pkl"
    DATA_FILE_GZ="data/${name}.gz"
    DATA_ORIGINAL="2wiki/${name}.json"

    setting="epoch=19-step=104659"

    CHECKPOINTS=checkpoints/$input/

    python3 predictor.py $CHECKPOINTS/$setting $DATA_FILE

    
    file1="${setting}_${name}.preds"

    python3 postprocess.py $CHECKPOINTS/$file1 $DATA_FILE $DATA_FILE_GZ

    file2="${setting}_${name}.preds.processed"

    python3 official_evaluation.py $CHECKPOINTS/$file2 $DATA_ORIGINAL $input

    echo "Finish: ${name}"

done
