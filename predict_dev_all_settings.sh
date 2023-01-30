#!/usr/bin/env bash

   
INPUTS=("1task" "2task_ans_sf" "2task_ans_rea" "3task") #  
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

    python3 official_evaluation.py $CHECKPOINTS/$file2 $DATA_ORIGINAL

    echo "Finish: ${name}"

done
