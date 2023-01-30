This is the repository for the paper: Analyzing the Effectiveness of the Underlying Reasoning Tasks in
Multi-hop Question Answering - EACL 2023 (Findings).


## Dataset Information
We use two datasets in our experiments: 2WikiMultihopQA and HotpotQA-small
- [Raw data and pre-process data (file .gz) for dev and train of 2Wiki](https://www.dropbox.com/scl/fo/44k9ay42twxax4l7r4ppi/h?dl=0&rlkey=zpcbeaeg5c2ls4b99fgv1yyq5)
- [Raw data and pre-process data for dev and train of HotpotQA-small](https://www.dropbox.com/scl/fo/cs3h9le399e177aqojhth/h?dl=0&rlkey=ncm6a3l9zlcelxpm888kdjenq)
- [Debiased data](https://www.dropbox.com/scl/fo/fccc2tjkvmmfj0fssty48/h?dl=0&rlkey=0bd7jn6w6g65nybdg8bfxizxl)
- [Adversarial data](https://www.dropbox.com/scl/fo/8lic16lp1x6d1iaxn68ed/h?dl=0&rlkey=kj8bfcjhuw1l0iljyjtodm07w)

We follow the steps in https://github.com/yuwfan/HGN to obtain file .gz data from raw data.


## How to Run the Code

### Set up environment
``` bash install_packages.sh ```


### Prepare data for training
- Download bigbird-roberta-base model from this link: https://huggingface.co/google/bigbird-roberta-base
- Edit variables: data_dir, pretrained_model_dir, data_file
- Run: 
``` python3 preprocess.py ```


### Training 
``` python3 main.py ```


### For evaluation on dev file
``` python3 predictor.py $checkpoint $data_file ```

``` python3 postprocess.py $prediction_file $processed_data_file $original_data_file ```

``` python3 official_evaluation.py path/to/prediction path/to/gold ```


## Reproduce the results
- Download our [checkpoints](https://www.dropbox.com/scl/fo/o9l8d8nrshu6z6cvmkof6/h?dl=0&rlkey=qb6o9c33yfahr3y4w2vwgpxo6)
- Run file ``` predict_dev_all_settings.sh ``` (Note: if you want to use this file for the test set in 2Wiki, comment line #25 about evaluation)

## References
- We base on [HGN](https://github.com/yuwfan/HGN) for data preprocessing.
- We re-use the class Example from the HGN model and update it to work with our dataset.

