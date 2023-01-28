This is the repository for the paper: Analyzing the Effectiveness of the Underlying Reasoning Tasks in
Multi-hop Question Answering - EACL 2023 (Findings).


## Dataset Information
We use two datasets in our experiments: 2WikiMultihopQA and HotpotQA-small
- [Raw data and pre-process data (file .gz) for dev and train of 2Wiki]()
- [Raw data and pre-process data for dev and train of HotpotQA-small]()
- [Debiased data]()
- [Adversarial data]()

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
- Run
``` python3 main.py ```


### For evaluation on dev file
- Run
``` python3 predictor.py $checkpoint $data_file ```

``` python3 postprocess.py $prediction_file $processed_data_file $original_data_file ```

``` python3 official_evaluation.py path/to/prediction path/to/gold ```


## Reproduce the results
- Donwload our checkpoints
- Run file predict_dev_all_settings.sh
