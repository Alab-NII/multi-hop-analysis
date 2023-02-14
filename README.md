This is the repository for the paper: [Analyzing the Effectiveness of the Underlying Reasoning Tasks in
Multi-hop Question Answering](https://arxiv.org/abs/2302.05963) - EACL 2023 (Findings).


## Dataset Information
We use two datasets in our experiments: 2WikiMultihopQA and HotpotQA-small
- [Pre-process data (file .gz) for dev and train of 2Wiki](https://www.dropbox.com/s/dcrr5m0sxhexr84/2wiki.zip?dl=0) (Please download raw data from the Github repository of the [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) dataset)
- [Raw data and pre-process data for dev and train of HotpotQA-small](https://www.dropbox.com/s/uqir2a5pkvi1383/hotpotqa-small.zip?dl=0)
- [Debiased data](https://www.dropbox.com/s/34551va9ydr8zgf/debiased-data.zip?dl=0)
- [Adversarial data](https://www.dropbox.com/s/dkcm3m16u13lf29/adversarial-data.zip?dl=0)

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
- Download our [checkpoints](https://www.dropbox.com/s/b0d65poctqs38w8/checkpoints.zip?dl=0)
- Run file ``` predict_dev_all_settings.sh ``` (Note: if you want to use this file for the test set in 2Wiki, comment line #25 about evaluation)

## References
- We base on [HGN](https://github.com/yuwfan/HGN) for data preprocessing.
- We re-use the class Example from the HGN model and update it to work with our dataset.

