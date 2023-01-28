#!/bin/bash
pip install tqdm
pip install spacy
python3 -m spacy download en_core_web_lg
pip install joblib
pip install numpy
pip install pandas
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning
pip install transformers sentencepiece
pip install loguru