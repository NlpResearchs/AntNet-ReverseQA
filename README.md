# AntNet-rverseQA

The compiled data and implementation of AUNet understanding the answer of Reverse QA.

## Data set description

Twenty-four participants were allowed to construct 50 to 60 questions given 18 to 22 answers from seven domains such as purchases, leisure interests and medical health. The details given blow:

<!-- mdformat off(no table) -->

| Data set | questions | answers | samples(true/false/uncertain) |
| -------- | --------- | ------- | ----------------------------- |
| Tdata    | 536       | 10817   | 4610/4452/1755                |
| Mdata    | 517       | 23445   | 20929/28876/9989              |

<!-- mdformat on -->

## Code details

**Requirement:**  
>Python=3.6, Tensorflow=1.3.1, pyltp=0.2.1 and numpy=1.16.2

**Train and Test**  
>1. Train the model,you need to download the pretrained model [ltp_model](http://ltp.ai/download.html) and download the bert model [BERT](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) if you want to use bert pretrained vector.

>2. Run  the following command to train or evaluate the AntNet respectively.

```shell
python train_choose.py  
python test_with_choose.py
python train_judge.py  
python test_with_judge.py
