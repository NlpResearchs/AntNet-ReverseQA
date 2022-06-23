# Deep Human Answer Understanding for Natural Reverse QA

This is the code and data for Deep Human Answer Understanding for Natural Reverse QA.

## Data set description

Thirty participants were allowed to construct 50 to 60 questions given 18 to 25 answers from seven domains, namely, encyclopedia, insurance, personal, purchases, leisure interests, medical health, and exercise. The details are as follows:

<!-- mdformat off(no table) -->

| Data set | questions | answers | samples(true/false/uncertain) |
| -------- | --------- | ------- | ----------------------------- |
| Tdata    | 536       | 10817   | 4610/4452/1755                |
| Mdata    | 1007      | 23445   | 20929/28876/9989              |

<!-- mdformat on -->

## Code details

**Requirement:**  
>Python=3.6, Tensorflow=1.3.1, pyltp=0.2.1 and numpy=1.16.2

**Train and Test**  
>1. Train the model,you need to download the pretrained model [ltp_model](http://ltp.ai/download.html) and download the bert model [BERT](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) if you want to use bert pretrained vector.

>2. Run  the following command to train or evaluate the AntNet respectively.

    python train_choose.py  
    python test_with_choose.py
    python train_judge.py  
    python test_with_judge.py
