## MASCOT Encoder and Instance Attention Network for Few-Shot Relation Classification

**model evaluation performance:** 

| Dataset  |  5-way-1-shot  | 5-way-5-shot | 10-way-1-shot | 10-way-5-shot |
| :----: | :----: | :----: | :----: | :----: |
| FelRel | 81.28      | 89.25       |69.03       |80.65       | 

**file structure:**  

    |--data
        |--glove.6B.50d.json  
        |--glove.local.50d.json  
        |--train.json 
        |--train_acd.json
        |--test_acd.json 
        |--val.json  
        |--val_pubmed.json  
    |--models
        |--baseline.py
        |--IA.py
        |--IA_pair.py
        |--MAIN.py
        |--MANN.py
        |--MLMAN.py
        |--PR.py
        |--PT.py
        |--Relation.py
    |--toolkits
        |--encoders
            |--BRAN.py
            |--Mascot.py
            |--encoder_BRAN.py
            |--encoder_Mascot.py
            |--encoder_CNN.py
            |--encoder_TE.py
        |--data_loader.py
        |--embedding.py
        |--framework.py
        |--utils.py
    |--README.md
    |--train.py

**training model:**
>python train.py
    
**training parameter setting:**
> * --model: Decide training model, which could be:
> +     base: the baseline model
>       ia: the instance attention model
>       ia_pair: the instance attention model with pair function
>       iadm: the input-adaptive distance metric 
>       main: our main model
>       mann: the memory aggregated neural network
>       mlman: the multi-level matching and aggregation network
>       pr: baseline with prototype rectification 
>       pt: baseline with power transform function
>       relation: the relation network 
> * --encoder: Choose the sentence encoder in: 
> +     mascot: the MASCOT encoder
>       bran: the BRAN encoder
>       cnn: the CNN encoder
>       te: the Transformer encoder
> * --N_for_train \& --N_for_test: the training and evaluate N, which should be 20 \& 10 or 10 \& 5
> * --K, --Q, --na_rate: similar to conventional few-shot learning setting, usually would be 5, 1, and 1, where the na rate is (na_rate * Q) / (na_rate * Q + N)
> * --learning_rate: learning rate would be 1e-1 for most of model, but 5e-2 for training 10-way-K-shot models
> * --glove: default choose pre_trained glove, or could use
> +      --glove local --reprocess
> to use local glove file "glove.local.50d.json"
> * --load_model \& --save_model: the load or save model name

**Data Requirement:**   
all training and evaluate data could be found at [here](https://drive.google.com/drive/folders/17pwbulrE6HoUBHnEjrIihC-Om-0z3YFB?usp=sharing)
or you can download data from [FewRel github](https://github.com/thunlp/FewRel) and download standard glove mat from [here](https://nlp.stanford.edu/projects/glove/)
