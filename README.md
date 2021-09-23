## Few-shot relation extraction on ancient Chinese documents

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

**running environment:**
>
>GPU: RTX 2080Ti
>
>CUDA: 11.4 / cudnn: 8.2.2
>
>PyTorch: 1.9.0

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
>       mlman: special empty encoder for mlman model
> * --N_for_train \& --N_for_test: the training and evaluate N, which should be 20 \& 10 or 10 \& 5, and for acd dataset, N should not bigger that class number(7)
> * --K, --Q, --na_rate: similar to conventional few-shot learning setting, usually would be 5, 1, and 1, where the na rate is (na_rate * Q) / (na_rate * Q + N)
> * --batch: batch size, 1, 2, 4, 8... bigger size usually leads to a better performance, for unified comparison and efficiency, we use 1 as base parameter
> * --learning_rate: learning rate would be 1e-1 for most of model, but 5e-2 for training 10-way-K-shot models
> * --glove: default choose pre_trained glove, or could use
> +      --glove local --reprocess
> +      --glove acd --reprocess
> to use local / acd glove file "glove.local.50d.json" or "glove.acd.50d.json"
> * --load_model \& --save_model: the load or save model name
> * --train_file \& --test_file: the training and test file name, such as train_acd, test_acd

**Data Requirement:**   
all training and evaluate data could be found at [here](https://drive.google.com/drive/folders/17pwbulrE6HoUBHnEjrIihC-Om-0z3YFB?usp=sharing)
or you can download data from [FewRel github](https://github.com/thunlp/FewRel) and download standard glove mat from [here](https://nlp.stanford.edu/projects/glove/)
