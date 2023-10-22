# Intent Classification Service

- [Intent Classification Service](#intent-classification-service)
  - [Challenge Task](#challenge-task)
  - [Approach](#approach)
    - [Data Preparation](#data-preparation)
    - [Data](#data)
    - [Modeling](#modeling)
- [Instructions](#instructions)
  - [Installation](#installation)
  - [Building or downloading docker image](#building-or-downloading-docker-image)
    - [Building docker image](#building-docker-image)
    - [Dowload pre-built docker image](#dowload-pre-built-docker-image)
- [API Documentation](#api-documentation)
- [Data Preparation and Model Training](#data-preparation-and-model-training)
- [Evaluation](#evaluation)
  - [Accuracy](#accuracy)
  - [AU-ROC](#au-roc)
    - [Overall](#overall)
    - [Per Intent Class](#per-intent-class)
  - [Classificaiton Report](#classificaiton-report)
- [Future Improvements](#future-improvements)


## Challenge Task

Your task is to implement a neural network-based intent classifier that can be used to provide inferencing service via an HTTP Service. The boiler plate for the Service is implemented in file `server.py` and you'll have to implement the API function for inferencing as per the API documentation provided below. The neural network interface has been defined in `intent_classifer.py`. You can add any methods and functionality to this class you deem necessary.

You may use any deep learning library (Tensorflow, Keras, PyTorch, ...) you wish and you can also use pre-existing components for building the network architecture if these would be useful in real-life production systems. Provide tooling and instructions for training the network from scratch.

Also provide a jupyter notebook for model development which trains and tests the model. The final output of this notebook should be trained models with respect to their tests. Evaluation metrics should include the following:

1. Accuracy
2. Precision
3. Recall
4. F1
5. Any other metric that you think is suitable for the comparison

In addition, the same notebook provides a section to evaluate models in production. Assuming the following scenario:

You have both of the models in production and no labeled data is available to you. How would you compare them? Which metrics would you use for this kind of comparison? For example, you can use metrics based on confidence values or related ones.

## Approach

In this notebook we are training a multilingual intent classification, for the purpose of POC I am selecting following languages:

1. English
2. Hindi
3. Spanish

### Data Preparation

The given ATIS dataset is provided in English, I have created a parallel dataset using google translation for Hindi and Spanish.

In the training dataset, we have the following distribution of data:

```
 flight                        3426
 airfare                        403
 ground_service                 235
 airline                        148
 abbreviation                   108
 aircraft                        78
 flight_time                     52
 quantity                        49
 distance                        20
 city                            18
 airport                         18
 ground_fare                     17
 flight+airfare                  17
 capacity                        16
 flight_no                       12
 meal                             6
 restriction                      5
 airline+flight_no                2
 ground_service+ground_fare       1
 airfare+flight_time              1
 cheapest                         1
 aircraft+flight+flight_no        1
```

From what I can observe, the data is quite unbalanced and also some of the classes seems to be a combination of others, which gives me an indication that we can perhaps model this as a multi-label classification problem. Also, it is possible that the user query may have multiple intents and a multi-label model is a good choice to handle such a scenario.

For this reason, I have transformed the created a multi-label dataset, with the following 17 classes:
```
'ground_service', 'abbreviation', 'ground_fare', 'airline', 'city',
'aircraft', 'flight_no', 'airport', 'flight', 'quantity', 'meal',
'capacity', 'restriction', 'airfare', 'distance', 'flight_time', 'cheapest'
```

### Data
- `data` - The provided ATIS dataset
  - downlaod link: `https://drive.google.com/drive/folders/1I2cALZXOIaz9WnmdtubpVavflkUvFRHG?usp=share_link`
- `data_mlabel` - The multilabel version of provided ATIS dataset
  - download link: `https://drive.google.com/drive/folders/1-0VzuUa16j3nEcqywHVKlJONMPg0F40S?usp=share_link`
- `multilingual_data` - Parallel Multilingual translated data of provdied ATIS dataset
  - download link - `https://drive.google.com/drive/folders/1A-t73esVP27KbC9eAEMBv6klu8jpQSd-?usp=share_link`

### Modeling

The model architecture is a simple one, which I believe is a strong baseline for the task and can be used for handling multi-lingual queries.

1. Encoder

    As I am tring to train a multilingual model, the first step in the NLP pipeline would be to have a [`bert-base-multilingual-cased`](https://huggingface.co/bert-base-multilingual-cased) from huggingface-transformers. This BERT varient is a pretrained model on the top 104 languages with the largest Wikipedia using a masked language modeling (MLM) objective. We can easily extend our approach to handle queries in 104 languages, however the performance might differ between languages depending on the amount of data used in multi-lingual BERT pretraining.

2. Decoder (classifier)

    The decoder is a single linear layer mapping the encoder output to our 17-output classes.

3. Loss Function

    Binary Cross Entropy is a suitable loss function for multi-label modeling in this scenario.

# Instructions

## Installation

* Create a conda environment using:
    - `conda create -n "intent-clf-env" python=3.10.11`
* Install dependencies
    - `pip install -r requirements.txt`
* In the repository you will find `.env.bkp` file. You need to create a copy of the file:
    - `cp .env.bkp .env`
    - Setup the environment variables:
        - ```
            PORT=8080
            CHECKPOINT_PATH="./model/best-checkpoint-v1.ckpt"
            ML_BINARIZER_PATH="./model/ml_binarizer.pkl"
          ```
* Download the model files in the `models` directory:
    - `best-checkpoint-v1.ckpt` - `https://drive.google.com/file/d/1-P-mIf9ChF04LzZ63EZiitUiGthVfYzm/view?usp=share_link`
    - `ml_binarizer.pkl` - `https://drive.google.com/file/d/1-Q5671xZmR54XSChXq9yv1yFN41-tZVJ/view?usp=share_link`
* Running the Flask server
    - `python server.py`

## Building or downloading docker image

### Building docker image
* The code has a `Dockerfile` pre-setup
* In the repository you will find `.env.bkp` file. You need to create a copy of the file:
    - `cp .env.bkp .env`
    - Setup the environment variables:
        - ```
            PORT=8080
            CHECKPOINT_PATH="./model/best-checkpoint-v1.ckpt"
            ML_BINARIZER_PATH="./model/ml_binarizer.pkl"
          ```
* Download the model files in the `models` directory:
    - `best-checkpoint-v1.ckpt` - `https://drive.google.com/file/d/1-P-mIf9ChF04LzZ63EZiitUiGthVfYzm/view?usp=share_link`
    - `ml_binarizer.pkl` - `https://drive.google.com/file/d/1-Q5671xZmR54XSChXq9yv1yFN41-tZVJ/view?usp=share_link`
* Build docker images:
    - `sudo docker build . -t lordzuko/intent-clf-service:v1.0.0`
* Running docker container
    - `sudo docker run -p 8080:8080 lordzuko/intent-clf-service:v1.0.0`

### Dowload pre-built docker image
* `docker pull lordzuko/intent-clf-service:v1.0.0`

# API Documentation
The documentation provides how to use the API, with `python`, `curl` etc.
* [POSTMAN documentation](https://documenter.getpostman.com/view/30635450/2s9YRB4CSg#6401b2aa-c4d3-4881-9ba6-7182af00ef43) for the service can be found here.

# Data Preparation and Model Training
- The process for model training and evaluation is described in notebook: `notebooks/multi_lingual_multilabel_intent_clf.ipynb`
- The process for model evalation during production scenario is described in noteboo: `notebooks/Production_Evaluation.ipynb` 
# Evaluation


## Accuracy
```
Best Threshold: 0.30
Train Accuracy: 0.999
Val Accuracy: 0.998
Test Accuracy: 0.995
```
## AU-ROC
### Overall
```
Best Threshold: 0.20
Train AUROC: 0.874
Val AUROC: 0.833
Test AUROC: 0.805
```
### Per Intent Class

You can format the validation and test results for markdown as follows:

| Label            | Validation AUROC | Test AUROC     |
|------------------|------------------|-----------------|
| abbreviation     | 1.000000         | 0.999959        |
| aircraft         | 0.998834         | 0.997295        |
| airfare          | 0.999978         | 0.989687        |
| airline          | 0.999845         | 0.992285        |
| airport          | 1.000000         | 0.999959        |
| capacity         | 1.000000         | 0.999260        |
| cheapest         | 0.000000         | 0.000000        |
| city             | 1.000000         | 0.936331        |
| distance         | 1.000000         | 1.000000        |
| flight           | 0.999852         | 0.989004        |
| flight_no        | 0.992795         | 1.000000        |
| flight_time      | 0.992006         | 1.000000        |
| ground_fare      | 1.000000         | 0.997722        |
| ground_service   | 0.998485         | 0.999989        |
| meal             | 0.992806         | 0.926189        |
| quantity         | 0.999711         | 0.996064        |
| restriction      | 0.000000         | 0.000000        |
|


## Classificaiton Report

| Label         | Train Support | Validation Support | Test Support | Validation Precision | Test Precision | Validation Recall | Test Recall | Validation F1-Score | Test F1-Score |
|---------------|---------------|--------------------|--------------|-----------------------|-----------------|------------------|------------|-------------------|---------------|
| abbreviation  | 309.0         | 15.0               | 78.0         | 1.00                | 0.97           | 1.00             | 0.99       | 1.00              | 0.98          |
| aircraft      | 227.0         | 10.0               | 24.0         | 0.83                | 0.88           | 1.00             | 0.88       | 0.91              | 0.88          |
| airfare       | 1190.0        | 73.0               | 183.0        | 0.97                | 0.97           | 1.00             | 0.93       | 0.99              | 0.95          |
| airline       | 421.0         | 29.0               | 90.0         | 1.00                | 0.96           | 0.97             | 0.96       | 0.98              | 0.96          |
| airport       | 53.0          | 1.0                | 39.0         | 1.00                | 0.95           | 1.00             | 0.97       | 1.00              | 0.96          |
| capacity      | 46.0          | 2.0                | 63.0         | 1.00                | 1.00           | 1.00             | 0.94       | 1.00              | 0.97          |
| cheapest      | 3.0           | 0.0                | 0.0          | 0.00                | 0.00           | 0.00             | 0.00       | 0.00              | 0.00          |
| city          | 51.0          | 3.0                | 15.0         | 1.00                | 1.00           | 1.00             | 0.40       | 1.00              | 0.57          |
| distance      | 58.0          | 2.0                | 30.0         | 1.00                | 0.39           | 1.00             | 1.00       | 1.00              | 0.57          |
| flight        | 9822.0        | 510.0              | 1881.0       | 1.00                | 0.99           | 0.99             | 0.98       | 1.00              | 0.99          |
| flight_no     | 43.0          | 2.0                | 27.0         | 1.00                | 1.00           | 0.50             | 1.00       | 0.67              | 1.00          |
| flight_time   | 151.0         | 8.0                | 3.0          | 0.78                | 0.27           | 0.88             | 1.00       | 0.82              | 0.43          |
| ground_fare   | 51.0          | 3.0                | 21.0         | 1.00                | 0.29           | 1.00             | 0.95       | 1.00              | 0.44          |
| ground_service| 672.0         | 36.0               | 108.0        | 0.97                | 0.92           | 1.00             | 1.00       | 0.99              | 0.96          |
| meal          | 17.0          | 1.0                | 18.0         | 0.00                | 0.00           | 0.00             | 0.00       | 0.00              | 0.00          |
| quantity      | 142.0         | 5.0                | 9.0          | 0.83                | 0.21           | 1.00             | 1.00       | 0.91              | 0.35          |
| restriction   | 15.0          | 0.0                | 0.0          | 0.00                | 0.00           | 0.00             | 0.00       | 0.00              | 0.00          |
|

| Summary         | Train Support | Validation Support | Test Support | Validation Precision | Test Precision | Validation Recall | Test Recall | Validation F1-Score | Test F1-Score |
|---------------|---------------|--------------------|--------------|-----------------------|-----------------|------------------|------------|-------------------|---------------|
| micro avg     | 13271.0       | 700.0              | 2589.0       | 0.99                | 0.93           | 0.99             | 0.97       | 0.99              | 0.95          |
| macro avg     | 13271.0       | 700.0              | 2589.0       | 0.79                | 0.64           | 0.78             | 0.76       | 0.78              | 0.65          |
| weighted avg  | 13271.0       | 700.0              | 2589.0       | 0.99                | 0.96           | 0.99             | 0.97       | 0.99              | 0.96          |
| samples avg   | 13271.0       | 700.0              | 2589.0       | 0.99                | 0.95           | 0.99             | 0.97       | 0.99              | 0.96          |
|



# Future Improvements
1. The model is currently being loaded from checkpoint, which has optimizer and gradient placeholders, which is increasing the file size. A better way would be to save model dict and load the model from state dict. This will reduce the model file size, which will ultimately decrease the size of resulting docker image.
2. Evaluation is not done at language level, which can be important to make any language specific improvements and updates.
3. The `/intent` api endpoint does not provide an `api_version` field which could be important to keep track of the current version of intent classification service which is producing the output. The downstream applications might take advantage of it to handle business logic or update them according to api_version.
   1. It can also be important in AB-testing
   2. Also, data management could take advantage of this field
4. The `/intent` api does not check for the language. This could be problametic as our model currently only supports for 3 languages, whereas the tokenizer we are using can support 104 langauges. Not detecting the supported languages could lead to unforseen model performance issues.
5. Data imbalance among intent classes is currently a bottlenect and upsampling needs to be down for low data classes. We can use translation and paraphrasing to tackle the data imbalance issues. 