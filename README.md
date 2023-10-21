# Intent Classification Service

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
    - Download link: [TODO] - add link here
* Running the Flask server
    - `python server.py`

## Docker build
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
    - Download link: [TODO] - add link here
* Build docker images:
    - `sudo docker build . -t lordzuko/intent-clf-service:v1.0.0`
* Running docker container
    - `sudo docker run -p 8080:8080 lordzuko/intent-clf-service:v1.0.0`

# API Documentation
The documentation provides how to use the API, with `python`, `curl` etc.
* [POSTMAN documentation](https://documenter.getpostman.com/view/30635450/2s9YRB4CSg#6401b2aa-c4d3-4881-9ba6-7182af00ef43) for the service can be found here.
