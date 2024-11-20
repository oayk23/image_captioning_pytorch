# Image captioning with Transformers.

This repo contains image captioning with *PyTorch*.

---

## Motivation

My undergraduate thesis is about video captioning and for this i should understand with basics of the **image captioning**.That's why i started with image captioning with transformers.

## What does this application actually do ?

You can train a deep neural network for image-captioning with this repo and could be used for inference with **inference.py** file.

## Why Transformers

Because this technology is too powerful when you have enough data and seperates from **RNN**'s and **LSTM**'s with it's important part of *Attention* mechanism. It's purpose is catching relationship between past tokens with current one.

## Limitations and Challenges

Its hard to train a Transformer model because attention is computationally expensive and time consuming. Experiments ran with `RTX 2070` GPU and number of encoder and decoder layers in the transformer is 1,2 respectively. For keep this repo easily runnable i used minimal amount of layers. Data is `Flickr8k` and dataset contains limited data and that's a limitation too.

---

## Data,Model and Resources

Data used: [here](https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip "Flickr8k Dataset")

Resources: 

          * CPU: Intel i7 10750H
          
          * GPU: RTX 2070

          * RAM: 16 GB

Model: 

       EfficientNetB0: for feature extraction

       TransformerEncoder: for encoder
       
       TransformerDecoder: for decoder

---

## How to use

1. Get dataset in [here](https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip "Flickr8k Dataset")
2. Modify Config file in `config.py` for set the proper arguments in functions.
3. Train the model using `train.py`. This script will create 3 files: `encoder.pth` file for encoders **state_dict**,`decoder.pth` file for decoders **statedict**, `word2idx.json` file that acts like a tokenizer.
4. Get inference with running `inference.py` script. Note that you have to specify the image path,image save path in the script.

---

## Results

Some of the images correctly describes the image but sometimes model fails to generate proper caption. You can find more on the examples folder.

Here is the one of examples:

![alt text](https://github.com/oayk23/image_captioning_pytorch/blob/main/examples/example.png)

## Acknowledgements

Thanks to [senadkurtisi](https://github.com/senadkurtisi) for his [repo](https://github.com/senadkurtisi/pytorch-image-captioning). I modified some of his codes for my usage.

Also thanks keras team for their inspiring [work](https://keras.io/examples/vision/image_captioning/)






