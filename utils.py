import os
import numpy as np
import torch
import json

from modules import Encoder,Decoder,FeatureExtractor
from config import Config


def load_captions_data(config:Config):
    """
    Loads and prepares captioning data.

    Arguments:
        config: Configuration file.

    Returns:
        caption_mapping: Dictionary that contains image paths paired with captions.
        word2idx: dictionary that contains words corresponding to their numerical representation.
    """
    with open(config.path_to_txt,"r") as file:

        data = file.readlines()
        caption_mapping = {}
        text_data = set()
        images_to_skip = set()
        word2idx = {"<pad>":0,"<start>":1,"<end>":2}
        for line in data:
            line = line.rstrip("\n")
            img_name, caption = line.split("\t")
            img_name = img_name.split("#")[0]
            img_name = os.path.join(config.images_path,img_name)
            tokens = caption.strip().split()
            for token in tokens:
                text_data.add(token.lower())

            if len(tokens) < 5 and len(tokens) > config.maxlen:
                images_to_skip.add(img_name)

            if img_name.endswith(".jpg") and img_name not in images_to_skip:
                caption = "<start> " + caption.strip() + " <end>"


                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]
        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]
        len_word2idx = len(word2idx)
        for i,token in enumerate(text_data):
            word2idx[token] = len_word2idx+i

        return caption_mapping,word2idx

def train_val_split(caption_data:dict, train_size = 0.8,shuffle = True):
    """
    Splits data to train and validation.

    Arguments:
        caption_data (dict): caption data that contains image-caption pairs.
        train_size (float): indicates size of the train. Must be fraction.
        shuffle (boolean): shuffles data before split.

    Returns:
        train_data (dict): splitted train data.
        val_data (dict): splitted validation data.
    """

    all_images = list(caption_data.keys())

    if shuffle:
        np.random.shuffle(all_images)

    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    return training_data,validation_data


def set_up_causal_mask(maxlen:int,device):
    """
    Sets up causal mask for attention mechanism.

    Arguments:
        maxlen (int): Sequence length of the model.
        device (torch.device): Device.(whether 'cuda' or 'cpu')

    Returns:
        mask (torch.Tensor): upper triungular casusal mask for let the model prevent the seeing the future tokens.
    """
    mask = torch.triu(torch.ones((maxlen,maxlen)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask==0,float("-inf")).masked_fill(mask==1,float(0.0)).to(device)
    mask.requires_grad = False
    return mask

def calculate_accuracy(y_pred:torch.Tensor,y_true:torch.Tensor):
    """
    Element-wise accuracy.
    """
    assert y_pred.shape == y_true.shape,"y_pred and y_true shape must be equal"
    return (y_pred == y_true).sum() / y_pred.numel()


def greedy_decode(config:Config,encoder:Encoder,decoder:Decoder,feature_extractor:FeatureExtractor,idx2word:dict,word2idx:dict,device,image:torch.Tensor):
    """
    Function that performs greedy decoding on the given image.

    Arguments:
        config: Configuration class
        encoder (Encoder): Encoder class.
        decoder (Decoder): Decoder class.
        feature_extractor (FeatureExtractor): FeatureExtractor class.
        idx2word (dict): text representations of the indexes.
        word2idx (dict): index representations of the texts.
        device (torch.device): Device. (whether 'cuda' or 'cpu')
        image (torch.Tensor): Tensor-based image.

    Returns:
        caption (str): Caption decoded by decoder.
    """

    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    image_features = feature_extractor(image)
    encoder_outputs = encoder(image_features)

    words = torch.Tensor([word2idx['<start>']] + [word2idx['<pad>']] * (config.maxlen-1)).to(device).long().unsqueeze(0)
    pad = torch.Tensor([True] * config.maxlen).to(device).bool().unsqueeze(0)
    generated_caption = []
    for i in range(config.maxlen -1):
        pad[:,i] = False
        y_pred_prob = decoder(x=words,encoder_outputs=encoder_outputs,tgt_key_padding_mask=pad)
        y_pred_prob = y_pred_prob[:,i].clone()
        y_pred = y_pred_prob.argmax(-1)
        generated_caption.append(idx2word[y_pred[0].item()])
        if y_pred[0] == word2idx['<end>']:
            break

        if i < (config.maxlen-1):
            words[:,i+1] = y_pred.view(-1)

    generated_caption.remove("<end>")

    caption = " ".join(generated_caption)

    return caption

def save_models(encoder:Encoder,decoder:Decoder):
    torch.save(encoder.state_dict(),"encoder.pth")
    torch.save(decoder.state_dict(),"decoder.pth")


if __name__ == "__main__":
    data,word2idx = load_captions_data(config=Config)
    with open("word2idx.json", "w") as json_file:
        json.dump(word2idx,json_file)