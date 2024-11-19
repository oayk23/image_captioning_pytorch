from modules import FeatureExtractor,Encoder,Decoder
from dataset import CaptionDataset
from config import Config
from utils import calculate_accuracy,load_captions_data,save_models,\
    set_up_causal_mask,train_val_split,greedy_decode

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import json


def train(config:Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    captions_data,word2idx = load_captions_data(config)
    config.vocab_size = len(word2idx)
    with open("word2idx.json","w") as json_file:
        json.dump(word2idx,json_file)
    train_data,valid_data = train_val_split(captions_data,train_size=0.9)
    train_dataset = CaptionDataset(train_data,config,word2idx)
    valid_dataset = CaptionDataset(valid_data,config,word2idx)
    train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=config.batch_size,shuffle=True)

    feature_extractor = FeatureExtractor().to(device)
    encoder = Encoder(config).to(device)
    decoder = Decoder(config).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    mask = set_up_causal_mask(config.maxlen,device)
    train_loss = []
    valid_loss = []
    training_accuracy = []
    validation_accuracy = []

    for _ in tqdm(range(config.epochs)):
        feature_extractor.eval()
        encoder.train()
        decoder.train()
        print("Training step is starting...")
        for images,ys in train_dataloader:
            images = images.to(device)
            image_features = feature_extractor(images)

            for y in ys:
                optimizer.zero_grad()
                input_tokens = y[0].to(device)
                tgt_tokens = y[1].to(device)
                pad = y[2].to(device)
                encoder_outputs = encoder(image_features)
                preds = decoder(x=input_tokens,encoder_outputs=encoder_outputs,tgt_key_padding_mask = pad,tgt_attention_mask=mask)
                pad = torch.logical_not(pad)
                preds = preds[pad]
                tgt_tokens = tgt_tokens[pad]
                loss = loss_fn(preds,tgt_tokens)
                loss.backward()
                optimizer.step()
                acc = calculate_accuracy(y_pred=preds.argmax(dim=-1),y_true=tgt_tokens)
                train_loss.append(round(loss.item(),2))
                training_accuracy.append(round(acc.item(),2))

        print("train_dataloader finished. Validation starting...")
        encoder.eval()
        decoder.eval()
        for images,ys in valid_dataloader:
            images = images.to(device)
            image_features = feature_extractor(images)
            with torch.no_grad():
                for y in ys:
                    input_tokens = y[0].to(device)
                    tgt_tokens = y[1].to(device)
                    pad = y[2].to(device)
                    encoder_outputs = encoder(image_features)
                    preds = decoder(x=input_tokens,encoder_outputs=encoder_outputs,tgt_key_padding_mask = pad,tgt_attention_mask=mask)
                    pad = torch.logical_not(pad)
                    preds = preds[pad]
                    tgt_tokens = tgt_tokens[pad]
                    loss = loss_fn(preds,tgt_tokens)
                    acc = calculate_accuracy(y_pred=preds.argmax(dim=-1),y_true=tgt_tokens)
                    valid_loss.append(round(loss.item(),2))
                    validation_accuracy.append(round(acc.item(),2))
        print("Validation step is finished.")
    print("Training Completed. Models are saving...")
    save_models(encoder,decoder)
    
    return [train_loss,training_accuracy,valid_loss,validation_accuracy]

if __name__ == "__main__":
    config = Config()
    plots = train(config)
    train_loss,train_acc,valid_loss,valid_acc = plots
    fig,(ax1,ax2) = plt.subplots(1,2)
    fig.suptitle("Metrics")
    ax1.plot(train_loss,label="train loss")
    ax1.plot(valid_loss,label="valid loss")
    ax1.set_title("train and validation losses.")
    ax2.plot(train_acc,label="train acc")
    ax2.plot(valid_acc,label="valid acc")
    ax2.set_title("train and validation accuracies.")
    plt.show()

    
