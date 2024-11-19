from utils import greedy_decode
from modules import FeatureExtractor,Encoder,Decoder
from config import Config

from PIL import Image
import json
import matplotlib.pyplot as plt
import torchvision
import torch

def infer(config,image_path,save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open("word2idx.json","r") as json_file:
        word2idx = json.load(json_file)
    config.vocab_size = len(word2idx)
    encoder = Encoder(config).to(device)
    feat_ext = FeatureExtractor().to(device)
    decoder = Decoder(config).to(device)
    decoder.load_state_dict(torch.load("decoder.pth"))
    encoder.load_state_dict(torch.load("encoder.pth"))

    idx2word = {v:k for k,v in word2idx.items()}
    image = Image.open(image_path)
    plt.imshow(image)
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.ToTensor()
        ]
    )
    image = transforms(image)
    caption = greedy_decode(image=image,config=config,word2idx=word2idx,decoder=decoder,device=device,encoder=encoder,feature_extractor=feat_ext,idx2word=idx2word)
    plt.title(caption)
    plt.savefig(save_path)
    plt.show()
    print(caption)

if __name__ == "__main__":
    config = Config()
    image_path = r"dataset\Flicker8k_Dataset\133189853_811de6ab2a.jpg"
    save_path = "example4.png"

    infer(config,image_path,save_path)