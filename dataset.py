from torch.utils.data import Dataset
import torchvision
import torch

from config import Config
from PIL import Image


class CaptionDataset(Dataset):
    """
    Captioning Dataset for Flickr8k data.
    """
    def __init__(self,caption_data:dict,config:Config,word2idx:dict):
        """
        Inializes classes construction method.

        Arguments:
            caption_data (dict): Caption data prepared and processed before.
            config: Configuration class.
            word2idx (dict):Word to index dictionary.

        Returns:
            None.

        """
        self.caption_data = caption_data
        self.config = config

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((config.image_size,config.image_size)),
                torchvision.transforms.ToTensor()
            ]
        )
        self.images = list(caption_data.keys())
        self.captions = list(caption_data.values())
        self.word2idx = word2idx

    # loads and prepares images using torchvision.transforms
    def load_and_prepare_images(self,image_path):
        """
        Loads and prepares given image path.

        Arguments:
            image_path (str): Path of the image.

        Returns:
            image (torch.Tensor): Preprocessed tensor type of the image.
        """
        image = Image.open(image_path)
        image = self.transforms(image)

        return image

    # loads and prepares captions and returns list of preprocessed captions.
    def load_and_prepare_captions(self,captions:list):
        """
        Loads and prepares given list of captions.

        Arguments:
            captions (list): List of captions. Contains 5 elements per image pair.

        Returns:
            preprocessed_captions (list): List of preprocessed captions that contains: input_tokens,target_tokens,tgt_padding_mask respectively.
        """
        preprocessed_captions = []
        for i in range(5):
            caption = captions[i]
            tokens = caption.split()
            tokens = [token.strip().lower() for token in tokens]

            if len(tokens) > self.config.maxlen:
                tokens = tokens[:self.config.maxlen]

            input_tokens = tokens[:-1].copy()
            target_tokens = tokens[1:].copy()
            sample_size = len(input_tokens)
            padding_size = self.config.maxlen - sample_size

            if padding_size > 0:
                padding_vec = ["<pad>" for _ in range(padding_size)]
                input_tokens += padding_vec.copy()
                target_tokens += padding_vec.copy()

            input_tokens = [self.word2idx.get(token) for token in input_tokens]
            target_tokens = [self.word2idx.get(token) for token in target_tokens]


            input_tokens = torch.Tensor(input_tokens).long()
            target_tokens = torch.Tensor(target_tokens).long()

            tgt_padding_mask = torch.ones([self.config.maxlen, ])
            tgt_padding_mask[:sample_size] = 0.0
            tgt_padding_mask = tgt_padding_mask.bool()

            preprocessed_captions.append([input_tokens[:self.config.maxlen],target_tokens[:self.config.maxlen],tgt_padding_mask])

        return preprocessed_captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Gets specified data.
        """
        image = self.load_and_prepare_images(self.images[index])
        y = self.load_and_prepare_captions(self.captions[index])
        return image,y