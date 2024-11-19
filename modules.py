import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import math

from config import Config


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnet = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        self.model = nn.Sequential(*list(efficientnet.children())[:-2])

        # setting all the parameters not requires_grad for not computing and applying gradients.
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self,image):
        """
        Forward pass of the module.
        """
        image_features:torch.Tensor = self.model(image)
        # (batch,1280,8,8) -> (batch,64,1280)
        image_features = image_features.reshape(image_features.shape[0],image_features.shape[1],(image_features.shape[2]*image_features.shape[3])).permute(0,2,1)
        return image_features
    
class Encoder(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.d_model,nhead=config.n_heads_encoder,batch_first=True),1
        )
        self.ff_1 = nn.Linear(1280,2048)
        self.ff_2 = nn.Linear(2048,config.d_model)
        self.layer_norm_1 = nn.LayerNorm(1280)
        self.layer_norm_2 = nn.LayerNorm(config.d_model)

    def forward(self,image_features:torch.Tensor):
        image_features = self.layer_norm_1(image_features)
        proj = F.relu(self.ff_1(image_features))
        proj = F.relu(self.ff_2(proj))
        return self.encoder(proj)

class PositionalEncoding(nn.Module):
    def __init__(self,config:Config):

        super(PositionalEncoding, self).__init__()
        self.embed_dim = config.d_model

        # Create the positional encodings matrix
        position = torch.arange(0, config.maxlen).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, config.d_model, 2) * (-math.log(10000.0) / config.d_model))

        # Apply sine to even indices and cosine to odd indices
        pe = torch.zeros(config.maxlen, config.d_model)  # [max_len, embed_dim]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch-first dimension
        self.register_buffer('positional_encodings', pe.unsqueeze(0))  # [1, max_len, embed_dim]

    def forward(self, x):

        seq_len = x.size(1)  # Extract sequence length from input
        x = x + self.positional_encodings[:, :seq_len, :]
        return x
    
class Decoder(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.d_model)
        self.pos_emb = PositionalEncoding(config)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=config.d_model,nhead=config.n_heads_decoder,batch_first=True),2
        )
        self.out = nn.Linear(config.d_model,config.vocab_size)

    def forward(self,x,encoder_outputs,tgt_attention_mask=None,tgt_key_padding_mask= None):
        x = self.pos_emb(self.embedding(x))
        decoder_out = self.decoder(
            tgt=x,
            memory=encoder_outputs,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_attention_mask
        )
        preds = self.out(decoder_out)
        return preds
    
