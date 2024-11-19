class Config:
    maxlen = 27 # max sequence length
    image_size = 256 # image size for resizing
    batch_size = 128 # batch size per dataloader
    epochs = 6 # num epochs
    learning_rate = 1e-5 # learning rate for optimizer
    d_model = 512   # model's dimension
    n_heads_encoder = 1 # encoders num_heads for multihead attention
    n_heads_decoder = 2 # decoders num_heads for multihead attention
    path_to_txt = r"dataset\Flickr8k.token.txt" # path to data
    images_path = r"dataset\Flicker8k_Dataset"  # path to images file for preprocessing step.
    vocab_size = None # this will be changed later
