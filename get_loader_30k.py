import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Class to generate the vocabulary for our LSTM
class Vocabulary:
    def __init__(self, freq_threshold: int):
        # Defining some needed tokens for Resnet model
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        # Reversed ITOS dict
        self.stoi = {k:v for v,k in self.itos.items() }
        # Frequency threshold indicator, leading to ignore
        self.freq_threshold = freq_threshold  
        
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def vocab_tokenizer(text: str):
        return [token.lower() for token in word_tokenize(text)]
    
    def build_vocabulary(self, captions):
        frequencies = {}
        # Start idx 4 because of previous itos tokens
        idx = 4
        for caption in captions:
            for word in self.vocab_tokenizer(caption):
                # print(word)
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1  
                    
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1      
                    
    def to_one_hot(self, text: str):
        tokenized_text = self.vocab_tokenizer(text)
        return [self.stoi[word] if word in self.stoi else self.stoi['<UNK>'] for word in tokenized_text]

      
# Class for our dataloader to access
class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir: str, dataframe, transform=None, freq_threshold: int=3):
        # Data path
        self.data_dir = data_dir
        
        # Image captions dataframe
        self.df = dataframe
            
        # Transform value
        self.transform = transform
        
        # Images and captions from DF
        self.images = self.df['image']
        self.captions = self.df['caption'].astype(str)
        
        # Build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())
        self.freq_threshold = freq_threshold
        self.max_caption_length = self.get_max_caption_length()  
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx: int):
        caption = self.captions.iloc[idx]
        img_dir = self.images.iloc[idx]
        img = Image.open(os.path.join(self.data_dir, img_dir)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        one_hot_caption = [self.vocab.stoi['<SOS>']]
        one_hot_caption.extend(self.vocab.to_one_hot(caption))
        one_hot_caption.append(self.vocab.stoi['<EOS>'])
        padded_vector = self.padded_caption(one_hot_caption)
        return img, torch.tensor(padded_vector), img_dir
    
    def get_max_caption_length(self):
        max_length = 0
        for caption in self.captions:
            tokenized_caption = self.vocab.vocab_tokenizer(caption)
            max_length = max(max_length, len(tokenized_caption))  
        return max_length 
    
    def padded_caption(self, caption):
        padded_caption = caption[:self.max_caption_length]
        padded_caption += [self.vocab.stoi["<PAD>"]] * (self.max_caption_length - len(padded_caption))
        return padded_caption
    
def get_loader(data_dir, dataframe, transform=None, batch_size=None, num_workers=1, shuffle=True, pin_memory=True):
    dataset = ImageCaptionDataset(data_dir=data_dir, dataframe=dataframe, transform=transform)
    pad_idx = dataset.vocab.stoi['<PAD>']
    data_loader  = DataLoader(dataset=dataset, batch_size=batch_size,
                         num_workers=num_workers, shuffle=shuffle,
                         pin_memory=pin_memory, drop_last=True) 
    return data_loader 
 
def get_vocab(data_dir, dataframe, transform=None):
    dataset = ImageCaptionDataset(data_dir=data_dir, dataframe=dataframe, transform=transform)
    vocab = dataset.vocab
    return vocab.itos

def show_image(tensor, title=None):
    """Imshow for Tensor"""
    tensor = tensor.cpu().numpy().transpose((1,2,0))
    plt.imshow(tensor)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def get_pad_index(data_dir, dataframe, transform=None):
    dataset = ImageCaptionDataset(data_dir=data_dir, dataframe=dataframe, transform=transform)
    pad_idx = dataset.vocab.stoi['<PAD>']
    return pad_idx


