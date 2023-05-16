import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
 def __init__(self, embed_size):
   super(CNNEncoder, self).__init__()
   self.inception = models.inception_v3(pretrained=True,aux_logits=False)
   self.inception.fc = nn.Linear(self.inception.fc.in_features,embed_size)
   self.relu = nn.ReLU()
   self.dropout = nn.Dropout(0.5)

 def forward(self, input):
   features = self.inception(input)
   return self.dropout(self.relu(features))

class DecoderRNN(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size):
    super(DecoderRNN, self).__init__()
    self.embed = nn.Embedding(vocab_size, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_size)
    self.linear = nn.Linear(hidden_size, vocab_size)
    self.dropout = nn.Dropout(0.5)

  def forward(self, features, captions):
    embeddings = self.dropout(self.embed(captions))
    embeddings = torch.cat((features.unsqueeze(0), embeddings), 
                                                        dim=0)
    hiddens, _ = self.lstm(embeddings)
    outputs = self.linear(hiddens)
    return outputs
  
class Encoder_Decoder(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size):
    super(Encoder_Decoder, self).__init__()
    self.cnn = CNNEncoder(embed_size)
    self.decoderRNN = DecoderRNN(embed_size, hidden_size,
                                 vocab_size)
  def forward(self, images, captions):
    features = self.cnn(images)
    outputs = self.decoderRNN(features, captions)
    return outputs
  
#Pseudocode de como realizar attention, se tendria q poner en el decoder.
self.attention = nn.Linear(embed_size, embed_size)
self.softmax = nn.Softmax()
attention_output = self.softmax(self.attention(feature_embedding))
feature_embedding = feature_embedding * attention_output