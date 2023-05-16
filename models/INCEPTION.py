import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
 def __init__(self, embed_size): # embed_size is the embedding size we want.
   super(CNNEncoder, self).__init__()
   #We are taking the last fully connected layer of the Inception network.
   self.inception = models.inception_v3(pretrained=True,aux_logits=False) 
   #Manually changing last Inception layer to map/connect to the embedding size we want.
   self.inception.fc = nn.Linear(self.inception.fc.in_features,embed_size)
   self.relu = nn.ReLU()
   self.dropout = nn.Dropout(0.5)

 def forward(self, input):
   features = self.inception(input)
   return self.dropout(self.relu(features))

