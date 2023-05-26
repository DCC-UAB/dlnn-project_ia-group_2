import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self,embed_size):
        super(EncoderCNN,self).__init__()
        resnet = models.resnet50(pretrained=True) 
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1] # To extract the features of Rsenet from the last layer before the Softmax is applied
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features,embed_size) 
        
    def forward(self,images):
        features = self.resnet(images) # resenet features shape - torch.Size([4, 2048, 1, 1])
        features = features.view(features.size(0),-1)  # resenet features viewed shape - torch.Size([4, 2048])
        features = self.embed(features) # resenet features embed shape - torch.Size([4, 400]
        
        return features

class DecoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers=1,drop_prob=0.3):
            super(DecoderRNN,self).__init__()
            self.embedding = nn.Embedding(vocab_size,embed_size)
            self.lstm = nn.LSTM(embed_size,hidden_size,num_layers=num_layers,batch_first=True)
            self.batch_norm = nn.BatchNorm1d(hidden_size)  # Add batch normalization layer
            self.fcn = nn.Linear(hidden_size,vocab_size)
            self.drop = nn.Dropout(drop_prob)
        
    def forward(self, features, captions, teacher_forcing_prob=0.5):
        # vectorize the caption
        # caption shape - torch.Size([4, 14])
        embeds = self.embedding(captions[:,:-1]) # shape of embeds - torch.Size([4, 14, 400])
        # features shape - torch.Size([4, 400])
        x = torch.cat((features.unsqueeze(1),embeds),dim=1) # features unsqueeze at index 1 shape - torch.Size([4, 1, 400])
        # shape of x - torch.Size([4, 15, 400])
        x,_ = self.lstm(x)
        # shape of x after lstm - torch.Size([4, 15, 512])
        x = self.fcn(x)

        if self.training and teacher_forcing_prob > 0.0:
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_prob
            if use_teacher_forcing:
                 x = x[:, :-1, :] # Exclude the last predicted step, , so ground truth is used from the second to the last time, ignoring predicted step.

        return x

    def generate_caption(self,inputs,hidden=None,max_len=25,vocab=None):

        # Given the image features generate the caption
        batch_size = inputs.size(0)
        captions = []
        for i in range(max_len):
            output,hidden = self.lstm(inputs,hidden)
            output = self.fcn(output)
            output = output.view(batch_size,-1)
        
            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)
            
            #save the generated word
            captions.append(predicted_word_idx.item())
            
            #end if <EOS detected>
            if vocab[predicted_word_idx.item()] == "<EOS>":
                break
            
            # Embed the predicted word to the next time step
            inputs = self.embedding(predicted_word_idx.unsqueeze(1))
        
            #convert the vocab idx to words and return generated sentence
        return [vocab[idx] for idx in captions]  
  
class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, drop_prob)

    def forward(self, images, captions, teacher_forcing_prob=0.5):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, teacher_forcing_prob)
        return outputs