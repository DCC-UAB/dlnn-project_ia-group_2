import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN,self).__init__()
        
        self.resnet = models.resnet50(pretrained=True) 
        for name, param in self.resnet.named_parameters():
            if 'fc.bias' in name or 'fc.weights' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
       
    def forward(self,images):
        features = self.resnet(images) # resenet features shape - torch.Size([4, 2048, 1, 1])
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.5):
        super(DecoderRNN,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        # self.batch_norm = nn.BatchNorm1d(hidden_size)  # Add batch normalization layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, features, captions):
        # vectorize the caption
        # caption shape - torch.Size([4, 14])
        embeds = self.dropout(self.embedding(captions[:,:-1])) # shape of embeds - torch.Size([4, 14, 400])
        # features shape - torch.Size([4, 400])
        output = torch.cat((features.unsqueeze(1), embeds), dim=1) # features unsqueeze at index 1 shape - torch.Size([4, 1, 400])
        # shape of x - torch.Size([4, 15, 400])
        output,_ = self.lstm(output)

        # shape of x after lstm - torch.Size([4, 15, 512])
        output = self.linear(self.dropout(output))

        return output

    def generate_caption(self, inputs, hidden=None, max_len=25, vocab=None):
    # Given the image features generate the captions
        batch_size = inputs.size(0)
        captions = []
        
        for i in range(max_len):
            output, hidden = self.lstm(inputs, hidden)
            output = self.linear(output)
            output = output.view(batch_size, -1)
        
            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)
            
            #save the generated word
            captions.append(predicted_word_idx.item())
            
            #end if <EOS detected>
            if vocab[predicted_word_idx.item()] == "<EOS>":
                break
            
            #send generated word as the next caption
            inputs = self.embedding(predicted_word_idx.unsqueeze(0))
        
        #covert the vocab idx to words and return sentence
        return [vocab[idx] for idx in captions]


class EncoderDecoder(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers=1,drop_prob=0.3):
        super(EncoderDecoder,self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, drop_prob)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
# resenet features shape - torch.Size([4, 2048, 1, 1])
# resenet features viewed shape - torch.Size([4, 2048])
# resenet features embed shape - torch.Size([4, 400])
# caption shape - torch.Size([4, 14])
# shape of embeds - torch.Size([4, 14, 400])
# features shape - torch.Size([4, 400])
# features unsqueeze at index 1 shape - torch.Size([4, 1, 400])
# shape of x - torch.Size([4, 15, 400])
# shape of x after lstm - torch.Size([4, 15, 512])
# shape of x after fcn - torch.Size([4, 15, 2994])

# model = EncoderCNN(360)

# print(model)