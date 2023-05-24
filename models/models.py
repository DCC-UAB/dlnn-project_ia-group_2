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

        if self.training and teacher_forcing_prob > 0.0: # WHen in training mode will enter this if .
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_prob # Probability to use teacher forcing or not.
            if use_teacher_forcing:
                x = x[:, :-1, :]  # Use ground truth captions from the second to the last time step

        return x


 # bajar la temperature para resultados menos random, subirla para mas randomness.
def generate_caption(self, inputs, hidden=None, max_len=25, vocab=None, beam_width=5, temperature=1.0):
    batch_size = inputs.size(0)

    captions = []
    partial_captions = [{'sequence': [vocab['<START>']], 'hidden': hidden, 'score': 0.0}]

    for _ in range(max_len):
        candidates = []
        for partial_caption in partial_captions:
            sequence = partial_caption['sequence']
            hidden = partial_caption['hidden']
            inputs = self.embedding(torch.tensor(sequence[-1]).unsqueeze(0))

            output, hidden = self.lstm(inputs, hidden)
            output = self.fcn(output)
            output = output.view(batch_size, -1)

            probabilities = F.softmax(output / temperature, dim=1)
            top_probs, top_words = torch.topk(probabilities, beam_width)

            for i in range(beam_width):
                word_idx = top_words[0, i].item()
                word_prob = top_probs[0, i].item()
                score = partial_caption['score'] + torch.log(word_prob)

                if vocab[word_idx] == '<EOS>':
                    captions.append({'sequence': sequence + [word_idx], 'score': score})
                else:
                    candidates.append({'sequence': sequence + [word_idx], 'hidden': hidden, 'score': score})

        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:beam_width]
        partial_captions = candidates

        if len(partial_captions) == 0:
            break

    captions.extend(partial_captions)
    best_caption = max(captions, key=lambda x: x['score'])
    generated_caption = [vocab[idx] for idx in best_caption['sequence']]
    generated_caption = generated_caption[1:-1]  # Remove <START> and <EOS> tokens

    return generated_caption
  

class EncoderDecoder(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers=1,drop_prob=0.3):
        super(EncoderDecoder,self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size,hidden_size,vocab_size,num_layers,drop_prob)
    
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

'''
# Conventional and convolutional neural network

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
'''