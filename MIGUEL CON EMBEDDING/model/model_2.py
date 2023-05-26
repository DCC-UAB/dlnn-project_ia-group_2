import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models



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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3, weight_matrix=None, pretrained=True):
        super(DecoderRNN, self).__init__()
        if weight_matrix is not None:
            self.embedding, vocab_size, embed_size = self.create_embedding_layer(weight_matrix, pretrained)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(drop_prob)
        self.vocab_size = vocab_size

    def forward(self, features, captions, use_teacher_forcing = True):
        embeds = self.embedding(captions[:, :-1])
        x = torch.cat((features.unsqueeze(1), embeds), dim=1)
        x, _ = self.lstm(x)
        x = self.fcn(x)
        
        if use_teacher_forcing:
            return x
        else:
            return x.argmax(dim=2)

  

    def create_embedding_layer(self, weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim

    def generate_caption(self, features, max_len, vocab, use_teacher_forcing=False):
                # Given the image features generate the caption
        batch_size = inputs.size(0)
        captions = []
        for i in range(max_len):
            output,hidden = self.lstm(inputs,hidden)
            output = self.fcn(output)
            output = output.view(batch_size,-1)
            
            if use_teacher_forcing:
                predicted_word_idx = output.argmax(dim=1)
            else:
                predicted_word_idx = output.argmax(dim=1)
                inputs = self.embedding(predicted_word_idx.unsqueeze(1))
        
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

    
class EncoderDecoder_2(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,num_layers=1,drop_prob=0.3, weight_matrix=None):
        super(EncoderDecoder_2, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size,hidden_size,vocab_size,num_layers,drop_prob, weight_matrix=None)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs