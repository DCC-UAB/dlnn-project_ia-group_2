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

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        embeddings = self.drop(embeddings)

        # Concatenate features and embeddings
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(inputs)

        # Reshape and apply fully connected layer
        outputs = self.fcn(lstm_out[:, -1, :])

        return outputs

    def create_embedding_layer(self, weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim

    def generate_caption(self, features, max_len, vocab):
        batch_size = features.size(0)
        hidden = None
        captions = []

        # Generate captions one word at a time
        for _ in range(max_len):
            if len(captions) == 0:
                inputs = self.embedding(torch.zeros(batch_size, 1).long().to(features.device))
            else:
                inputs = self.embedding(torch.LongTensor(captions).unsqueeze(1).to(features.device))

            lstm_out, hidden = self.lstm(torch.cat((features.unsqueeze(1), inputs), dim=1), hidden)
            outputs = self.fcn(lstm_out.squeeze(1))

            # Get predicted word indices
            _, predicted_word_idx = outputs.max(dim=1)
            predicted_word_idx = predicted_word_idx.squeeze().tolist()

            # Save the generated word
            captions.append(predicted_word_idx)

            # End if <EOS> detected
            if all(idx == vocab["<EOS>"] for idx in predicted_word_idx):
                break

        # Convert the vocab indices to words and return the generated sentence
        generated_sentence = [vocab.idx2word[idx] for idx in captions]
        return generated_sentence
    
class EncoderDecoder_2(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,num_layers=1,drop_prob=0.3, weight_matrix=None):
        super(EncoderDecoder_2, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size,hidden_size,vocab_size,num_layers,drop_prob, weight_matrix=None)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs