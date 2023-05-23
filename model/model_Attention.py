import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        
        self.resnet = models.resnet50(pretrained=True) 
        for name, param in self.resnet.named_parameters():
            if 'fc.bias' in name or 'fc.weights' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
       
    def forward(self, images):
        features = self.resnet(images)
        return self.dropout(self.relu(features))


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, encoder_out, decoder_hidden):
        encoder_att = self.encoder_att(encoder_out)
        decoder_att = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(encoder_att + decoder_att.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.5):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + hidden_size, hidden_size)
        self.attention = Attention(embed_size, hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, features, captions):
        batch_size = features.size(0)
        num_pixels = features.size(1)
        hidden_state = self.init_hidden_state(features)
        cell_state = self.init_cell_state(features)
        embeddings = self.dropout(self.embedding(captions))
        predictions = []
        
        for t in range(embeddings.size(1)):
            attention_weighted_encoding, _ = self.attention(features, hidden_state[0])
            lstm_input = torch.cat((embeddings[:, t], attention_weighted_encoding), dim=1)
            hidden_state, cell_state = self.lstm(lstm_input, (hidden_state, cell_state))
            output = self.linear(hidden_state)
            predictions.append(output)
        
        return torch.stack(predictions, dim=1)
    
    def generate_caption(self, inputs, hidden=None, max_len=25, vocab=None):
        batch_size = inputs.size(0)
        captions = []
        
        for i in range(max_len):
            attention_weighted_encoding, _ = self.attention(inputs, hidden[0][0])
            lstm_input = torch.cat((inputs[:, -1], attention_weighted_encoding), dim=1)
            hidden = self.lstm(lstm_input, hidden)
            output = self.linear(hidden[0])
            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())
            
            if vocab[predicted_word_idx.item()] == "<EOS>":
                break
            
            inputs = self.embedding(predicted_word_idx.unsqueeze(0))
        
        return [vocab[idx] for idx in captions]
    
    def init_hidden_state(self, features):
        return torch.zeros(features.size(0), self.hidden_size).to(features.device)
    
    def init_cell_state(self, features):
        return torch.zeros(features.size(0), self.hidden_size).to(features.device)


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, drop_prob)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
