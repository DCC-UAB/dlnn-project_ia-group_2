import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3, weight_matrix=None, pretrained=True):
        super(DecoderRNN, self).__init__()
        if weight_matrix is not None:
            self.embedding, vocab_size, embed_size = self.create_embedding_layer(weight_matrix, pretrained)
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
            self.fcn = nn.Linear(hidden_size, vocab_size)
            self.drop = nn.Dropout(drop_prob)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
            self.fcn = nn.Linear(hidden_size, vocab_size)
            self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.drop(lstm_out)
        outputs = self.fcn(lstm_out)
        return outputs

    def create_embedding_layer(self, weight_matrix, pretrained):
        vocab_size, embed_size = weight_matrix.shape
        embedding = nn.Embedding(vocab_size, embed_size)
        embedding.weight = nn.Parameter(torch.tensor(weight_matrix, dtype=torch.float32))
        embedding.weight.requires_grad = pretrained
        return embedding, vocab_size, embed_size

def generate_caption(encoder, decoder, image, vocabulary, max_length=20):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        image = image.unsqueeze(0)
        features = encoder(image)
        states = None
        captions = []

        for _ in range(max_length):
            inputs = torch.tensor([vocabulary.stoi['<SOS>']]).unsqueeze(0)
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            outputs, states = decoder(features, inputs, states)
            predicted = outputs.argmax(2)
            word = vocabulary.itos[predicted.item()]
            captions.append(word)

            if word == '<EOS>':
                break

        caption = ' '.join(captions)
        return caption