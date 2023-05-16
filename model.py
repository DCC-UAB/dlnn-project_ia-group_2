import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size:int, train_CNN:bool=False):
        super(EncoderCNN, self).__init__()
        # Boolean to train or not encoder
        self.train_CNN = train_CNN
        # Load inception model --> aux_logits=False cause not training
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
        # Change last layer to input of size of the embed
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        # Run x:image through the Inception model
        features = self.inception(images)
        # Always activate last layer parameters
        # If train == True: activate rest of parameters
        for name, param in self.inception.named_parameters():
            if 'fc.bias' in name or 'fc.weight' in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN
        # Return features ran through relu and dropout    
        return self.dropout(self.relu(features))
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size:int, hidden_size:int, vocab_size:int, num_layers:int):
        super(DecoderRNN, self).__init__()
        # Size of embed
        self.embed = nn.Embedding(vocab_size, embed_size)
        # LSTM from embedding size to hidden size
        self.lstm = nn.LSTM(embed_size, hidden_size)
        # Linear layer to output
        self.linear = nn.Linear(hidden_size, vocab_size)
        # Set dropout value
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, img_features, captions):
        # Run the captions through the embedding
        embeddings = self.embed(self.dropout(captions))
        # Concatenate feature with embedded caption on DIM=0
        embeddings = torch.cat((img_features.unsqueeze(0), embeddings), dim=0)
        # Run the embedding output to the LSTM
        output, hidden = self.lstm(embeddings)
        # Convert output to vocab_size to run through new embedding
        output = self.linear(output)
        return output


class EncoderCNN2DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(EncoderCNN2DecoderRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        output = self.decoder(features, captions)
        return output        

    def caption_image(self, image, vocabulary, max_length=40):
        caption = []
        with torch.no_grad():
            decoder_input = self.encoder(image).unsqueeze(0)
            states = None
            
            for i in range(max_length):
                output, states = self.decoder.lstm(decoder_input, states).squeeze(0)
                output = self.decoder.linear(output)
                predicted = output.argmax(1)
                
                caption.append(predicted.item())
                decoder_input = self.decoder.embed(predicted).unsqueeze(0)
                
                if vocabulary.itos[predicted] == '<EOS>':
                    break
                
        return [vocabulary.itos[idx] for idx in caption]