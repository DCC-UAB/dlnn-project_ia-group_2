import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import get_loader_v2

# ENCODER
class CNNEncoder(nn.Module):
 def __init__(self, embed_size, dropout = 0.2): # embed_size is the embedding size we want.
   super(CNNEncoder, self).__init__()
   #We are taking the last fully connected layer of the Inception network.
   self.inception = models.inception_v3(pretrained=True, aux_logits=False) 
   #Manually changing last Inception layer to map/connect to the embedding size we want.
   self.inception.fc = nn.Linear(self.inception.fc.in_features,embed_size)
   # ReLu layer.
   self.relu = nn.ReLU()
   # To prevent overfitting.
   self.dropout = nn.Dropout(dropout)

 def forward(self, input):
   features = self.inception(input)
   return self.dropout(self.relu(features))


# DECODER, podriamos probar BERT como decoder si tenemos tiempo q es mas pro.
class DecoderRNN(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, dropout = 0.2):
    super(DecoderRNN, self).__init__()
    # Size of embed
    self.embed = nn.Embedding(vocab_size, embed_size)
    # LSTM from embedding size to hidden size
    self.lstm = nn.LSTM(embed_size, hidden_size)
    # Linear layer to output
    self.linear = nn.Linear(hidden_size, vocab_size)
    # To prevent overfitting, can be discarded if we want.
    self.dropout = nn.Dropout(dropout)

    #Intento primitivo de attention.
    self.attention = nn.Linear(hidden_size, embed_size)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, features, captions):
    embeddings = self.dropout(self.embed(captions))
    embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)                                            
    hiddens, _ = self.lstm(embeddings)

    # Compute attention weights
    attention_scores = self.attention(hiddens)
    attention_weights = self.softmax(attention_scores)
    # Apply attention weights to hidden states
    attended_features = torch.sum(hiddens * attention_weights, dim=0)

    outputs = self.linear(attended_features)
    return outputs


class Encoder_Decoder(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size):
    super(Encoder_Decoder, self).__init__()
    self.cnn = CNNEncoder(embed_size)
    self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size)
                                  
  def forward(self, images, captions):
    features = self.cnn(images)
    outputs = self.decoderRNN(features, captions)
    return outputs
  

# Training

def train_model(model, data_loader, num_epochs, batch_size):
    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        loss_values = []

        for images, captions in data_loader:
            # Move images and captions to the device
            images = images.to(device)
            captions = captions.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, captions)

            # Compute the loss
            loss = criterion(images, captions) # Ns q criterion poner.

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update total loss
            loss_values.append(loss.detach().cpu().item())

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")

    # Return the trained model
    return model, loss_values


embed_size = 256
hidden_size = 512
vocab_size = 10000  # Poner la size de verdad
batch_size = 16
num_epochs = 100

model = Encoder_Decoder(embed_size, hidden_size, vocab_size)

data_loader = get_loader_v2(dataset, batch_size=batch_size, shuffle=True) # camviar datset para que sea el nuestro.

trained_model = train_model(model, data_loader, num_epochs, batch_size)