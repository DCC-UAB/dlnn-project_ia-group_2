import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from model import EncoderCNN2DecoderRNN
from get_loader_v2_train_val_test import get_loader, get_length_vocab, get_pad_index, get_vocab, show_image

from get_loader_v2_train_val_test import Vocabulary
from get_loader_v2_train_val_test import main


import pandas as pd
from sklearn.model_selection import train_test_split
'''
data_dir = 'data/Images/'
captions_file = 'data/captions.txt'

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

# Split data into train and test sets
df_captions = pd.read_csv(captions_file)
unique_images = df_captions['image'].unique()
train_images, test_images = train_test_split(unique_images, test_size=0.2, random_state=42)
train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)

train_df = df_captions[df_captions['image'].isin(train_images)]
val_df = df_captions[df_captions['image'].isin(val_images)]
test_df = df_captions[df_captions['image'].isin(test_images)]

lenght_train_df = get_length_vocab(data_dir=data_dir, dataframe=train_df, transform=transform)
lenght_val_df = get_length_vocab(data_dir=data_dir, dataframe=val_df, transform=transform)
lenght_test_df = get_length_vocab(data_dir=data_dir, dataframe=test_df, transform=transform)

pad_index = get_pad_index(data_dir=data_dir, dataframe=train_df, transform=transform)

vocab_train_df = get_vocab(data_dir=data_dir, dataframe=train_df, transform=transform)
vocab_val_df = get_vocab(data_dir=data_dir, dataframe=val_df, transform=transform)
vocab_test_df = get_vocab(data_dir=data_dir, dataframe=test_df, transform=transform)


# Create train, validation, and test data loaders
train_dataloader = get_loader(data_dir=data_dir, dataframe=train_df, transform=transform)
val_dataloader = get_loader(data_dir=data_dir, dataframe=val_df, transform=transform)
test_dataloader = get_loader(data_dir=data_dir, dataframe=test_df, transform=transform)
    
'''
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
        features = self.resnet(images)
#         print(f"resenet features shape - {features.shape}")
        features = features.view(features.size(0),-1)
#         print(f"resenet features viewed shape - {features.shape}")
        features = self.embed(features)
#         print(f"resenet features embed shape - {features.shape}")
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers=1,drop_prob=0.3):
        super(DecoderRNN,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers=num_layers,batch_first=True)
        self.fcn = nn.Linear(hidden_size,vocab_size)
        self.drop = nn.Dropout(drop_prob)
    
    def forward(self,features, captions):
        # vectorize the caption
#         print(f"captions - {captions[:,:-1]}")
#         print(f"caption shape - {captions[:,:-1].shape}")
        embeds = self.embedding(captions[:,:-1])
#         print(f"shape of embeds - {embeds.shape}")
        # concat the features and captions
#         print(f"features shape - {features.shape}")
#         print(f"features unsqueeze at index 1 shape - {features.unsqueeze(1).shape}")
        x = torch.cat((features.unsqueeze(1),embeds),dim=1)
#         print(f"shape of x - {x.shape}")
        x,_ = self.lstm(x)
#         print(f"shape of x after lstm - {x.shape}")
        x = self.fcn(x)
#         print(f"shape of x after fcn - {x.shape}")
        return x
    
    def generate_caption(self,inputs,hidden=None,max_len=20,vocab=None):
    # Inference part
    # Given the image features generate the captions
    
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
            
            #send generated word as the next caption
            inputs = self.embedding(predicted_word_idx.unsqueeze(0))
        
        #covert the vocab idx to words and return sentence
        return [vocab[idx] for idx in captions]

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    data_dir = 'data/Images/'
    captions_file = 'data/captions.txt'

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    # Split data into train and test sets
    df_captions = pd.read_csv(captions_file)
    unique_images = df_captions['image'].unique()
    train_images, test_images = train_test_split(unique_images, test_size=0.2, random_state=42)
    train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)

    train_df = df_captions[df_captions['image'].isin(train_images)]
    val_df = df_captions[df_captions['image'].isin(val_images)]
    test_df = df_captions[df_captions['image'].isin(test_images)]

    lenght_train_df = get_length_vocab(data_dir=data_dir, dataframe=train_df, transform=transform)
    lenght_val_df = get_length_vocab(data_dir=data_dir, dataframe=val_df, transform=transform)
    lenght_test_df = get_length_vocab(data_dir=data_dir, dataframe=test_df, transform=transform)

    pad_index = get_pad_index(data_dir=data_dir, dataframe=train_df, transform=transform)

    vocab_train_df = get_vocab(data_dir=data_dir, dataframe=train_df, transform=transform)
    vocab_val_df = get_vocab(data_dir=data_dir, dataframe=val_df, transform=transform)
    vocab_test_df = get_vocab(data_dir=data_dir, dataframe=test_df, transform=transform)
    
    
    # Create train, validation, and test data loaders
    train_dataloader = get_loader(data_dir=data_dir, dataframe=train_df, transform=transform)
    val_dataloader = get_loader(data_dir=data_dir, dataframe=val_df, transform=transform)
    test_dataloader = get_loader(data_dir=data_dir, dataframe=test_df, transform=transform)

    # Hyperparameters
    embed_size = 400
    hidden_size = 512
    vocab_size = lenght_train_df
    num_layers = 2
    learning_rate = 0.0001

    # initialize model, loss etc
    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 20
    print_every = 2000

    for epoch in range(1,num_epochs+1):   
        for idx, (image, captions) in enumerate(train_dataloader):
            image,captions = image.to(device),captions.to(device)

            # Zero the gradients.
            optimizer.zero_grad()

            # Feed forward
            outputs = model(image, captions)
            
            # Calculate the batch loss.
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

            
            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()
            
            if (idx+1)%print_every == 0:
                print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))
                
                
                #generate the caption
                model.eval()
                with torch.no_grad():
                    dataiter = iter(train_dataloader)
                    img,_ = next(dataiter)
                    features = model.encoder(img[0:1].to(device))
                    print(f"features shape - {features.shape}")
                    caps = model.decoder.generate_caption(features.unsqueeze(0),vocab=vocab_train_df)
                    caption = ' '.join(caps)
                    print(caption)
                    show_image(img[0],title=caption)
                    
                model.train()
if __name__ == "__main__":
    main()
    