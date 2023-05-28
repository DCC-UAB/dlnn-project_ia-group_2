import torch
from get_loader import show_image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import best_bleu_cap
import torchvision.transforms as transforms



# Training function that calculates the average train loss at each epoch
def train(criterion, model, optimizer, loader, device):

    total_samples = 0
    total_loss = 0.0
    model.train()

    for batch_idx, (images, captions,_) in enumerate(loader):
        images = images.to(device)
        captions = captions.to(device)
        batch_size = images.size(0)
        total_samples += batch_size
        optimizer.zero_grad()
        outputs = model(images, captions)
        loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size

    average_loss = total_loss / total_samples
    return average_loss


# Function to train and generate captions at the same time to analyze how the model learns
# and improves its captions predictions with the pass of the epochs
def train_and_visualize_caps(epoch, train_dataloader, val_dataloader, model, optimizer, criterion, vocab, val_df, device):
    print_every = 400
    total_loss = 0
    total_samples = 0
    model.train()
    for batch_idx, (image, captions,_) in enumerate(iter(train_dataloader)):
        images, captions = image.to(device), captions.to(device)
        batch_size = images.size(0)
        total_samples += batch_size
        optimizer.zero_grad()
        outputs = model(images, captions)
        
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
        if (batch_idx + 1) % print_every == 0:
            print("Train Epoch: {} Batch [{}/{}]\tLoss: {:.5f}".format(epoch,
            batch_idx + 1, len(train_dataloader), loss.item()
        ))
            #Generate the caption
            model.eval()
            with torch.no_grad():
                dataiter = iter(val_dataloader)
                img,captions_val,img_dir = next(dataiter)
                df_filtered = val_df.loc[val_df['image'] == img_dir[0], 'caption']
                original_captions = [caption.lower() for caption in df_filtered] # List of all the original captions
                features = model.encoder(img[0:1].to(device))
                caps = model.decoder.generate_caption(features.unsqueeze(0),vocab=vocab)
                pred_caption = ' '.join(caps)
                pred_caption = ' '.join(pred_caption.split()[1:-1]) # To erase SOS and EOS tokens from predicted caption
                original_caption, bleu_score = best_bleu_cap(original_captions, pred_caption) # Call to function in utils.py
                print("Best original caption (1 out of 5):", original_caption)
                print("Predicted caption:", pred_caption)
                print("BLEU score :", bleu_score)
                show_image(img[0],title=pred_caption)
            model.train()
            
    average_loss = total_loss / total_samples
    print("Train Epoch: {} - Training set:  AVERAGE TRAINING LOSS: {:.5f}".format(epoch, average_loss))
    return average_loss

        
