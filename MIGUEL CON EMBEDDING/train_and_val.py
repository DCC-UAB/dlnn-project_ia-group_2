import torch
from get_loader import show_image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils_2 import best_bleu_cap
import torchvision.transforms as transforms

# Here we only print and calculate the validation loss
def validate(criterion, model, loader, device): # vocab tendria q ser train_vocab_df

    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for images, captions,_ in loader:
            images = images.to(device)
            captions = captions.to(device)
            batch_size = images.size(0)
            total_samples += batch_size

            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))
            total_loss += loss.item() * batch_size

    average_loss = total_loss / total_samples
    print("Validation set: AVERAGE VALIDATION LOSS: {:.5f}".format(average_loss))
    return average_loss

# Here we only print and calculate the train loss
def train(epoch, criterion, model, optimizer, loader, device):
    total_samples = 0
    total_loss = 0.0
    print_every = 250

    model.train()

    for batch_idx, (images, captions,_) in enumerate(loader):
        images = images.to(device)
        
        batch_size = images.size(0)
        total_samples += batch_size
        optimizer.zero_grad()

        outputs = model(images, captions)
        loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size

        if (batch_idx + 1) % print_every == 0:
            print("Train Epoch: {} Batch [{}/{}]\tLoss: {:.5f}".format(
                epoch, batch_idx + 1, len(loader), loss.item()
            ))

    average_loss = total_loss / total_samples
    print("Train Epoch: {} Average Loss: {:.5f}".format(epoch, average_loss))

    return average_loss
        
# In this function we train the model and visualize the 
# generated caption for image in the val set every 400 in the batch 
# per epoch we see the train loss 2 times (every 400 and the batch is of size 800))
# Also per epoch we visualize the avergae training loss and val loss in the batch and the plot to compare them
# Here we have merged visualization of caps with training to use ONLY 1 function to train
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
            #generate the caption
            model.eval()
            with torch.no_grad():
                dataiter = iter(val_dataloader)
                img,captions_val,img_dir = next(dataiter)
                df_filtered = val_df.loc[val_df['image'] == img_dir[0], 'caption']
                original_captions = [caption.lower() for caption in df_filtered] # list of all the original captions
                features = model.encoder(img[0:1].to(device))
                caps = model.decoder.generate_caption(features.unsqueeze(0),vocab=vocab)
                pred_caption = ' '.join(caps)
                pred_caption = ' '.join(pred_caption.split()[1:-1]) # to erase sos and eos tokens from pred caption
                original_caption, bleu_score = best_bleu_cap(original_captions, pred_caption) # call to function in utils.py
                print("Best original caption (1 out of 5):", original_caption)
                print("Predicted caption:", pred_caption)
                print("BLEU score :", bleu_score)
                show_image(img[0],title=pred_caption)
            model.train()
            
    average_loss = total_loss / total_samples
    print("Train Epoch: {} - Training set:  AVERAGE TRAINING LOSS: {:.5f}".format(epoch, average_loss))
    return average_loss
                    
# In this function we visualize the generated captions 
# for the val or test set once the model is trained to see its performance
def evaluate_caps( model, loader, df, vocab, device):
    print_every = 50
    #generate the caption
    model.eval()
    with torch.no_grad():
        for idx, (img, captions,img_dir) in enumerate(iter(loader)):
            if (idx+1)%print_every == 0:
                df_filtered = df.loc[df['image'] == img_dir[0], 'caption']
                original_captions = [caption.lower() for caption in df_filtered] # list of all the original captions
                features = model.encoder(img[0:1].to(device))
                print(f"features shape - {features.shape}")
                caps = model.decoder.generate_caption(features.unsqueeze(0),vocab=vocab)
                pred_caption = ' '.join(caps)
                pred_caption = ' '.join(pred_caption.split()[1:-1]) # to erase sos and eos tokens from pred caption
                original_caption, bleu_score = best_bleu_cap(original_captions, pred_caption) # call to function in utils.py
                print("Best original caption (1 out of 5):", original_caption)
                print("Predicted caption:", pred_caption)
                print("BLEU score:", bleu_score)
                show_image(img[0],title=pred_caption)

'''

from tqdm.auto import tqdm
import wandb

def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images, labels, model, optimizer, criterion, device="cuda"):
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
    
'''