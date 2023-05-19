import torch
from get_loader_v2_train_val_test import show_image
import matplotlib.pyplot as plt



def validate(criterion, model, loader, vocab_size, vocab, device): # vocab tendria q ser train_vocab_df

    val_loss = 0
    #print_every = 500

    model.eval()
    
    with torch.no_grad():

        for batch_idx, (image, captions) in enumerate(iter(loader)):

            image, captions = image.to(device), captions.to(device)
            outputs = model(image, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            val_loss += loss.item()  
            
            '''
            if (batch_idx+1)%print_every == 0:
                    #generate the caption
                    dataiter = iter(loader)
                    img,_ = next(dataiter)
                    features = model.encoder(img[0:1].to(device))
                    print(f"features shape - {features.shape}")
                    caps = model.decoder.generate_caption(features.unsqueeze(0),vocab=vocab)
                    caption = ' '.join(caps)
                    print(caption)
                    show_image(img[0],title=caption)
'      

            '''

    val_loss /= len(loader.dataset)
    print("\nValidation set: Average loss: {:.5f}".format(val_loss))

    return val_loss


def train(epoch, criterion, model, optimizer, loader, vocab_size, device):
    
    total_loss = 0.0
    print_every = 100

    model.train()

    for batch_idx, (image, captions) in enumerate(iter(loader)):
        
        # Zero gradients
        optimizer.zero_grad()

        image,captions = image.to(device),captions.to(device)
        
        outputs = model(image, captions)
        # Calculate the batch loss
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        # Backward pass.
        loss.backward()
        # Update the parameters in the optimizer.
        optimizer.step()     
        
        total_loss += loss.item()  
          
        if (batch_idx+1)%print_every == 0:
            print("Train Epoch: {} loss: {:.5f}".format(epoch,loss.item()))

 

    return total_loss / len(loader.dataset)

    

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