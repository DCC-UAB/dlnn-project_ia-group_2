import torch
import nltk
import random
from loaders import show_image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

def validate(criterion, model, loader, vocab_size, vocab, device, teacher_forcing_prob=0.5):
    losses = []
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (image, captions) in enumerate(iter(loader)):
            image, captions = image.to(device), captions.to(device)
            
            outputs = model(image, captions, teacher_forcing_prob)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            
            losses.append(loss.item())

    val_loss = torch.mean(torch.tensor(losses))

    print("Validation set: Average loss: {:.5f}".format(val_loss))
    return val_loss

def train(epoch, criterion, model, optimizer, loader, vocab_size, device, teacher_forcing_prob=0.5):
    print_every = 500
    losses = []
    
    model.train()
        
    for batch_idx, (images, captions) in enumerate(iter(loader)):
        # Zero gradients
        optimizer.zero_grad()

        images, captions = images.to(device), captions.to(device)
        
        if model.training and teacher_forcing_prob > 0.0:
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_prob
            if use_teacher_forcing:
                # Use ground truth captions for teacher forcing
                outputs = model(images, captions, teacher_forcing_prob=1.0)
            else:
                # Use model's predictions for teacher forcing
                outputs = model(images, captions)
        else:
            # Disable teacher forcing
            outputs = model(images, captions)
        
        # Calculate the batch loss
        loss = criterion(outputs.reshape(-1, vocab_size), captions.reshape(-1))
        
        # Backward pass.
        loss.backward()
        
        # Update the parameters in the optimizer.
        optimizer.step()     
        
        losses.append(loss.item())
        
        if (batch_idx) % print_every == 0:
            print("Train Epoch: {}; Loss: {:.5f}".format(epoch + 1, loss.item()))

    return losses


def val_visualize_captions(model, train_loader, val_loader, criterion, optimizer, device, vocab_size, vocab, epochs):
    print_every = 400
    model.train()
    for epoch in range(1, epochs+1):
        for idx, (image, captions) in enumerate(iter(train_loader)):
            image, captions = image.to(device), captions.to(device)

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
            
            if (idx+1) % print_every == 0:
                print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))
                
                
                # Generate the caption
                model.eval()
                with torch.no_grad():
                    dataiter = iter(val_loader)
                    img, _ = next(dataiter)
                    features = model.encoder(img[0:1].to(device))
                    print(f"features shape - {features.shape}")
                    caps = model.decoder.generate_caption(features.unsqueeze(0), vocab=vocab)
                    caption = ' '.join(caps)
                    print(caption)
                    show_image(img[0], title=caption)
                    
                model.train()
