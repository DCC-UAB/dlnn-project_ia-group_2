import wandb
import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from get_loader import show_image
from utils.utils import best_bleu_cap, img_denorm

# Function to validate the trained model, returns the average loss
def validate(criterion, model, loader, device): 

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
    return average_loss

# In this function we visualize the generated captions 
# for the val or test set once the model is trained 
def evaluate_caps(model, loader, df, vocab, device):
    print_every = 20
    #generate the caption
    model.eval()
    with torch.no_grad():
        for idx, (img, captions,img_dir) in enumerate(iter(loader)):
            if (idx+1)%print_every == 0:
                    df_filtered = df.loc[df['image'] == img_dir[0], 'caption']
                    original_captions = [caption.lower() for caption in df_filtered] # list of all the original captions
                    features = model.encoder(img[0:1].to(device))
                    caps = model.decoder.generate_caption(features.unsqueeze(0),vocab=vocab)
                    pred_caption = ' '.join(caps)
                    pred_caption = ' '.join(pred_caption.split()[1:-1]) # to erase sos and eos tokens from pred caption
                    original_caption, bleu_score = best_bleu_cap(original_captions, pred_caption) # call to function in utils.py
                    print("Best original caption (1 out of 5):", original_caption)
                    print("Predicted caption:", pred_caption)
                    print("BLEU score:", bleu_score)
                    img = img_denorm(img=img, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))
                    # img = TF.to_pil_image(img)
                    
                    show_image(img[0].cpu(),title=pred_caption)
        
# Function to calculate the average test BLEU            
def average_test_BLEU(model, loader, df, vocab, device):
    model.eval()
    total_bleu_score = 0.0
    total_predictions = 0

    with torch.no_grad():
        for idx, (img, captions, img_dir) in enumerate(iter(loader)):
            df_filtered = df.loc[df['image'] == img_dir[0], 'caption']
            original_captions = [caption.lower() for caption in df_filtered] # list of all the original captions
            features = model.encoder(img[0:1].to(device))
            caps = model.decoder.generate_caption(features.unsqueeze(0),vocab=vocab)
            pred_caption = ' '.join(caps)
            pred_caption = ' '.join(pred_caption.split()[1:-1]) # to erase sos and eos tokens from pred caption
            total_predictions += 1
            original_caption, bleu_score = best_bleu_cap(original_captions, pred_caption) # call to function in utils.py
            total_bleu_score += bleu_score

    average_bleu_score = total_bleu_score / total_predictions
    print("Average test BLEU-1 score:", average_bleu_score)