import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
import numpy as np
from model.model import *
import nltk

def best_bleu_cap(list_original_caps, pred_cap):

    # Initialization of variables to return
    best_bleu_score = 0.0
    best_caption = ""
    
    # Iterate over the 5 captions that the dataset provides for each image
    for reference_caption in list_original_caps:
        
        # Tokenize
        reference_tokens = nltk.word_tokenize(reference_caption)
        generated_tokens = nltk.word_tokenize(pred_cap )
        
        # Calculate BLEU-1, based on unigrams
        bleu_score = nltk.translate.bleu_score.sentence_bleu([reference_tokens], generated_tokens, weights=(1, 0, 0, 0))
        
        # Keep the original caption that maximizes more BLEU-1 score 
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            best_caption = reference_caption
    
    return best_caption, best_bleu_score


def weights_matrix(vocab, emb_dim, glove_embedding):
    """
    Input:
        - Vocabulary of dataset format -> {idx : word}
        - Embedding dimension
        
    Function returns a matrix with weight values that reresent the pretrained embedding.
    
    """
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0

    for i, word in enumerate(vocab):
        try: 
            weights_matrix[i] = glove_embedding[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
    
    return weights_matrix


def img_denorm(img, mean, std):
    #for ImageNet the mean and std are:
    #mean = np.asarray([ 0.485, 0.456, 0.406 ])
    #std = np.asarray([ 0.229, 0.224, 0.225 ])

    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

    res = img.squeeze(0)
    res = denormalize(res)

    #Image needs to be clipped since the denormalize function will map some
    #values below 0 and above 1
    res = torch.clamp(res, 0, 1)
    
    return(res)