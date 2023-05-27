
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
import numpy as np
from FORCING_model2 import EncoderDecoder
import nltk

def best_bleu_cap(list_original_caps, pred_cap):
    reference_captions = list_original_caps  # Lista de leyendas originales
    generated_caption = pred_cap  # Leyenda generada por el modelo

    best_bleu_score = 0.0
    best_caption = ""

    for reference_caption in reference_captions:
        # Tokenizar las leyendas de referencia y la leyenda generada
        reference_tokens = nltk.word_tokenize(reference_caption)
        generated_tokens = nltk.word_tokenize(generated_caption)
        
        # Calcular el puntaje BLEU
        bleu_score = nltk.translate.bleu_score.sentence_bleu([reference_tokens], generated_tokens)
        
        # Actualizar la mejor puntuación BLEU y la mejor leyenda
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            best_caption = reference_caption
    
    return best_caption, best_bleu_score


