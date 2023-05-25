import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.forcing_model import *
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


'''

def get_data(slice=1, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  equiv to slicing with [::slice] 
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader


def make(config, device="cuda"):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
    
'''