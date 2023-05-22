import wandb
import torch
from get_loader_v2_train_val_test import show_image

def test(criterion, model, loader, device): # vocab tendria q ser train_vocab_df

    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for images, captions in loader:
            images = images.to(device)
            captions = captions.to(device)
            batch_size = images.size(0)
            total_samples += batch_size

            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))
            total_loss += loss.item() * batch_size

    average_loss = total_loss / total_samples
    return average_loss

def val_visualize_captions_test(model, train_loader, val_loader, criterion, optimizer, device, vocab_size, vocab, epochs):
    print_every = 250
    model.train()
    for epoch in range(1, epochs+1):
        for idx, (image, captions) in enumerate(iter(train_loader)):
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
                        dataiter = iter(val_loader)
                        img,_ = next(dataiter)
                        features = model.encoder(img[0:1].to(device))
                        print(f"features shape - {features.shape}")
                        caps = model.decoder.generate_caption(features.unsqueeze(0),vocab=vocab)
                        caption = ' '.join(caps)
                        print(caption)
                        show_image(img[0],title=caption)
                        
                    model.train()

'''
def test(model, test_loader, device="cuda", save:bool= True):
    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    if save:
        print(len(images))
        # Save the model in the exchangeable ONNX format
        torch.onnx.export(model,  # model being run
                          images,  # model input (or a tuple for multiple inputs)
                          "model.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
        wandb.save("model.onnx")
        
'''