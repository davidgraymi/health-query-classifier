from classifier.utils import CHECKPOINT_PATH, DATETIME_FORMAT, get_models, CATEGORIES, DEVICE

from datetime import datetime
import datasets as ds
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

def get_model_train_test():
    # Login using e.g. `huggingface-cli login` to access this dataset

    def add_static_label(row, column_name, label):
        row[column_name] = label
        return row

    # Miriad
    miriad = ds.load_dataset("tomaarsen/miriad-4.4M-split", split={"train":"train[:8000]", "test":"test[:2000]", "validation":"eval[:2000]"})
    miriad = miriad.rename_column("question", "text")
    miriad = miriad.remove_columns("passage_text")
    miriad = miriad.map(add_static_label, fn_kwargs={"column_name": "label", "label": "medical"})
    # print(miriad)

    # Insurance
    insurance = ds.load_dataset("deccan-ai/insuranceQA-v2", split={"train":"train[:8000]", "test":"test[:2000]", "validation":"validation[:2000]"})
    insurance = insurance.rename_column("input", "text")
    insurance = insurance.remove_columns(["output"])
    insurance = insurance.map(add_static_label, fn_kwargs={"column_name": "label", "label": "insurance"})
    # print(insurance)

    # Interleave datasets (mix the datasets into one randomly)
    train = ds.interleave_datasets([miriad["train"], insurance["train"]])
    _ , unique_indices = np.unique(train["text"], return_index=True, axis=0)
    train = train.select(unique_indices.tolist())
    test = ds.interleave_datasets([miriad["test"], insurance["test"]])
    _ , unique_indices = np.unique(test["text"], return_index=True, axis=0)
    test = test.select(unique_indices.tolist())
    validation = ds.interleave_datasets([miriad["validation"], insurance["validation"]])
    _ , unique_indices = np.unique(validation["text"], return_index=True, axis=0)
    validation = validation.select(unique_indices.tolist())
    
    # Get models
    embedding_model, classifier = get_models()
    
    return embedding_model, classifier, train, test, validation, CATEGORIES

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch)['logits']
            test_loss += loss_fn(pred, batch['label']).item()
            correct += (pred.argmax(1) == batch['label']).type(torch.float).sum().item()

    avg_loss = test_loss / num_batches
    accuracy = correct / size

    return avg_loss, accuracy

def train_loop(dataloader, model, loss_fn, optimizer, batch_size = 64, epochs = 10):
    size = len(dataloader.dataset)
    total_loss = 0
    batch_losses = []

    # Set models to training mode
    model.train()

    for iteration, batch in enumerate(dataloader):
        # --- 1. Zero Gradients ---
        # Only zero gradients for the parameters you want to update (the classifier head)
        optimizer.zero_grad()
        
        # --- 3. Forward Pass: Embeddings -> Logits ---
        # The classifier head takes the embeddings from the body
        pred = model(batch)['logits']
        
        # --- 4. Calculate Loss ---
        loss = loss_fn(pred, batch['label'])

        # --- 5. Backward Pass & Update ---
        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        batch_losses.append(cur_loss)
        total_loss += cur_loss

        if iteration % 100 == 0:
            current = iteration * batch_size + len(batch['label'])
            print(f"loss: {cur_loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    return total_loss, batch_losses

def checkpoint(save_dir, checkpoint) -> str:
    return f"{save_dir}/ckpt-{checkpoint+1}.pth"

def label_to_int(embedding_model, label_names: list):
    """Creates a dictionary mapping label strings to their integer IDs."""
    label_map = {name: i for i, name in enumerate(label_names)}
    
    def collate_fn(batch):
        # 1. Extract texts and labels from the batch (list of dictionaries)
        texts = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]
        
        # 2. Tokenize the texts using the embedding model's tokenizer
        # The tokenizer is attached to the embedding_model
        with torch.no_grad():
            tokenized_text = embedding_model.encode(
                texts,
                convert_to_tensor=True,
                device=DEVICE
            ).clone().detach()
        
        # 3. Convert string labels to integers
        int_labels = [label_map[l] for l in labels]
        tokenized_labels = torch.tensor(int_labels, dtype=torch.long)
        
        # 4. Add the labels as a PyTorch tensor
        tokenized_batch = {'sentence_embedding': tokenized_text.to(DEVICE), 'label': tokenized_labels.to(DEVICE)}
        
        return tokenized_batch
    
    return collate_fn

def train():
    start_datetime = datetime.now()

    save_dir = f'{CHECKPOINT_PATH}/{start_datetime.strftime(DATETIME_FORMAT)}'
    os.makedirs(save_dir, exist_ok=True)

    embedding_model, model, train_ds, test_ds, validation_ds, labels = get_model_train_test()
    batch_size = 64
    custom_collate_fn = label_to_int(embedding_model, labels)

    train_dataloader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    test_dataloader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    validation_dataloader = DataLoader(
        validation_ds, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    loss_fn = model.get_loss_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    save_per_epoch = 1
    epochs = 1
    patience = 1
    min_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss_epoch': [],
        'train_loss_batch': [],
        'validation_accuracy': [],
        'validation_loss_epoch': [],
        'test_accuracy': [],
        'test_loss': []
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}:\n-------------------------------")

        # Train
        total_loss, batch_losses = train_loop(train_dataloader, model, loss_fn, optimizer)
        avg_epoch_loss = total_loss / len(train_dataloader)
        history['train_loss_epoch'].append(avg_epoch_loss)
        history['train_loss_batch'].extend(batch_losses)

        summary = f"Epoch {epoch+1}:"

        # Save checkpoint
        if epoch % save_per_epoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint(save_dir, epoch))
            summary += f" -- {checkpoint(save_dir, epoch)}\n"

            history_df = pd.DataFrame.from_dict(history, orient='index').transpose()
            history_df.to_csv(f"{save_dir}/history.csv", index=False)
        else:
            summary += "\n"

        # Validate
        val_loss_avg, val_accuracy = test_loop(validation_dataloader, model, loss_fn)
        history['validation_accuracy'].append(val_accuracy)
        history['validation_loss_epoch'].append(val_loss_avg)

        summary += f" - loss: {avg_epoch_loss}\n"
        summary += f" - training loss: {avg_epoch_loss}\n"
        summary += f" - validation loss: {val_loss_avg:>8f}\n"
        summary += f" - validation accuracy: {(100*val_accuracy):>0.1f}%\n"

        print(summary)

        if val_loss_avg < min_val_loss:
            min_val_loss = val_loss_avg
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered due to no improvement in validation loss.")
                break

    # Save the final model
    torch.save(model.state_dict(), f"{save_dir}/final.pth")

    # Save loss history
    history_df = pd.DataFrame.from_dict(history, orient='index').transpose()
    history_df.to_csv(f"{save_dir}/history.csv", index=False)

    # Evaluate on test dataset
    test_loss_avg, test_accuracy = test_loop(test_dataloader, model, loss_fn)
    history['test_accuracy'].append(test_accuracy)
    history['test_loss'].append(test_loss_avg)
    print(f"Test: Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss_avg:>8f}")

    # Plot training loss per batch
    fig, ax = plt.subplots()
    ax.plot(history['train_loss_batch'])
    ax.set_title('Training Loss per Batch')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    fig.savefig(f"{save_dir}/loss.png")

if __name__ == "__main__":
    train()
