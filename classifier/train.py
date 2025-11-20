from head import ClassifierHead
from visualize import visualize_dataset

from datetime import datetime
import datasets as ds
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from setfit import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import sys
import torch
from torch.utils.data import DataLoader

MODEL_NAME = "sentence-transformers/embeddinggemma-300m-medical"

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def get_model(num_labels: int) -> tuple[SentenceTransformer, ClassifierHead]:
    """
    Loads embeddinggemma-300m-medical model and initializes the classification head.
    """
    try:
        model_body = SentenceTransformer(
            MODEL_NAME,
            # prompts={
            #     'retrieval (query)': 'task: search result | query: ',
            #     'retrieval (document)': 'title: {title | "none"} | text: ',
            #     'qa': 'task: question answering | query: ',
            #     'fact verification': 'task: fact checking | query: ',
            #     'classification': 'task: classification | query: ',
            #     'clustering': 'task: clustering | query: ',
            #     'semantic similarity': 'task: sentence similarity | query: ',
            #     'code retrieval': 'task: code retrieval | query: '
            # },
            # default_prompt_name='classification',
        )

        model_head = ClassifierHead(num_labels)

    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        print("Please ensure you have an internet connection and the transformers library installed.")
        raise RuntimeError("Failed to load the embedding model.")
    
    return model_body.to(device), model_head.to(device)

def get_model_train_test():
    # Login using e.g. `huggingface-cli login` to access this dataset
    cats = ds.ClassLabel(names=["medical", "insurance"])

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
    embedding_model, classifier = get_model(len(cats.names))
    
    return embedding_model, classifier, train, test, validation, cats.names

def test_loop(dataloader, model, embedding_model, loss_fn):
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

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_loop(dataloader, model, embedding_model, loss_fn, optimizer, batch_size = 64, epochs = 5):
    size = len(dataloader.dataset)
    total_loss = 0

    # Set models to training mode
    model.train()

    for batch in dataloader:
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(batch['label'])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        total_loss += loss.item()
    
    return total_loss

def checkpoint(datetime_start, checkpoint) -> str:
    return f"checkpoints/ckpt_{datetime_start.strftime('%Y%m%d_%H%M%S')}_epoch{checkpoint+1}.pth"

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
            tokenized_text = embedding_model.encode(texts, convert_to_tensor=True, device=device).detach()
        
        # 3. Convert string labels to integers
        int_labels = [label_map[l] for l in labels]
        tokenized_labels = torch.tensor(int_labels, dtype=torch.long)
        
        # 4. Add the labels as a PyTorch tensor
        tokenized_batch = {'sentence_embedding': tokenized_text, 'label': tokenized_labels}
        
        # # 5. Move inputs to the device
        # tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
        
        return tokenized_batch
    
    return collate_fn, label_map

def train():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    embedding_model, model, train_ds, test_ds, validation_ds, labels = get_model_train_test()
    batch_size = 64
    custom_collate_fn, label_map = label_to_int(embedding_model, labels)

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
        # collate_fn=custom_collate_fn
    )
    validation_dataloader = DataLoader(
        validation_ds, 
        batch_size=batch_size, 
        shuffle=True,
        # collate_fn=custom_collate_fn
    )

    start_datetime = datetime.now()

    loss_fn = model.get_loss_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    size = len(train_ds)
    save_per_epoch = 1
    epochs = 10

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        # Train
        total_loss = train_loop(train_dataloader, model, embedding_model, loss_fn, optimizer)

        # Save checkpoint
        if epoch % save_per_epoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_loss': total_loss
            }, checkpoint(start_datetime, epoch))
        
        # Test
        test_loop(test_dataloader, model, embedding_model, loss_fn)
        print(f"Epoch {epoch+1} Loss: {total_loss / size}")

    torch.save(model.state_dict(), f"models/classifier_weights_{start_datetime.strftime('%Y%m%d_%H%M%S')}.pth")

if __name__ == "__main__":
    train()
