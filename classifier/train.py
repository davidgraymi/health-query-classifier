from classifier.utils import CHECKPOINT_PATH, DATETIME_FORMAT, get_models, CATEGORIES, DEVICE, CLASSIFIER_NAME
from classifier.config import HF_TOKEN
from huggingface_hub import HfApi
from jinja2 import Template

import argparse
from datetime import datetime
import datasets as ds
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

def even_split(prefix: str, target: int, splits: int, total: int) -> str:
    result = ""
    target_amount_per_split = int(target / splits)
    total_amount_per_split = int(total / splits)

    for i in range(splits):
        left = total_amount_per_split*i
        right = left + target_amount_per_split
        result += f"{prefix}[{int(left)}:{int(right)}]"

        if i != splits - 1:
            result += "+"

    return result

def get_model_train_test():
    # Login using e.g. `huggingface-cli login` to access this dataset

    def add_static_label(row, column_name, label):
        row[column_name] = label
        return row

    # Miriad
    train_split = even_split("train", 50000, 100, 4470000)
    miriad = ds.load_dataset("tomaarsen/miriad-4.4M-split", split={"train":train_split, "test": "test", "validation": "eval"})
    miriad = miriad.rename_column("question", "text")
    miriad = miriad.remove_columns("passage_text")
    miriad = miriad.map(add_static_label, fn_kwargs={"column_name": "label", "label": "medical"})
    # print(miriad)

    # Insurance
    train_split = even_split("train", 5000, 20, 21300)
    insurance = ds.load_dataset("deccan-ai/insuranceQA-v2", split={"train":train_split, "test":"test", "validation":"validation"})
    insurance = insurance.rename_column("input", "text")
    insurance = insurance.remove_columns(["output"])
    insurance = insurance.map(add_static_label, fn_kwargs={"column_name": "label", "label": "insurance"})
    # print(insurance)

    # Interleave datasets (mix the datasets into one randomly)
    train = ds.interleave_datasets([miriad["train"], insurance["train"]], stopping_strategy="all_exhausted")
    _ , unique_indices = np.unique(train["text"], return_index=True, axis=0)
    train = train.select(unique_indices.tolist())
    test = ds.interleave_datasets([miriad["test"], insurance["test"]], stopping_strategy="all_exhausted")
    _ , unique_indices = np.unique(test["text"], return_index=True, axis=0)
    test = test.select(unique_indices.tolist())
    validation = ds.interleave_datasets([miriad["validation"], insurance["validation"]], stopping_strategy="all_exhausted")
    _ , unique_indices = np.unique(validation["text"], return_index=True, axis=0)
    validation = validation.select(unique_indices.tolist())

    print(f"train: {len(train)}, validation: {len(validation)}, test: {len(test)}")
    
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

def generate_model_card(save_dir: str, accuracy: float, loss: float, epoch: int):
    with open("classifier/modelcard_template.md", "r") as f:
        template_content = f.read()
    
    template = Template(template_content)
    
    card_content = template.render(
        model_id=CLASSIFIER_NAME,
        model_summary="A simple medical query triage classifier.",
        model_description="This model classifies queries into 'medical' or 'insurance' categories. It uses EmbeddingGemma-300M as a backbone.",
        developers="David Gray",
        model_type="Text Classification",
        language="en",
        license="mit",
        base_model="sentence-transformers/embeddinggemma-300m-medical",
        repo=f"https://huggingface.co/{CLASSIFIER_NAME}",
        results_summary=f"Epoch: {epoch+1}\nValidation Accuracy: {accuracy*100:.2f}%\nValidation Loss: {loss:.4f}",
        training_data="Miriad (medical) and InsuranceQA (insurance) datasets.",
        testing_metrics="Accuracy, Loss",
        results=f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}"
    )
    
    with open(f"{save_dir}/README.md", "w") as f:
        f.write(card_content)

def push_model_card(save_dir: str, repo_id: str, token: str = None):
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=f"{save_dir}/README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )

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

def train(push_to_hub: bool = False):
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

        # Validate
        val_loss_avg, val_accuracy = test_loop(validation_dataloader, model, loss_fn)
        history['validation_accuracy'].append(val_accuracy)
        history['validation_loss_epoch'].append(val_loss_avg)

        summary += f" - loss: {avg_epoch_loss}\n"
        summary += f" - training loss: {avg_epoch_loss}\n"
        summary += f" - validation loss: {val_loss_avg:>8f}\n"
        summary += f" - validation accuracy: {(100*val_accuracy):>0.1f}%\n"

        # Save checkpoint
        if epoch % save_per_epoch == 0:
            # Save model
            model.save_pretrained(save_dir)
            
            # Generate and push model card
            # generate_model_card(save_dir, val_accuracy, val_loss_avg, epoch)
            # push_model_card(save_dir, CLASSIFIER_NAME, token=HF_TOKEN)
            
            summary += f" -- {save_dir}\n"

            history_df = pd.DataFrame.from_dict(history, orient='index').transpose()
            history_df.to_csv(f"{save_dir}/history.csv", index=False)

            # Push model to Hugging Face
            if push_to_hub:
                model.push_to_hub(CLASSIFIER_NAME, token=HF_TOKEN)
        else:
            summary += "\n"

        print(summary)

        if val_loss_avg < min_val_loss:
            min_val_loss = val_loss_avg
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered due to no improvement in validation loss.")
                break

    # Evaluate on test dataset
    test_loss_avg, test_accuracy = test_loop(test_dataloader, model, loss_fn)
    history['test_accuracy'].append(test_accuracy)
    history['test_loss'].append(test_loss_avg)
    print(f"Test: Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss_avg:>8f}")

    # Save the final model
    model.save_pretrained(save_dir)
    
    # generate_model_card(save_dir, test_accuracy, test_loss_avg, epochs-1)
    # push_model_card(save_dir, CLASSIFIER_NAME, token=HF_TOKEN)

    # Save loss history
    history_df = pd.DataFrame.from_dict(history, orient='index').transpose()
    history_df.to_csv(f"{save_dir}/history.csv", index=False)

    # Plot training loss per batch
    fig, ax = plt.subplots()
    ax.plot(history['train_loss_batch'])
    ax.set_title('Training Loss per Batch')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    fig.savefig(f"{save_dir}/loss.png")

    if push_to_hub:
        model.push_to_hub(CLASSIFIER_NAME, token=HF_TOKEN)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train a classifier for triaging health queries"
    )
    ap.add_argument(
        "--push", action="store_true",
        help="Push model to Hugging Face"
    )
    args = ap.parse_args()

    train(push_to_hub=args.push)
