import os
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    # CHANGED: Default to DeBERTa-v3
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--train", default="data_combined/train.jsonl")
    ap.add_argument("--dev", default="data_combined/dev.jsonl")
    ap.add_argument("--out_dir", default="out_deberta")
    # CHANGED: Lower batch size for DeBERTa (it is memory intensive)
    ap.add_argument("--batch_size", type=int, default=16) 
    ap.add_argument("--epochs", type=int, default=20)
    # CHANGED: Lower LR is better for DeBERTa
    ap.add_argument("--lr", type=float, default=2e-5) 
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def evaluate(model, dataloader, device):
    """Calculates validation loss on the dev set."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    return total_loss / max(1, len(dataloader))


def plot_losses(train_losses, val_losses, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_plot.png"))
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load Datasets
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )
    
    dev_dl = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name)
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Tracking
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_dl))
        train_losses.append(avg_train_loss)
        
        # --- Validation ---
        avg_val_loss = evaluate(model, dev_dl, args.device)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
            print(f"  -> New best model saved!")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{args.patience}")
            
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    plot_losses(train_losses, val_losses, args.out_dir)
    print(f"Training complete. Loss plot saved to {args.out_dir}/loss_plot.png")


if __name__ == "__main__":
    main()
