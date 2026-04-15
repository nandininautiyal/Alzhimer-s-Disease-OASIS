import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.microbifpn import create_model
from data.dataset import CustomDataset
from data.augument import get_train_transforms, get_val_transforms
from config import microbifpn_config
from utils.plotting import plot_CM

# ─────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', default='./df_train.csv')
    parser.add_argument('--val_csv', default='./df_val.csv')
    parser.add_argument('--test_csv', default='./df_test.csv')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()

# ─────────────────────────────
def run_epoch(model, dataloader, optimizer, device, training=True):
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(dataloader, leave=False):

            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = model.compute_loss(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = (all_preds == all_labels).mean()
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0

    return {
        "loss": total_loss / len(dataloader),
        "acc": acc,
        "auc": auc,
        "precision": prec,
        "recall": rec,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs
    }

# ─────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)
    df_test = pd.read_csv(args.test_csv)

    train_ds = CustomDataset(df_train, transform=get_train_transforms())
    val_ds = CustomDataset(df_val, transform=get_val_transforms())
    test_ds = CustomDataset(df_test, transform=get_val_transforms())

    
    labels = train_ds.labels_binary
    class_counts = np.bincount(labels)

    
    weights = 1.0 / (class_counts ** 0.7)
    sample_weights = weights[labels]

    
    sampler = WeightedRandomSampler(sample_weights, 3500)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = create_model(microbifpn_config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-4,   
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=microbifpn_config['lr_step_size'],
        gamma=microbifpn_config['lr_gamma']
    )

    best_acc = 0.0
    patience = 4
    counter = 0

    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")

        train_m = run_epoch(model, train_loader, optimizer, device, True)
        val_m = run_epoch(model, val_loader, optimizer, device, False)

        print(f"Train Acc: {train_m['acc']:.4f} | Val Acc: {val_m['acc']:.4f}")

        if val_m["acc"] > best_acc:
            best_acc = val_m["acc"]
            counter = 0
            os.makedirs("./weights_finalized", exist_ok=True)
            torch.save(model.state_dict(), "./weights_finalized/best_model.pth")
            print("✔ Model Saved")
        else:
            counter += 1

        if counter >= patience:
            print("⛔ Early stopping triggered")
            break

        scheduler.step()

    # ───── TEST ─────
    print("\nTesting...")
    model.load_state_dict(torch.load("./weights_finalized/best_model.pth"))

    test_m = run_epoch(model, test_loader, None, device, False)

    print("Accuracy:", test_m["acc"])
    print("AUC:", test_m["auc"])

    cm = confusion_matrix(test_m["labels"], test_m["preds"])
    print("\nConfusion Matrix:\n", cm)

    print("\nClassification Report:\n",
          classification_report(test_m["labels"], test_m["preds"], zero_division=0))

    os.makedirs("./training_plots", exist_ok=True)
    plot_CM(cm, "./training_plots")


if __name__ == "__main__":
    main()