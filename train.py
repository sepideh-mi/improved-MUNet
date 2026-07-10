import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import BraTSDataset, LightBraTSDataset
from model import MUNet
from loss import SegmentationLoss
import os, csv, time, numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# ----- Configuration (core settings) -----
class Config:
    epochs = 100
    batch_size = 2
    lr = 5e-5
    num_workers = 0
    in_channels, out_channels = 4, 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "/content/drive/MyDrive/checkpoints"
    train_data_dir = "/content/drive/MyDrive/Munet/Train_validation_Test/train"
    val_data_dir = "/content/drive/MyDrive/Munet/Train_validation_Test/val"
    log_file = os.path.join(checkpoint_dir, "training_logs.csv")
    phase_specific = True
    METRICS = ['accuracy', 'precision', 'recall', 'mIoU', 'dice_score']
    lr_scheduler = 'cosine'
    lr_warmup_epochs = 5
    early_stopping_patience = 20


def masks_to_one_hot(masks, num_classes):
    return F.one_hot(masks.long(), num_classes).permute(0,3,1,2).float()

def safe_mean(vals):
    vals = [v.item() if torch.is_tensor(v) else v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else 0.0

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), [b[2] for b in batch]

# ----- Training loop 
def main():
    os.makedirs(Config.checkpoint_dir, exist_ok=True)

    # Datasets & Loaders
    train_set = BraTSDataset(Config.train_data_dir, csv_path=...,
                             use_one_third_data=Config.use_one_third_data, phase="train")
    val_set = LightBraTSDataset(Config.val_data_dir, csv_path=...,
                                use_one_third_data=Config.use_one_third_data, phase="val")

    # Use weighted sampler for class imbalance
    sample_weights = train_set.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, sampler=sampler,
                              collate_fn=collate_fn, num_workers=Config.num_workers)
    val_loader = DataLoader(val_set, batch_size=Config.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=Config.num_workers)

    # Model, Optimizer, Scheduler, Loss
    model = MUNet(Config.in_channels, Config.out_channels).to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=1e-6)
    criterion = SegmentationLoss(class_weights=torch.ones(Config.out_channels).to(Config.device)).to(Config.device)

    # Resume if possible
    start_epoch = 0
    if os.path.exists(os.path.join(Config.checkpoint_dir, "latest_model.pth")):
        checkpoint = torch.load(...)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, Config.epochs):
        # ----- Training -----
        model.train()
        train_loss, train_metrics = [], {m: [] for m in Config.METRICS}
        for images, masks, _ in tqdm(train_loader, desc=f"Train E{epoch+1}"):
            images, masks = images.to(Config.device), masks.to(Config.device)
            masks_one_hot = masks_to_one_hot(masks, Config.out_channels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks_one_hot)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss.append(loss.item())
            # collect metrics
            with torch.no_grad():
                batch_metrics = criterion.calculate_metrics(outputs, masks_one_hot)
                for k, v in batch_metrics.items():
                    if k in train_metrics: train_metrics[k].append(v)

        # ----- Validation -----
        model.eval()
        val_loss, val_metrics = [], {m: [] for m in Config.METRICS}
        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc=f"Val E{epoch+1}"):
                images, masks = images.to(Config.device), masks.to(Config.device)
                masks_one_hot = masks_to_one_hot(masks, Config.out_channels)
                outputs = model(images)
                loss = criterion(outputs, masks_one_hot)
                val_loss.append(loss.item())
                batch_metrics = criterion.calculate_metrics(outputs, masks_one_hot)
                for k, v in batch_metrics.items():
                    if k in val_metrics: val_metrics[k].append(v)

        # Aggregate metrics
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        train_summary = {k: safe_mean(v) for k, v in train_metrics.items()}
        val_summary = {k: safe_mean(v) for k, v in val_metrics.items()}

        # LR scheduling
        if Config.lr_scheduler == 'cosine':
            scheduler.step()
        # else: plateau would use val loss

        # Checkpoint & early stopping
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(Config.checkpoint_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= Config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Log to CSV
        with open(Config.log_file, 'a') as f:
            writer = csv.writer(f)
            row = [epoch+1, avg_train_loss, avg_val_loss] + \
                  [train_summary.get(m, 0) for m in Config.METRICS] + \
                  [val_summary.get(m, 0) for m in Config.METRICS] + \
                  [optimizer.param_groups[0]['lr']]
            if epoch == 0: writer.writerow(['Epoch','TrainLoss','ValLoss'] + [f'Train_{m}' for m in Config.METRICS] + [f'Val_{m}' for m in Config.METRICS] + ['LR'])
            writer.writerow(row)

        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, LR {optimizer.param_groups[0]['lr']:.2e}")

if __name__ == "__main__":
    main()
