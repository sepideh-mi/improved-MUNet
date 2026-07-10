import os, csv, torch, numpy as np, nibabel as nib
from torch.utils.data import DataLoader
from model import MUNet
from dataset import BraTSDataset
from loss import SegmentationLoss
from scipy.ndimage import distance_transform_edt

# ----- Helper: 95% Hausdorff Distance -----
def hd95(pred, target):
    if pred.sum() == 0 or target.sum() == 0:
        return float('nan')
    p, t = pred.cpu().numpy().astype(bool), target.cpu().numpy().astype(bool)
    d1 = distance_transform_edt(~t)[p]
    d2 = distance_transform_edt(~p)[t]
    return max(np.percentile(d1, 95) if len(d1) else float('inf'),
               np.percentile(d2, 95) if len(d2) else float('inf'))

# ----- Main Tester -----
class Tester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = "/content/drive/MyDrive/Munet/Train_validation_Test/test"
        self.ckpt_path = "/content/drive/MyDrive/checkpoints/best_model.pth"
        self.out_dir = "./outputs"
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pred_masks"), exist_ok=True)

        self.in_channels, self.out_channels = 4, 4
        self.model = MUNet(self.in_channels, self.out_channels).to(self.device)
        self._load_model()
        self._setup_data()
        self.loss_fn = SegmentationLoss(num_classes=self.out_channels).to(self.device)

    def _load_model(self):
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        print("✅ Model loaded")

    def _setup_data(self):
        self.dataset = BraTSDataset(self.data_path, use_one_third_data=False, use_augmentation=False)
        self.loader = DataLoader(self.dataset, batch_size=2, shuffle=False, num_workers=0)

    def _save_mask(self, mask, case_id):
        nib.save(nib.Nifti1Image(mask.cpu().numpy().astype(np.uint8), np.eye(4)),
                 os.path.join(self.out_dir, "pred_masks", f"{case_id}_pred.nii.gz"))

    def evaluate(self):
        metrics = {k: [] for k in ['dice_score','accuracy','precision','recall','fpr',
                                   'wt_iou','tc_iou','et_iou','wt_dice','tc_dice','et_dice',
                                   'wt_fpr','tc_fpr','et_fpr','wt_hd95','tc_hd95','et_hd95']}
        with torch.no_grad():
            for idx, batch in enumerate(self.loader):
                if batch is None:
                    continue
                images, masks, types = batch
                images, masks = images.to(self.device), masks.to(self.device)
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                masks_one_hot = torch.nn.functional.one_hot(masks, self.out_channels).permute(0,3,1,2).float()
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)

                # batch metrics from loss function
                batch_metrics = self.loss_fn.calculate_metrics(outputs, masks_one_hot)
                for k in metrics:
                    if k in batch_metrics:
                        metrics[k].append(batch_metrics[k].item())

                # per‑sample HD95 for each region
                for i in range(preds.shape[0]):
                    case_id = f"case_{idx*2+i}"
                    self._save_mask(preds[i], case_id)
                    # region masks (WT, TC, ET)
                    regions = {'wt': [1,2,3], 'tc': [1,3], 'et': [3]}
                    for name, classes in regions.items():
                        pred_region = torch.zeros_like(preds[i], dtype=torch.bool)
                        true_region = torch.zeros_like(masks[i], dtype=torch.bool)
                        for c in classes:
                            pred_region |= (preds[i] == c)
                            true_region |= (masks[i] == c)
                        metrics[f'{name}_hd95'].append(hd95(pred_region, true_region))

        # Average and output
        result = {k: np.nanmean([v for v in vals if not np.isnan(v)]) for k, vals in metrics.items()}
        self._save_csv(result)
        return result

    def _save_csv(self, res):
        with open(os.path.join(self.out_dir, "evaluation_metrics.csv"), 'w') as f:
            w = csv.writer(f)
            w.writerow(['Region','Metric','Value'])
            for region, metrics in [('All', ['dice_score','accuracy','precision','recall','fpr']),
                                    ('WT', ['wt_dice','wt_iou','wt_fpr','wt_hd95']),
                                    ('TC', ['tc_dice','tc_iou','tc_fpr','tc_hd95']),
                                    ('ET', ['et_dice','et_iou','et_fpr','et_hd95'])]:
                for m in metrics:
                    w.writerow([region, m.upper(), f"{res.get(m, float('nan')):.4f}"])

    def run(self):
        res = self.evaluate()
        print("\n=== Evaluation Results ===")
        for k, v in res.items():
            print(f"{k}: {v:.4f}")
        print(f"CSV saved to {self.out_dir}/evaluation_metrics.csv")

if __name__ == "__main__":
    Tester().run()
