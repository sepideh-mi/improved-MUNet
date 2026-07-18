# ---- Loss Combinations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0,
                 num_classes=4, class_weights=None, focal_gamma=2.0):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

    def forward(self, pred, target, return_individual=False):
        probs = F.softmax(pred, dim=1)
        dice = self._dice_loss(probs, target)
        miou = self._miou_loss(probs, target)
        boundary = self._boundary_loss(probs, target)
        focal = self._focal_loss(pred, target)
        total = self.alpha * dice + self.beta * miou + self.gamma * boundary + self.delta * focal
        if return_individual:
            return {'total': total, 'dice': dice, 'miou': miou, 'boundary': boundary, 'focal': focal}
        return total

    def _focal_loss(self, pred, target):
        probs = F.softmax(pred, dim=1)
        true_probs = (probs * target).sum(dim=1)  # [B,H,W]
        mod = (1 - true_probs) ** self.focal_gamma
        ce = -torch.log(true_probs + 1e-8)
        loss = mod * ce
        if self.class_weights is not None:
            classes = target.argmax(dim=1)
            w = torch.zeros_like(loss)
            for c in range(self.num_classes):
                w[classes == c] = self.class_weights[c]
            loss = loss * w
        return loss.mean()

    def _dice_loss(self, P, G):
        smooth = 1e-6
        loss = 0.0; weight_sum = 0.0
        for c in range(self.num_classes):
            p = P[:, c].reshape(-1); g = G[:, c].reshape(-1)
            if g.sum() == 0 and c == 0:
                continue
            inter = (p * g).sum(); union = p.sum() + g.sum()
            if union == 0:
                continue
            dice = (2 * inter + smooth) / (union + smooth)
            w = self.class_weights[c] if self.class_weights is not None else 1.0
            loss += (1 - dice) * w
            weight_sum += w
        return loss / max(weight_sum, 1)

    def _miou_loss(self, P, G):
        smooth = 1e-6
        loss = 0.0; weight_sum = 0.0
        for c in range(self.num_classes):
            inter = (P[:, c] * G[:, c]).sum(dim=(1,2))
            union = P[:, c].sum(dim=(1,2)) + G[:, c].sum(dim=(1,2)) - inter
            valid = union > 0
            if not valid.any():
                continue
            iou = (inter[valid] + smooth) / (union[valid] + smooth)
            w = self.class_weights[c] if self.class_weights is not None else 1.0
            loss += (1 - iou.mean()) * w
            weight_sum += w
        return loss / max(weight_sum, 1)

    def _boundary_loss(self, P, G):
        def grad_mag(x):
            sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
            sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
            gx = F.conv2d(x.unsqueeze(1), sobel_x, padding=1)
            gy = F.conv2d(x.unsqueeze(1), sobel_y, padding=1)
            mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
            return (mag / mag.max()).squeeze(1) if mag.max() > 0 else mag.squeeze(1)
        loss = 0.0; count = 0
        for c in range(1, self.num_classes):
            pb = grad_mag(P[:, c]); gb = grad_mag(G[:, c])
            inter = (pb * gb).sum(); union = pb.sum() + gb.sum()
            if union > 0:
                loss += 1 - (2*inter + 1e-6) / (union + 1e-6)
                count += 1
        return loss / max(count, 1)

    # ---------- Metrics (for evaluation) ----------
    def calculate_metrics(self, pred, target):
        probs = F.softmax(pred, dim=1)
        pred_cls = probs.argmax(dim=1)
        target_cls = target.argmax(dim=1)
        metrics = {
            'accuracy': self._acc(pred_cls, target_cls),
            'precision': self._precision(pred_cls, target_cls),
            'recall': self._recall(pred_cls, target_cls),
            'fpr': self._fpr(pred_cls, target_cls),
            'mIoU': self._mean_iou(probs, target),
            'dice_score': self._mean_dice(probs, target),
            'wt_iou': self._region_iou(probs, target, [1,2,3]),
            'tc_iou': self._region_iou(probs, target, [1,3]),
            'et_iou': self._region_iou(probs, target, [3]),
            'wt_dice': self._region_dice(probs, target, [1,2,3]),
            'tc_dice': self._region_dice(probs, target, [1,3]),
            'et_dice': self._region_dice(probs, target, [3]),
            'wt_fpr': self._region_fpr(pred_cls, target_cls, [1,2,3]),
            'tc_fpr': self._region_fpr(pred_cls, target_cls, [1,3]),
            'et_fpr': self._region_fpr(pred_cls, target_cls, [3]),
        }
        return metrics

    # helper metrics
    def _acc(self, p, t): return (p == t).float().mean()
    def _precision(self, p, t):
        per = []
        for c in range(self.num_classes):
            tp = ((p == c) & (t == c)).sum().float()
            pp = (p == c).sum().float()
            per.append(tp / (pp + 1e-6))
        return torch.tensor(per).mean()
    def _recall(self, p, t):
        per = []
        for c in range(self.num_classes):
            tp = ((p == c) & (t == c)).sum().float()
            ap = (t == c).sum().float()
            per.append(tp / (ap + 1e-6))
        return torch.tensor(per).mean()
    def _fpr(self, p, t):
        per = []
        for c in range(self.num_classes):
            fp = ((p == c) & (t != c)).sum().float()
            an = (t != c).sum().float()
            per.append(fp / (an + 1e-6))
        return torch.tensor(per).mean()
    def _mean_iou(self, P, G):
        ious = []
        for c in range(self.num_classes):
            inter = (P[:,c] * G[:,c]).sum(dim=(1,2))
            union = P[:,c].sum(dim=(1,2)) + G[:,c].sum(dim=(1,2)) - inter
            ious.append(((inter + 1e-6) / (union + 1e-6)).mean())
        return torch.tensor(ious).mean()
    def _mean_dice(self, P, G):
        dices = []
        for c in range(self.num_classes):
            inter = (P[:,c] * G[:,c]).sum(dim=(1,2))
            dice = (2*inter + 1e-6) / (P[:,c].sum(dim=(1,2)) + G[:,c].sum(dim=(1,2)) + 1e-6)
            dices.append(dice.mean())
        return torch.tensor(dices).mean()
    def _region_iou(self, P, G, classes):
        pred_region = torch.stack([P[:,c] for c in classes], dim=1).max(dim=1)[0] > 0.5
        target_region = torch.stack([G[:,c] for c in classes], dim=1).max(dim=1)[0] > 0.5
        inter = (pred_region * target_region).sum(dim=(1,2))
        union = pred_region.sum(dim=(1,2)) + target_region.sum(dim=(1,2)) - inter
        return ((inter + 1e-6) / (union + 1e-6)).mean()
    def _region_dice(self, P, G, classes):
        pred_region = torch.stack([P[:,c] for c in classes], dim=1).max(dim=1)[0] > 0.5
        target_region = torch.stack([G[:,c] for c in classes], dim=1).max(dim=1)[0] > 0.5
        inter = (pred_region * target_region).sum(dim=(1,2))
        denom = pred_region.sum(dim=(1,2)) + target_region.sum(dim=(1,2))
        return ((2*inter + 1e-6) / (denom + 1e-6)).mean()
    def _region_fpr(self, p, t, classes):
        pred_region = torch.stack([(p == c) for c in classes], dim=1).any(dim=1)
        target_region = torch.stack([(t == c) for c in classes], dim=1).any(dim=1)
        fp = (pred_region & ~target_region).sum().float()
        an = (~target_region).sum().float()
        return fp / (an + 1e-6)
