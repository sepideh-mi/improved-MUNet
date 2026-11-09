import torch
import torch.nn as nn
import torch.nn.functional as F
from et_focus import ETFocusMixin


class SegmentationLoss(ETFocusMixin, nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, num_classes=4, class_weights=None, 
                 focal_gamma=2.0, et_class_idx=3, et_focus_start_epoch=15, et_focus_max_epoch=70,
                 max_et_weight_multiplier=2.5, min_et_weight_multiplier=1.0):
        """
        UPDATED: Stabilized Dynamic ET Emphasis Strategy
        Args:
            et_focus_start_epoch: 15 (start later for better base learning)
            et_focus_max_epoch: 70 (longer, more gradual progression)
            max_et_weight_multiplier: 2.5 (more conservative maximum)
        """
        super(SegmentationLoss, self).__init__()
        
        self.alpha = alpha    # Dice weight
        self.beta = beta      # mIoU weight  
        self.gamma = gamma    # Boundary weight
        self.delta = delta    # Focal loss weight
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        
        # STABILIZED Dynamic ET Emphasis parameters
        self.et_class_idx = et_class_idx
        self.et_focus_start_epoch = et_focus_start_epoch
        self.et_focus_max_epoch = et_focus_max_epoch
        self.max_et_weight_multiplier = max_et_weight_multiplier
        self.min_et_weight_multiplier = min_et_weight_multiplier
        self.current_epoch = 0
        
        # Track ET performance for adaptive focusing
        self.et_dice_history = []
        self.et_focus_intensity = min_et_weight_multiplier
        
        print(f"✅ Loss weights - Dice: {alpha}, mIoU: {beta}, Boundary: {gamma}, Focal: {delta}")
        print(f"🎯 STABILIZED Dynamic ET Emphasis - Start: {et_focus_start_epoch}, Max: {et_focus_max_epoch}")
        print(f"🎯 ET Weight Range: {min_et_weight_multiplier} -> {max_et_weight_multiplier}")
        if self.class_weights is not None:
            print(f"✅ Class weights: {self.class_weights}")
        print(f"✅ Focal loss gamma: {focal_gamma}")

    def forward(self, predictions, targets, return_individual=False):
        """
        Forward pass with STABILIZED Dynamic ET Emphasis Strategy
        """
        # Apply softmax to get probabilities from raw logits
        probs = F.softmax(predictions, dim=1)
        
        # Calculate individual losses with dynamic ET emphasis
        dice_loss_val = self._dice_loss(probs, targets) * self.alpha
        miou_loss = self._mIoU_loss(probs, targets) * self.beta
        boundary_loss_val = self._boundary_loss(probs, targets) * self.gamma
        focal_loss_val = self._focal_loss(predictions, targets) * self.delta
        
        # Combine losses
        total_loss = dice_loss_val + miou_loss + boundary_loss_val + focal_loss_val
        
        if return_individual:
            loss_dict = {
                'total_loss': total_loss,
                'dice_loss': dice_loss_val,
                'mIoU_loss': miou_loss,
                'boundary_loss': boundary_loss_val,
                'focal_loss': focal_loss_val,
                'et_focus_intensity': torch.tensor(self.et_focus_intensity, device=total_loss.device)
            }
            
            # Debug: Print relative contributions
            total_val = total_loss.item()
            if total_val > 0:
                print(f"📊 Loss contributions - Dice: {dice_loss_val.item()/total_val:.2%}, "
                      f"mIoU: {miou_loss.item()/total_val:.2%}, "
                      f"Boundary: {boundary_loss_val.item()/total_val:.2%}, "
                      f"Focal: {focal_loss_val.item()/total_val:.2%}")
                print(f"🎯 ET Focus Intensity: {self.et_focus_intensity:.2f}x")
            
            return loss_dict
        
        return total_loss

    def _focal_loss(self, predictions, targets):
        """
        Focal Loss with STABILIZED Dynamic ET Emphasis
        """
        # Convert to probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Get the true class probabilities
        true_class_probs = torch.sum(probs * targets, dim=1)  # [B, H, W]
        
        # Calculate the modulating factor
        modulating_factor = (1 - true_class_probs) ** self.focal_gamma
        
        # Calculate cross entropy
        ce_loss = -torch.log(true_class_probs + 1e-8)
        
        # Apply focal weighting
        focal_loss_per_pixel = modulating_factor * ce_loss
        
        # Apply dynamic class balancing weights with ET emphasis
        dynamic_weights = self._get_dynamic_et_weights(self.class_weights)
        
        # Get the dominant class for each pixel in target
        target_classes = torch.argmax(targets, dim=1)  # [B, H, W]
        
        # Create weight map based on target classes
        weight_map = torch.zeros_like(focal_loss_per_pixel)
        for class_idx in range(self.num_classes):
            class_mask = (target_classes == class_idx)
            if class_mask.any():
                weight_map[class_mask] = dynamic_weights[class_idx]
        
        # Apply per-sample ET difficulty weighting
        et_difficulty_weights = self._get_et_difficulty_weights(predictions, targets)
        batch_size = predictions.size(0)
        for i in range(batch_size):
            et_mask = (target_classes[i] == self.et_class_idx)
            if et_mask.any():
                # Increase weights for difficult ET samples
                difficulty_weight = 1.0 + et_difficulty_weights[i]
                weight_map[i][et_mask] *= difficulty_weight
        
        focal_loss_per_pixel = focal_loss_per_pixel * weight_map
        
        # Average over spatial dimensions and batch
        focal_loss = torch.mean(focal_loss_per_pixel)
        
        return focal_loss

    def _dice_loss(self, P, G):
        """Dice loss with dynamic ET emphasis"""
        smooth = 1e-6
        dice_loss = 0.0
        total_weight = 0.0
        
        # Get dynamic weights with ET emphasis
        dynamic_weights = self._get_dynamic_et_weights(self.class_weights)
        
        for class_idx in range(self.num_classes):
            pred_flat = P[:, class_idx].contiguous().view(-1)
            target_flat = G[:, class_idx].contiguous().view(-1)
            
            # Skip if no target and it's background (common case)
            if class_idx == 0 and target_flat.sum() == 0:
                continue
                
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            
            # Avoid division by zero
            if union == 0:
                continue
                
            dice = (2. * intersection + smooth) / (union + smooth)
            
            # Apply dynamic class weighting with ET emphasis
            weight = dynamic_weights[class_idx]
                
            dice_loss += (1.0 - dice) * weight
            total_weight += weight
        
        return dice_loss / max(total_weight, 1)

    def _mIoU_loss(self, P, G):
        """mIoU loss with dynamic ET emphasis"""
        smooth = 1e-6
        iou_loss = 0.0
        total_weight = 0.0
        
        # Get dynamic weights with ET emphasis
        dynamic_weights = self._get_dynamic_et_weights(self.class_weights)
        
        for class_idx in range(self.num_classes):
            # Calculate IoU for this class
            intersection = torch.sum(P[:, class_idx] * G[:, class_idx], dim=(1, 2))
            union = torch.sum(P[:, class_idx], dim=(1, 2)) + torch.sum(G[:, class_idx], dim=(1, 2)) - intersection
            
            # Handle empty unions
            valid_mask = union > 0
            if not valid_mask.any():
                continue
                
            iou = (intersection + smooth) / (union + smooth)
            class_iou_loss = 1.0 - iou[valid_mask].mean()
            
            # Apply dynamic class weighting with ET emphasis
            weight = dynamic_weights[class_idx]
                
            iou_loss += class_iou_loss * weight
            total_weight += weight
        
        return iou_loss / max(total_weight, 1)

    def _boundary_loss(self, P, G):
        """CORRECTED: Proper differentiable boundary loss"""
        boundary_loss = 0.0
        count = 0
        
        # Calculate boundary loss for each tumor class
        for class_idx in range(1, self.num_classes):  # Skip background
            # Get probability maps for this class
            P_c = P[:, class_idx]  # [B, H, W]
            G_c = G[:, class_idx]  # [B, H, W]
            
            # Compute gradients to find boundaries (differentiable)
            pred_boundary = self._compute_gradient_magnitude(P_c)
            target_boundary = self._compute_gradient_magnitude(G_c)
            
            # Calculate boundary similarity using dice on boundaries
            smooth = 1e-6
            intersection = (pred_boundary * target_boundary).sum()
            union = pred_boundary.sum() + target_boundary.sum()
            
            if union > 0:
                boundary_dice = (2. * intersection + smooth) / (union + smooth)
                class_boundary_loss = 1.0 - boundary_dice
            else:
                class_boundary_loss = torch.tensor(0.0, device=P.device)
            
            boundary_loss += class_boundary_loss
            count += 1
        
        return boundary_loss / max(count, 1)

    def _compute_gradient_magnitude(self, x):
        """Compute gradient magnitude using Sobel filters (differentiable)"""
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, H, W]
        
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        # Compute gradients
        grad_x = F.conv2d(x, sobel_x, padding=1, stride=1)
        grad_y = F.conv2d(x, sobel_y, padding=1, stride=1)
        
        # Compute gradient magnitude
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # Normalize to [0, 1]
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        return gradient_magnitude.squeeze(1)  # [B, H, W]
