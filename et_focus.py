import torch
import torch.nn.functional as F


class ETFocusMixin:
    """Dynamic ET emphasis: epoch schedule, performance feedback, per-sample difficulty.
    Used by SegmentationLoss (loss.py)."""

    def set_current_epoch(self, epoch):
        """Update current epoch for dynamic ET emphasis scheduling"""
        self.current_epoch = epoch
        self._update_et_focus_intensity()

    def update_et_performance(self, et_dice_score):
        """Update ET performance history for adaptive focusing"""
        self.et_dice_history.append(et_dice_score)
        # Keep longer history for more stable calculations
        if len(self.et_dice_history) > 15:
            self.et_dice_history.pop(0)

    def _update_et_focus_intensity(self):
        """Calculate current ET focus intensity based on epoch and performance"""
        if self.current_epoch < self.et_focus_start_epoch:
            # Phase 1: Balanced training
            self.et_focus_intensity = self.min_et_weight_multiplier
        else:
            # Phase 2: Progressive ET focusing with STABILIZED progression
            progress = min(1.0, (self.current_epoch - self.et_focus_start_epoch) / 
                          (self.et_focus_max_epoch - self.et_focus_start_epoch))
            
            # Smoother progression using ease-in-out
            smooth_progress = self._smooth_step(progress)
            
            # Base progression
            base_intensity = self.min_et_weight_multiplier + smooth_progress * (
                self.max_et_weight_multiplier - self.min_et_weight_multiplier)
            
            # STABILIZED Performance-based adjustment
            performance_factor = self._calculate_performance_factor()
            
            # Combine with more conservative bounds
            self.et_focus_intensity = base_intensity * performance_factor
            
            # Tighter clamping to prevent extreme values
            self.et_focus_intensity = max(self.min_et_weight_multiplier, 
                                        min(self.et_focus_intensity, 
                                            self.max_et_weight_multiplier * 1.1))  # Reduced from 1.2
        
        print(f"📈 Epoch {self.current_epoch}: ET focus intensity = {self.et_focus_intensity:.2f}x")

    def _smooth_step(self, t):
        """Smooth step function for more gradual progression"""
        return t * t * (3 - 2 * t)

    def _calculate_performance_factor(self):
        """STABILIZED: More conservative performance-based adjustment"""
        if len(self.et_dice_history) < 8:  # Require more history for stability
            return 1.0
    
        # Use smoother moving averages
        recent_performance = sum(self.et_dice_history[-5:]) / 5
        older_performance = sum(self.et_dice_history[-10:-5]) / 5 if len(self.et_dice_history) >= 10 else recent_performance
        
        if older_performance == 0:
            return 1.0
        
        improvement_ratio = recent_performance / older_performance
        
        # MORE CONSERVATIVE adjustments
        if improvement_ratio < 0.98:  # Only adjust if significant degradation
            return 1.1  # Small increase (reduced from 1.2)
        elif improvement_ratio > 1.02:  # Good improvement
            return 0.95  # Small decrease (reduced from 0.8)
        else:  # Stable performance
            return 1.0

    def _get_dynamic_et_weights(self, base_weights):
        """Apply dynamic ET emphasis to class weights"""
        if base_weights is None:
            base_weights = torch.ones(self.num_classes, device=next(self.parameters()).device)
        
        dynamic_weights = base_weights.clone()
        dynamic_weights[self.et_class_idx] *= self.et_focus_intensity
        
        return dynamic_weights

    def _get_et_difficulty_weights(self, predictions, targets):
        """Calculate per-sample ET difficulty weights"""
        with torch.no_grad():
            probs = F.softmax(predictions, dim=1) #ET predictions confidence
            et_probs = probs[:, self.et_class_idx]  #models ET predictions
            et_targets = targets[:, self.et_class_idx]  #Ground Truth ET
            
            # Calculate ET prediction confidence: high predictions=high confidence
            et_confidence = torch.where(et_targets > 0.5, et_probs, 1 - et_probs)
            
            # Difficulty: 1-confidence(low confidence = high difficulty)
            et_difficulty = 1.0 - et_confidence
            
            # Apply smoothing and normalization per sample [0,1]
            et_difficulty = et_difficulty.mean(dim=[1, 2])  # Average over spatial dimensions
            et_difficulty = (et_difficulty - et_difficulty.min()) / (et_difficulty.max() - et_difficulty.min() + 1e-8)
            
            # Scale by current focus intensity
            et_difficulty = et_difficulty * self.et_focus_intensity
            
            return et_difficulty
