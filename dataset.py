```python
import os, glob, random, numpy as np, nibabel as nib, torch
from torch.utils.data import Dataset
import pandas as pd

# ----- class weights via median frequency balancing -----
def calculate_class_weights(class_counts):
    counts = torch.tensor(class_counts, dtype=torch.float32) + 1e-6
    non_zero = counts[counts > 0]
    median = torch.median(non_zero) if len(non_zero) else 1.0
    weights = median / counts
    weights /= weights.mean()
    return torch.clamp(weights, 0.5, 3.0)

# ----- dataset
class BraTSDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, out_channels=4,
                 slice_selection_strategy="largest_tumor",
                 phase="train", use_augmentation=False):
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.out_channels = out_channels
        self.strategy = slice_selection_strategy
        self.phase = phase
        self.use_aug = use_augmentation

        self.patients = self._get_patients()
        self.grades = self._load_grades()
        self.file_cache = self._preload_paths()
        self.slice_map = self._compute_optimal_slices()
        self.class_counts, self.class_weights = self._compute_weights(subset_size=50)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        case_id = os.path.basename(self.patients[idx])
        grade = self.grades.get(case_id, "Unknown")
        mod_paths, seg_path = self.file_cache.get(case_id, (None, None))
        if mod_paths is None or len(mod_paths) != 4:
            return None

        seg = nib.load(seg_path).get_fdata().astype(np.uint8)
        seg[seg == 4] = 3
        seg = np.clip(seg, 0, self.out_channels - 1)
        slice_idx = self.slice_map.get(case_id, seg.shape[2] // 2)

        slices = []
        for p in mod_paths:
            vol = nib.load(p).get_fdata().astype(np.float32)
            sl = vol[:, :, slice_idx]
            sl = np.clip(sl, 0, None)
            p99 = np.percentile(sl, 99.5)
            if p99 > 0:
                sl = np.clip(sl, 0, p99) / p99
            slices.append(sl)

        image = torch.from_numpy(np.stack(slices, axis=0)).float()
        mask  = torch.from_numpy(seg[:, :, slice_idx]).long()

        if self.use_aug:
            image, mask = self._augment(image, mask)
        return image, mask, grade

    # ----- get patients
    def _get_patients(self):
        all_dirs = sorted(glob.glob(os.path.join(self.root_dir, '*')))
        if self.csv_path and os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            id_col = next((c for c in df.columns if 'ID' in c or 'subject' in c.lower()), None)
            if id_col:
                ids = set(df[id_col].astype(str))
                return [d for d in all_dirs if os.path.basename(d) in ids]
        return all_dirs
    # ----- get grades
    def _load_grades(self):
        grades = {}
        if self.csv_path and os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            id_col = next((c for c in df.columns if 'ID' in c or 'subject' in c.lower()), None)
            grade_col = next((c for c in df.columns if 'grade' in c.lower() or 'hgg' in c.lower()), None)
            if id_col and grade_col:
                for _, row in df.iterrows():
                    g = str(row[grade_col]).upper()
                    grades[str(row[id_col])] = "HGG" if "HGG" in g else "LGG" if "LGG" in g else "Unknown"
        return grades

    # ---- preload path 

    def _compute_optimal_slices(self):
        slice_map = {}
        for case, (_, seg_path) in self.file_cache.items():
            seg = nib.load(seg_path).get_fdata().astype(np.uint8)
            seg[seg == 4] = 3
            seg = np.clip(seg, 0, self.out_channels - 1)
            best_slice = 0
            best_area = 0
            for z in range(seg.shape[2]):
                area = np.sum(seg[:, :, z] > 0)
                if area > best_area:
                    best_area = area
                    best_slice = z
            slice_map[case] = best_slice
        return slice_map

    def _compute_weights(self, subset_size=50):
        counts = np.zeros(self.out_channels, dtype=np.int64)
        for _ in range(min(subset_size, len(self))):
            idx = random.randint(0, len(self) - 1)
            sample = self[idx]
            if sample is None:
                continue
            _, mask, _ = sample
            counts += np.bincount(mask.flatten().numpy(), minlength=self.out_channels)
        if counts.sum() == 0:
            return None, torch.ones(self.out_channels, dtype=torch.float32)
        weights = calculate_class_weights(counts)
        return counts, weights
    # ---- augmentation 
    def _augment(self, img, mask):
        if random.random() < 0.5:
            img, mask = torch.flip(img, [-1]), torch.flip(mask, [-1])
        if random.random() < 0.5:
            img, mask = torch.flip(img, [-2]), torch.flip(mask, [-2])
        if random.random() < 0.3:
            k = random.randint(0, 3)
            img, mask = torch.rot90(img, k, [-2, -1]), torch.rot90(mask, k, [-2, -1])
        return img, mask
