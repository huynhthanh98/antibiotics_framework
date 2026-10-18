"""
Dataset classes cho Task 1 va Task 2.

Task 1: Load masked cell images tu thu muc masked/.
Task 2: Sliding window toan bo tren original images, loc patch theo std dev (uint16).

Cau truc thu muc (*_joblib):
    {drug_dir}/{split}/{label_str}/{masked|original}/{cell_file}
    label_str : 'resistant' (label=1) hoac 'sensitive' (label=0)
    split     : 'train' | 'val' | 'test'

CSV columns (*_joblib):
    split, drug, isolate, iso_id, anti_id, image_name,
    label, resistant, cell_file, mask_file, original_tiff, original_mask
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def open_as_rgb(path: str) -> Image.Image:
    """
    Mở ảnh và trả về PIL Image RGB 8-bit.

    Xử lý đúng ảnh 16-bit (uint16 TIFF — phổ biến trong ảnh kính hiển vi):
      - Đọc giá trị pixel thực (uint16, range ~0–65535)
      - Normalize về [0, 255] dùng percentile 1–99 (tránh bị saturate bởi outlier)
      - Convert sang RGB uint8

    Ảnh 8-bit thông thường: convert('RGB') bình thường.
    """
    img = Image.open(path)

    # Kiểm tra 16-bit (mode 'I;16', 'I;16B', hoặc 'I')
    if img.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
        arr = np.array(img, dtype=np.float32)
        # Percentile clip để tránh hot/dead pixel ảnh hưởng scale
        p_low  = float(np.percentile(arr, 1))
        p_high = float(np.percentile(arr, 99))
        if p_high > p_low:
            arr = np.clip(arr, p_low, p_high)
            arr = (arr - p_low) / (p_high - p_low) * 255.0
        else:
            arr = np.zeros_like(arr)
        arr = arr.astype(np.uint8)
        img_8bit = Image.fromarray(arr, mode='L')
        return img_8bit.convert('RGB')

    return img.convert('RGB')


def _apply_mask_erosion_pil(img: Image.Image, n_px: int) -> Image.Image:
    """
    Erode vùng tế bào (non-background) vào trong n_px pixel từ boundary.
    Dùng PIL MinFilter(3) mỗi iteration ≈ 1px erosion.
    Input/Output: PIL Image RGB.
    """
    if n_px <= 0:
        return img
    arr = np.array(img)
    fg = (arr.max(axis=2) > 20).astype(np.uint8) * 255
    mask_pil = Image.fromarray(fg)
    for _ in range(n_px):
        mask_pil = mask_pil.filter(ImageFilter.MinFilter(3))
    eroded = np.array(mask_pil) > 0
    result = arr.copy()
    result[~eroded] = 0
    return Image.fromarray(result)


class RandomMaskErosion:
    """
    Augmentation: ngẫu nhiên erode boundary cell mask 1–max_px pixels.
    Đặt TRƯỚC Resize/ToTensor — nhận và trả về PIL Image RGB.

    Mô phỏng mask không hoàn hảo khi segmentation bị under-estimate boundary.
    Xác suất áp dụng: p (mặc định 0.5).
    """

    def __init__(self, max_px: int = 3, p: float = 0.5):
        self.max_px = max_px
        self.p      = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        n_px = random.randint(1, self.max_px)
        return _apply_mask_erosion_pil(img, n_px)

    def __repr__(self):
        return f"RandomMaskErosion(max_px={self.max_px}, p={self.p})"


def get_train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform_with_erosion_aug(
    img_size: int = 224, max_erosion_px: int = 3
) -> transforms.Compose:
    """
    Train transform với RandomMaskErosion augmentation.
    Erosion áp dụng TRƯỚC resize để giữ đúng tỉ lệ pixel boundary bị mất.
    """
    return transforms.Compose([
        RandomMaskErosion(max_px=max_erosion_px, p=0.5),
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def label_to_str(label: int) -> str:
    return 'resistant' if label == 1 else 'sensitive'


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hoá DataFrame về schema nội bộ thống nhất:
        resistance_label, filename, original_file

    Hỗ trợ cả hai schema:
      - *_proper : resistance_label, filename, original_file
      - *_joblib : resistant, cell_file, original_tiff
    """
    df = df.copy()
    if 'resistance_label' not in df.columns:
        df['resistance_label'] = df['resistant'].astype(int)
    if 'filename' not in df.columns:
        df['filename'] = df['cell_file']
    if 'original_file' not in df.columns:
        df['original_file'] = df['original_tiff']
    return df


# =============================================================================
# TASK 1: Masked cell images
# =============================================================================

class Task1Dataset(Dataset):
    """
    Dataset cho Task 1: load cell images.
    Moi sample la mot cell image:
        {drug_dir}/{split}/{resistant|sensitive}/{image_folder}/{cell_file}
    """

    def __init__(self, df: pd.DataFrame, drug_dir: str, split: str, 
                 image_folder: str = 'masked', transform=None):
        self.drug_dir     = drug_dir
        self.split        = split
        self.image_folder = image_folder
        self.transform    = transform

        self.paths          = []
        self.labels         = []
        self.original_files = []
        self.filenames      = []
        self._build(_normalize_df(df))

    def _build(self, df: pd.DataFrame):
        missing = 0
        for _, row in df.iterrows():
            label  = int(row['resistance_label'])
            fname  = row['filename']
            orig_f = row['original_file']
            path   = os.path.join(
                self.drug_dir, self.split, label_to_str(label), self.image_folder, fname
            )
            if not os.path.exists(path):
                missing += 1
                continue
            self.paths.append(path)
            self.labels.append(label)
            self.original_files.append(orig_f)
            self.filenames.append(fname)
        if missing > 0:
            print(f"  [Task1Dataset] Warning: {missing} files not found, skipped.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        label = self.labels[idx]
        orig  = self.original_files[idx]
        fname = self.filenames[idx]
        try:
            # Dùng open_as_rgb để TIFF 16-bit (I;16) được scale hợp lý (percentile),
            # giống Task 2 — tránh ép tuyến tính /65535 làm tương phản kém.
            img = open_as_rgb(path)
        except Exception as e:
            print(f"  [Task1Dataset] Read error {path}: {e}")
            img = Image.new('RGB', (224, 224), color=0)
        if self.transform:
            img = self.transform(img)
        return img, label, orig, fname


# =============================================================================
# TASK 2 - Train: Sliding window + content filter (uint16 std dev)
# =============================================================================

class Task2TrainDataset(Dataset):
    """
    Dataset cho Task 2 - Training.

    Cat TAT CA patch patch_size x patch_size (stride=patch_size, khong overlap)
    tu moi anh goc bang sliding window. Loc bo cac patch background bang cach
    doc raw pixel (16-bit hoac 8-bit) va tinh std dev:
        - Neu std >= content_std_threshold -> giu lai de train
        - Nguoc lai -> bo qua (background)

    Nguong content_std_threshold phu thuoc vao kieu du lieu:
        - Anh 16-bit (uint16): dung ~1000 (default)
        - Anh 8-bit (uint8 RGB): dung ~5

    Trong __getitem__, doc lai anh RGB va cat patch, apply augmentation.
    """

    def __init__(self, df: pd.DataFrame, drug_dir: str, split: str = 'train',
                 patch_size: int = 224, content_std_threshold: float = 1000.0,
                 transform=None):
        self.patch_size            = patch_size
        self.content_std_threshold = content_std_threshold
        self.transform             = transform
        # List of (img_path, label, orig_file, coords)
        # coords = (y, x) or None (neu anh nho hon patch_size)
        self.samples = []
        self._build(_normalize_df(df), drug_dir, split)

    # ------------------------------------------------------------------
    def _build(self, df: pd.DataFrame, drug_dir: str, split: str):
        grouped       = df.groupby('original_file')
        n_imgs        = len(grouped)
        missing       = 0
        total_patches = 0
        kept_patches  = 0

        print(f"  [Task2TrainDataset] Scanning {n_imgs} original images "
              f"(content_std_threshold={self.content_std_threshold:.0f})...")

        for i, (orig_file, group) in enumerate(grouped, 1):
            label    = int(group['resistance_label'].iloc[0])
            basename = os.path.basename(orig_file)
            img_path = os.path.join(drug_dir, split, label_to_str(label), 'original', basename)
            if not os.path.exists(img_path):
                img_path = self._find_original(drug_dir, split, basename)
            if img_path is None:
                missing += 1
                continue

            try:
                # Doc raw pixel de tinh std dev (tranh mat thong tin do clip)
                raw_img = Image.open(img_path)
                arr     = np.array(raw_img, dtype=np.float32)
                # Neu RGB (3 kenh) -> chuyen grayscale de tinh std
                if arr.ndim == 3:
                    arr = arr.mean(axis=2)
                h, w = arr.shape
                ps   = self.patch_size

                ys = list(range(0, h - ps + 1, ps))
                xs = list(range(0, w - ps + 1, ps))

                if not ys or not xs:
                    # Anh nho hon patch -> dung toan bo anh
                    self.samples.append((img_path, label, orig_file, None))
                    kept_patches  += 1
                    total_patches += 1
                    continue

                for y in ys:
                    for x in xs:
                        total_patches += 1
                        if arr[y:y + ps, x:x + ps].std() >= self.content_std_threshold:
                            self.samples.append((img_path, label, orig_file, (y, x)))
                            kept_patches += 1

            except Exception as e:
                print(f"  [Task2TrainDataset] Read error {img_path}: {e}")

            if i % 50 == 0 or i == n_imgs:
                pct = 100.0 * kept_patches / max(total_patches, 1)
                print(f"    {i}/{n_imgs} images done, kept {kept_patches}/{total_patches} patches ({pct:.1f}%)")

        if missing > 0:
            print(f"  [Task2TrainDataset] Warning: {missing} original images not found.")
        pct_final = 100.0 * kept_patches / max(total_patches, 1)
        print(f"  [Task2TrainDataset] Final: {kept_patches}/{total_patches} patches kept ({pct_final:.1f}%)")

    # ------------------------------------------------------------------
    def _find_original(self, drug_dir: str, split: str, basename: str):
        for lbl in ['resistant', 'sensitive']:
            p = os.path.join(drug_dir, split, lbl, 'original', basename)
            if os.path.exists(p):
                return p
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, orig_file, coords = self.samples[idx]
        ps = self.patch_size
        try:
            img = open_as_rgb(img_path)
        except Exception as e:
            print(f"  [Task2TrainDataset] Read error {img_path}: {e}")
            img = Image.new('RGB', (ps, ps), 0)
            if self.transform:
                img = self.transform(img)
            return img, label, orig_file, f"{orig_file}_{idx}"

        if coords is not None:
            y, x = coords
            img  = img.crop((x, y, x + ps, y + ps))
        else:
            img = img.resize((ps, ps), Image.BILINEAR)

        if self.transform:
            img = self.transform(img)

        return img, label, orig_file, f"{orig_file}_{coords}"


# =============================================================================
# TASK 2 - Test: Sliding window (stride=patch_size, non-overlapping)
# =============================================================================

class Task2TestDataset(Dataset):
    """
    Dataset cho Task 2 - Testing: sliding window khong overlap.

    Cat toan bo patch patch_size x patch_size (stride=patch_size) tu anh goc.
    KHONG loc patch (giu tat ca de danh gia toan bo anh).
    Ket qua duoc tong hop o image-level bang Soft Vote (average prob).
    """

    def __init__(self, df: pd.DataFrame, drug_dir: str, split: str = 'test',
                 patch_size: int = 224, stride: int = 224, transform=None):
        self.patch_size = patch_size
        self.stride     = stride
        self.transform  = transform
        self.samples    = []
        self._build(_normalize_df(df), drug_dir, split)

    def _build(self, df: pd.DataFrame, drug_dir: str, split: str):
        grouped = df.groupby('original_file')
        missing = 0
        for orig_file, group in grouped:
            label    = int(group['resistance_label'].iloc[0])
            basename = os.path.basename(orig_file)
            img_path = os.path.join(drug_dir, split, label_to_str(label), 'original', basename)
            if not os.path.exists(img_path):
                img_path = self._find_original(drug_dir, split, basename)
            if img_path is None:
                missing += 1
                continue
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                ps = self.patch_size
                st = self.stride
                ys = list(range(0, h - ps + 1, st))
                xs = list(range(0, w - ps + 1, st))
                if not ys or not xs:
                    self.samples.append((img_path, label, orig_file, None))
                else:
                    for y in ys:
                        for x in xs:
                            self.samples.append((img_path, label, orig_file, (y, x)))
            except Exception as e:
                print(f"  [Task2TestDataset] Read size error {img_path}: {e}")
        if missing > 0:
            print(f"  [Task2TestDataset] Warning: {missing} original images not found.")

    def _find_original(self, drug_dir: str, split: str, basename: str):
        for lbl in ['resistant', 'sensitive']:
            p = os.path.join(drug_dir, split, lbl, 'original', basename)
            if os.path.exists(p):
                return p
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, orig_file, coords = self.samples[idx]
        ps = self.patch_size
        try:
            img = open_as_rgb(img_path)
        except Exception as e:
            print(f"  [Task2TestDataset] Read error {img_path}: {e}")
            img = Image.new('RGB', (ps, ps), 0)
            if self.transform:
                img = self.transform(img)
            return img, label, orig_file, str(idx)

        if coords is not None:
            y, x = coords
            img  = img.crop((x, y, x + ps, y + ps))
        else:
            img = img.resize((ps, ps), Image.BILINEAR)

        if self.transform:
            img = self.transform(img)

        return img, label, orig_file, f"{orig_file}_{coords}"


# =============================================================================
# TASK 2 - Precomputed Patches (load từ folder đã export)
# =============================================================================

class Task2PatchDataset(Dataset):
    """
    Dataset cho Task 2 sử dụng patches đã được export trước.

    Cấu trúc thư mục patches:
        {patches_dir}/{split}/
            resistant/*.png
            sensitive/*.png

    Nhanh hơn nhiều so với Task2TrainDataset vì không cần:
      - Đọc ảnh gốc 16-bit
      - Tính std để lọc background
      - Cắt patch on-the-fly
    """

    def __init__(self, patches_dir: str, split: str, transform=None):
        """
        Parameters
        ----------
        patches_dir : đường dẫn đến thư mục patches của drug
                      (e.g., /home/.../task2_patches/CIP)
        split       : 'train', 'val', hoặc 'test'
        transform   : torchvision transforms
        """
        self.patches_dir = patches_dir
        self.split       = split
        self.transform   = transform

        self.paths          = []
        self.labels         = []
        self.original_files = []
        self.filenames      = []

        self._build()

    def _build(self):
        split_dir = os.path.join(self.patches_dir, self.split)
        if not os.path.exists(split_dir):
            print(f"  [Task2PatchDataset] WARNING: {split_dir} does not exist!")
            return

        for label_int, label_str in [(1, 'resistant'), (0, 'sensitive')]:
            label_dir = os.path.join(split_dir, label_str)
            if not os.path.exists(label_dir):
                continue

            for fname in os.listdir(label_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                path = os.path.join(label_dir, fname)
                # Extract original file name from patch name
                # Format: {stem}_y{y:04d}_x{x:04d}.png or {stem}_full.png
                parts = fname.rsplit('_', 2)
                if len(parts) >= 3 and parts[-2].startswith('y') and parts[-1].startswith('x'):
                    orig_stem = '_'.join(parts[:-2])
                elif fname.endswith('_full.png'):
                    orig_stem = fname[:-9]  # remove '_full.png'
                else:
                    orig_stem = os.path.splitext(fname)[0]

                self.paths.append(path)
                self.labels.append(label_int)
                self.original_files.append(orig_stem)
                self.filenames.append(fname)

        print(f"  [Task2PatchDataset] Loaded {len(self.paths):,} patches from {split_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        label = self.labels[idx]
        orig  = self.original_files[idx]
        fname = self.filenames[idx]

        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"  [Task2PatchDataset] Read error {path}: {e}")
            img = Image.new('RGB', (224, 224), color=0)

        if self.transform:
            img = self.transform(img)

        return img, label, orig, fname

    @property
    def samples(self):
        """Compatibility với code khác cần .samples attribute."""
        return list(zip(self.paths, self.labels))


# =============================================================================
# Factory
# =============================================================================

def build_datasets(task: int, train_df: pd.DataFrame, test_df: pd.DataFrame,
                   drug_dir: str, cfg) -> dict:
    """
    Tao train_dataset va test_dataset cho mot fold cu the.

    Returns dict voi keys 'train' va 'test'.
    """
    train_transform = get_train_transform(cfg.img_size)
    test_transform  = get_test_transform(cfg.img_size)

    if task == 1:
        train_ds = Task1Dataset(train_df, drug_dir, 'train', 
                               image_folder=cfg.task1_image_folder, transform=train_transform)
        test_ds  = Task1Dataset(test_df,  drug_dir, 'test',  
                               image_folder=cfg.task1_image_folder, transform=test_transform)

    elif task == 2:
        train_ds = Task2TrainDataset(
            train_df, drug_dir, split='train',
            patch_size=cfg.patch_size,
            content_std_threshold=getattr(cfg, 'task2_content_std_threshold', 1000.0),
            transform=train_transform,
        )
        test_ds = Task2TestDataset(
            test_df, drug_dir, split='test',
            patch_size=cfg.patch_size,
            stride=cfg.test_stride,
            transform=test_transform,
        )
    else:
        raise ValueError(f"Task must be 1 or 2, got: {task}")

    return {'train': train_ds, 'test': test_ds}
