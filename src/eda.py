import os
import glob
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from PIL import Image
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
# เพิ่ม Root Path ป้องกัน ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_collection import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def calculate_image_hash(image_path):
    hasher = hashlib.md5()
    with open(image_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def run_eda():
    print("[INFO] เริ่มต้นกระบวนการ EDA...")
    images_dir, df_stats = load_dataset()
    
    if not images_dir or not os.path.exists(images_dir):
        print(f"[ERROR] ไม่พบ Path ของโฟลเดอร์รูปภาพ: {images_dir}")
        return

    data_list = []
    hashes = {}
    corrupted_files = []
    duplicate_files = []
    grayscale_files = []

    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    # ใช้ os.walk สแกนทุกโฟลเดอร์และไฟล์ภาพอัตโนมัติ
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                img_path = os.path.join(root, file)
                
                # กำหนด Category จากชื่อโฟลเดอร์ที่เก็บไฟล์ภาพนั้นๆ
                category = os.path.basename(root)

                # 1. ตรวจสอบไฟล์เสีย (Corrupted) และแปลงโหมดภาพ
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    with Image.open(img_path) as img:
                        mode = img.mode
                        
                        # --- วางส่วนนี้แทนที่ img_data = img.convert('RGB') เดิม ---
                        if mode in ('RGBA', 'LA', 'P'):
                            img_data = img.convert('RGBA').convert('RGB')
                        else:
                            img_data = img.convert('RGB')
                            
                        width, height = img.size
                        # --------------------------------------------------------
                except Exception:
                    corrupted_files.append(img_path)
                    continue

                # 2. ตรวจสอบ Grayscale/Non-RGB
                if mode != 'RGB':
                    grayscale_files.append(img_path)

                # 3. ตรวจสอบรูปซ้ำ (Duplicates)
                file_hash = calculate_image_hash(img_path)
                if file_hash in hashes:
                    duplicate_files.append((img_path, hashes[file_hash]))
                else:
                    hashes[file_hash] = img_path

                file_size_kb = os.path.getsize(img_path) / 1024.0
                aspect_ratio = width / height if height > 0 else 0

                img_np = np.array(img_data)
                r_mean = img_np[:, :, 0].mean()
                g_mean = img_np[:, :, 1].mean()
                b_mean = img_np[:, :, 2].mean()

                data_list.append({
                    'file_path': img_path,
                    'category': category,
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'file_size_kb': file_size_kb,
                    'mode': mode,
                    'r_mean': r_mean,
                    'g_mean': g_mean,
                    'b_mean': b_mean,
                    'mean_intensity': img_np.mean(),
                    'std_intensity': img_np.std()
                })

    if not data_list:
        print(f"[ERROR] ไม่พบไฟล์รูปภาพใน {images_dir} กรุณาตรวจสอบโครงสร้างไฟล์")
        return

    df = pd.DataFrame(data_list)
    print(f"\n[SUMMARY] ภาพที่สมบูรณ์: {len(df)} รูป")
    print(f"[SUMMARY] จำนวน Class ที่พบ: {df['category'].nunique()} Class")
    print(f"[SUMMARY] ไฟล์เสีย (Corrupted): {len(corrupted_files)} ไฟล์")
    print(f"[SUMMARY] รูปภาพซ้ำ (Duplicates): {len(duplicate_files)} ไฟล์")
    print(f"[SUMMARY] ภาพ Grayscale/Non-RGB: {len(grayscale_files)} ไฟล์")

    # 1. Class Distribution
    plt.figure(figsize=(12, 6))
    class_counts = df['category'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('Image Count per Class (Class Imbalance Check)')
    plt.xlabel('Cat Breed Category')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"))
    plt.close()

    # 2. Dimensions & Aspect Ratio & File Size
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.scatterplot(data=df, x='width', y='height', hue='category', alpha=0.6, ax=axes[0], legend=False)
    axes[0].set_title('Width vs Height Distribution')
    sns.histplot(df['aspect_ratio'], kde=True, color='purple', ax=axes[1])
    axes[1].set_title('Aspect Ratio Distribution')
    sns.histplot(df['file_size_kb'], kde=True, color='green', ax=axes[2])
    axes[2].set_title('File Size Distribution (KB)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "image_dimensions.png"))
    plt.close()

    # 3. Pixel Intensity Distribution
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df['r_mean'], color='red', label='Red Channel Mean')
    sns.kdeplot(df['g_mean'], color='green', label='Green Channel Mean')
    sns.kdeplot(df['b_mean'], color='blue', label='Blue Channel Mean')
    plt.title('Pixel Intensity Distribution per RGB Channel')
    plt.xlabel('Mean Pixel Value (0-255)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pixel_intensity.png"))
    plt.close()

    # 4. Sample Grid
    unique_cats = df['category'].unique()
    num_cats = len(unique_cats)
    cols = 4
    rows = (num_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()

    for idx, cat in enumerate(unique_cats):
        sample_img_path = df[df['category'] == cat]['file_path'].sample(1).values[0]
        img = Image.open(sample_img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(f"Class: {cat}")
        axes[idx].axis('off')

    for idx in range(num_cats, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sample_grid.png"))
    plt.close()

    print(f"\n[SUCCESS] เซฟรูปภาพทั้งหมดลงใน '{FIGURES_DIR}' เรียบร้อยแล้ว!")

if __name__ == "__main__":
    run_eda()