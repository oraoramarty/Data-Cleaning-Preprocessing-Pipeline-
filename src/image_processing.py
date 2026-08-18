import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def suggest_target_size(df_stats: pd.DataFrame) -> tuple:
    median_w = int(df_stats['width'].median())
    median_h = int(df_stats['height'].median())

    median_w -= median_w % 2
    median_h -= median_h % 2

    print(f"[INFO] ขนาดภาพ median จาก dataset: ({median_w}, {median_h})")
    return (median_w, median_h)


def resize_image(image: Image.Image, target_size: tuple = (224, 224),
                  keep_aspect_ratio: bool = True) -> Image.Image:
    if not keep_aspect_ratio:
        return image.resize(target_size, Image.Resampling.LANCZOS)

    target_w, target_h = target_size
    orig_w, orig_h = image.size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new('RGB', target_size, (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas


def resize_dataset(df: pd.DataFrame, output_dir: str,
                    target_size: tuple = (224, 224),
                    keep_aspect_ratio: bool = True) -> pd.DataFrame:
    records = []

    for _, row in df.iterrows():
        src_path = row['file_path']
        category = row['category']

        dst_folder = os.path.join(output_dir, category)
        os.makedirs(dst_folder, exist_ok=True)
        dst_path = os.path.join(dst_folder, os.path.basename(src_path))

        try:
            with Image.open(src_path) as img:
                img = img.convert('RGB')
                resized_img = resize_image(img, target_size, keep_aspect_ratio)
                resized_img.save(dst_path)
        except Exception as e:
            print(f"[WARNING] Resize ไม่สำเร็จ: {src_path} ({e})")
            continue

        records.append({
            'file_path': dst_path,
            'category': category,
            'width': target_size[0],
            'height': target_size[1],
        })

    result_df = pd.DataFrame(records)
    print(f"[SUCCESS] Resize เสร็จสิ้น: {len(result_df)}/{len(df)} รูป -> {output_dir}")
    return result_df


def show_resize_before_after(df: pd.DataFrame, target_size: tuple = (224, 224),
                              keep_aspect_ratio: bool = True,
                              n_samples: int = 4,
                              save_name: str = "resize_before_after.png"):
    samples = df.sample(min(n_samples, len(df)))

    fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))
    if len(samples) == 1:
        axes = axes.reshape(2, 1)

    for i, (_, row) in enumerate(samples.iterrows()):
        with Image.open(row['file_path']) as img:
            img = img.convert('RGB')
            resized = resize_image(img, target_size, keep_aspect_ratio)

            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Before\n{img.size}")
            axes[0, i].axis('off')

            axes[1, i].imshow(resized)
            axes[1, i].set_title(f"After\n{resized.size}")
            axes[1, i].axis('off')

    plt.suptitle("Resize: Before vs After")
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"[SUCCESS] เซฟภาพเปรียบเทียบ Before/After ที่: {save_path}")


if __name__ == "__main__":
    pass