import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageEnhance, ImageOps

FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

VALID_EXT = (".jpg", ".jpeg", ".png")


def load_images_from_folder(input_dir: str) -> pd.DataFrame:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {input_dir}")

    records = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(VALID_EXT):
                file_path = os.path.join(root, fname)
                category = os.path.basename(root)
                records.append({'file_path': file_path, 'category': category})

    df = pd.DataFrame(records)
    print(f"[INFO] โหลดภาพจาก {input_dir} ได้ {len(df)} รูป, {df['category'].nunique()} class")
    return df


def suggest_target_size(df_stats: pd.DataFrame) -> tuple:
    median_w = int(df_stats['width'].median())
    median_h = int(df_stats['height'].median())

    median_w -= median_w % 2
    median_h -= median_h % 2

    print(f"[INFO] ขนาดภาพ median จาก dataset: ({median_w}, {median_h})")
    return (median_w, median_h)



#ใช้ LANCZOS resampling สำหรับ resize ภาพก็เพราะ เวลาย่อภาพลง LANCZOS จะให้ผลลัพธ์ที่คมชัดและลด aliasing
#ได้ดีกว่า resampling methods อื่น ๆ เช่น NEAREST หรือ BILINEAR โดยเฉพาะเมื่อย่อภาพลงมาก ๆ

#ละที่เป็นขนาดภาพ target_size = (224, 224)
#ก็เพราะว่าเป็นขนาดมาตรฐานที่ใช้ในหลาย ๆ โมเดล deep learning เช่น ResNet, VGG, MobileNet ซึ่งถูกฝึกมาให้รับภาพขนาดนี้

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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


#normalize_pixels ทำ min-max scaling ธรรมดา (หารด้วย 255) ให้ค่าพิกเซลอยู่ในช่วง [0, 1]
#เป็นขั้นตอนพื้นฐานที่เกือบทุก pipeline ต้องทำ เพราะช่วยให้ gradient ตอนเทรนโมเดลเสถียรขึ้น

# standardize_pixels ทำต่ออีกขั้น ปรับให้ mean=0, std=1 ด้วยสูตร (x - mean) / std
# ใช้ค่า mean/std มาตรฐานของ ImageNet เพราะถ้าจะทำ Transfer Learning กับ pretrained model
# (ResNet, VGG ฯลฯ) ต้องปรับ input distribution ให้ตรงกับที่โมเดลเคยเห็นตอนเทรนมา
def normalize_pixels(image: Image.Image) -> np.ndarray:
    arr = np.array(image, dtype=np.float32)
    return arr / 255.0


def standardize_pixels(image: Image.Image, method: str = "imagenet") -> np.ndarray:
    arr = normalize_pixels(image)

    if method == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif method == "dataset":
        raise ValueError(
            "method='dataset' ต้องส่ง mean/std ที่คำนวณเองเข้ามาโดยตรง "
            "ใช้ compute_dataset_mean_std() หาค่าก่อน แล้วคำนวณ (arr - mean) / std เอง"
        )
    else:
        raise ValueError(f"method ต้องเป็น 'imagenet' หรือ 'dataset' เท่านั้น (ได้รับ: {method})")

    return (arr - mean) / std


def compute_dataset_mean_std(df: pd.DataFrame, sample_size: int = 1000) -> tuple:
    sample_df = df.sample(min(sample_size, len(df)))

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for _, row in sample_df.iterrows():
        with Image.open(row['file_path']) as img:
            arr = np.array(img.convert('RGB'), dtype=np.float64) / 255.0
            pixel_sum += arr.sum(axis=(0, 1))
            pixel_sq_sum += (arr ** 2).sum(axis=(0, 1))
            pixel_count += arr.shape[0] * arr.shape[1]

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    print(f"[INFO] Dataset mean (RGB): {mean}")
    print(f"[INFO] Dataset std  (RGB): {std}")
    return mean.astype(np.float32), std.astype(np.float32)



#โชว์เป็น histogram แทน imshow ตรง ๆ เพราะหลัง standardize ค่าพิกเซลจะติดลบได้ (เช่น -2 ถึง 2.5)
#ถ้า imshow ตรง ๆ matplotlib จะ clip ค่าจนภาพเพี้ยนสี ดูไม่ออกว่าการกระจายข้อมูลเปลี่ยนไปยังไง

def show_normalize_before_after(df: pd.DataFrame, method: str = "imagenet",
                                 n_samples: int = 4,
                                 save_name: str = "normalize_before_after.png"):
    samples = df.sample(min(n_samples, len(df)))

    fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))
    if len(samples) == 1:
        axes = axes.reshape(2, 1)

    for i, (_, row) in enumerate(samples.iterrows()):
        with Image.open(row['file_path']) as img:
            img = img.convert('RGB')
            raw_arr = np.array(img)
            processed_arr = standardize_pixels(img, method=method)

            axes[0, i].hist(raw_arr.ravel(), bins=50, color='steelblue')
            axes[0, i].set_title(f"Before\nrange [{raw_arr.min()}, {raw_arr.max()}]")

            axes[1, i].hist(processed_arr.ravel(), bins=50, color='orange')
            axes[1, i].set_title(f"After\nrange [{processed_arr.min():.2f}, {processed_arr.max():.2f}]")

    plt.suptitle(f"Pixel Distribution: Before vs After ({method} standardize)")
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"[SUCCESS] เซฟภาพเปรียบเทียบ Before/After ที่: {save_path}")


#ใช้ Median Filter เป็นค่า default สำหรับ denoise เพราะ dataset ภาพทั่วไป (โดยเฉพาะภาพถ่ายที่โหลดมาจากหลายแหล่ง)
#มักเจอ salt-and-pepper noise (จุดขาว/ดำกระจายเป็นจุด ๆ) ซึ่ง Median Filter จัดการได้ดีกว่า Gaussian Blur มาก
#เพราะมันแทนที่แต่ละพิกเซลด้วยค่ามัธยฐานของเพื่อนบ้าน ทำให้ noise หลุดออกไปโดยที่ขอบภาพ (edge) ยังคมอยู่
#ต่างจาก Gaussian Blur ที่ลด noise แบบเฉลี่ยถ่วงน้ำหนัก เลยทำให้ขอบภาพเบลอไปด้วย

#kernel_size ต้องเป็นเลขคี่เท่านั้น (ข้อกำหนดของ PIL.ImageFilter.MedianFilter) และ default = 3
#เพราะเป็นค่าที่เบาที่สุดที่ยังลบ noise แบบจุด ๆ ได้ โดยไม่ทำให้รายละเอียดเล็ก ๆ ในภาพหายไปเยอะเกินไป
#ถ้า noise เยอะกว่านี้ค่อยขยับขึ้นเป็น 5 หรือ 7

def denoise_image(image: Image.Image, method: str = "median",
                   kernel_size: int = 3) -> Image.Image:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size ต้องเป็นเลขคี่เท่านั้น (ได้รับ: {kernel_size})")

    if method == "median":
        return image.filter(ImageFilter.MedianFilter(size=kernel_size))
    elif method == "gaussian":
        radius = max(1, kernel_size // 2)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    else:
        raise ValueError(f"method ต้องเป็น 'median' หรือ 'gaussian' เท่านั้น (ได้รับ: {method})")


def denoise_dataset(df: pd.DataFrame, output_dir: str,
                     method: str = "median",
                     kernel_size: int = 3) -> pd.DataFrame:
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
                denoised_img = denoise_image(img, method=method, kernel_size=kernel_size)
                denoised_img.save(dst_path)
        except Exception as e:
            print(f"[WARNING] Denoise ไม่สำเร็จ: {src_path} ({e})")
            continue

        records.append({
            'file_path': dst_path,
            'category': category,
        })

    result_df = pd.DataFrame(records)
    print(f"[SUCCESS] Denoise เสร็จสิ้น: {len(result_df)}/{len(df)} รูป -> {output_dir}")
    return result_df


def show_denoise_before_after(df: pd.DataFrame, method: str = "median",
                               kernel_size: int = 3,
                               n_samples: int = 4,
                               save_name: str = "denoise_before_after.png"):
    samples = df.sample(min(n_samples, len(df)))

    fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))
    if len(samples) == 1:
        axes = axes.reshape(2, 1)

    for i, (_, row) in enumerate(samples.iterrows()):
        with Image.open(row['file_path']) as img:
            img = img.convert('RGB')
            denoised = denoise_image(img, method=method, kernel_size=kernel_size)

            axes[0, i].imshow(img)
            axes[0, i].set_title("Before")
            axes[0, i].axis('off')

            axes[1, i].imshow(denoised)
            axes[1, i].set_title(f"After\n({method}, k={kernel_size})")
            axes[1, i].axis('off')

    plt.suptitle(f"Denoise: Before vs After ({method})")
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"[SUCCESS] เซฟภาพเปรียบเทียบ Before/After ที่: {save_path}")



#เลือกใช้ 4 เทคนิคนี้เพราะ dataset เป็นภาพ "cat-breeds" (จำแนกสายพันธุ์แมว) โดยแต่ละเทคนิคช่วยจำลอง
#ความแปรปรวนที่พบได้จริงตอนถ่ายภาพ โดยไม่ทำให้ label (สายพันธุ์) เปลี่ยนไป:

#1. horizontal_flip (พลิกซ้าย-ขวา) — แมวหันซ้ายหรือขวาก็ยังเป็นสายพันธุ์เดิม ไม่มีความหมายเชิงทิศทาง
#   จึงเป็นเทคนิคที่ "ปลอดภัย" ที่สุดและได้ผลดีมากสำหรับงาน image classification ทั่วไป
#  *ไม่ใช้ vertical_flip* เพราะแมวกลับหัวไม่ใช่สิ่งที่เจอในข้อมูลจริง จะทำให้โมเดลเรียนรู้ pattern ที่ผิดธรรมชาติ

#2. rotation (หมุนมุมเล็กน้อย ±ไม่เกิน 20 องศา) — จำลองมุมกล้องที่เอียงตอนถ่ายจริง (มือสั่น/ถ่ายไม่ตรง)
#   จำกัดองศาไม่ให้เยอะเกินไป เพราะถ้าหมุนมาก ๆ ภาพจะเสียสัดส่วนและมี padding ดำเข้ามารบกวน

#3. brightness/contrast adjustment — จำลองสภาพแสงที่ต่างกัน (ถ่ายในบ้าน/กลางแจ้ง/มีแฟลช)
#   เพราะแสงเป็นตัวแปรที่เปลี่ยนบ่อยที่สุดใน dataset ที่รวบรวมจากหลายแหล่ง ช่วยให้โมเดลไม่ overfit กับความสว่างเฉพาะจุด

#4. random_crop_zoom (crop แล้วขยายกลับ) — จำลองระยะห่างจากกล้อง/การจัดองค์ประกอบภาพที่ต่างกัน
#   บังคับให้โมเดลโฟกัสที่ลักษณะเด่นของแมว (ลาย, รูปหน้า, หู) แทนที่จะจำ background หรือตำแหน่งของวัตถุในเฟรม

#ทุกเทคนิครวมกันในฟังก์ชันเดียว (augment_image) แบบสุ่มว่าจะใช้ตัวไหนบ้าง เพื่อให้แต่ละภาพที่ออกมามีความหลากหลาย
#ไม่ใช่ apply ทุกเทคนิคพร้อมกันเสมอ ซึ่งจะทำให้ภาพเพี้ยนเกินจริงจนไม่เหมือนข้อมูลจริง

def augment_image(image: Image.Image,
                   techniques: tuple = ("flip", "rotate", "brightness", "crop_zoom"),
                   rotate_range: float = 20.0,
                   brightness_range: tuple = (0.7, 1.3),
                   contrast_range: tuple = (0.7, 1.3),
                   crop_scale_range: tuple = (0.8, 1.0),
                   rng: np.random.Generator = None) -> Image.Image:
    if rng is None:
        rng = np.random.default_rng()

    img = image

    if "flip" in techniques and rng.random() < 0.5:
        img = ImageOps.mirror(img)

    if "rotate" in techniques:
        angle = rng.uniform(-rotate_range, rotate_range)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR,
                          fillcolor=(0, 0, 0))

    if "brightness" in techniques:
        brightness_factor = rng.uniform(*brightness_range)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        contrast_factor = rng.uniform(*contrast_range)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    if "crop_zoom" in techniques:
        orig_w, orig_h = img.size
        scale = rng.uniform(*crop_scale_range)
        crop_w, crop_h = int(orig_w * scale), int(orig_h * scale)

        left = rng.integers(0, max(1, orig_w - crop_w + 1))
        top = rng.integers(0, max(1, orig_h - crop_h + 1))
        img = img.crop((left, top, left + crop_w, top + crop_h))
        img = img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    return img


def augment_dataset(df: pd.DataFrame, output_dir: str,
                     n_augments_per_image: int = 3,
                     techniques: tuple = ("flip", "rotate", "brightness", "crop_zoom"),
                     seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    # เก็บภาพต้นฉบับไว้ในผลลัพธ์ด้วย ไม่ใช่แค่ภาพที่ augment แล้ว
    # เพราะการ train ควรเห็นทั้งข้อมูลจริงและข้อมูลที่ขยายเพิ่ม ไม่ใช่แทนที่กันไปเลย
    for _, row in df.iterrows():
        src_path = row['file_path']
        category = row['category']
        dst_folder = os.path.join(output_dir, category)
        os.makedirs(dst_folder, exist_ok=True)

        base_name, ext = os.path.splitext(os.path.basename(src_path))
        orig_dst = os.path.join(dst_folder, f"{base_name}{ext}")

        try:
            with Image.open(src_path) as img:
                img = img.convert('RGB')
                img.save(orig_dst)
                records.append({'file_path': orig_dst, 'category': category})

                for aug_idx in range(n_augments_per_image):
                    augmented = augment_image(img, techniques=techniques, rng=rng)
                    aug_dst = os.path.join(dst_folder, f"{base_name}_aug{aug_idx}{ext}")
                    augmented.save(aug_dst)
                    records.append({'file_path': aug_dst, 'category': category})
        except Exception as e:
            print(f"[WARNING] Augment ไม่สำเร็จ: {src_path} ({e})")
            continue

    result_df = pd.DataFrame(records)
    print(f"[SUCCESS] Augment เสร็จสิ้น: {len(df)} รูปต้นฉบับ -> {len(result_df)} รูปรวม -> {output_dir}")
    return result_df


def show_augment_before_after(df: pd.DataFrame,
                               techniques: tuple = ("flip", "rotate", "brightness", "crop_zoom"),
                               n_samples: int = 4,
                               n_variants: int = 3,
                               save_name: str = "augment_before_after.png"):
    samples = df.sample(min(n_samples, len(df)))
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(n_samples, n_variants + 1, figsize=(4 * (n_variants + 1), 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, (_, row) in enumerate(samples.iterrows()):
        with Image.open(row['file_path']) as img:
            img = img.convert('RGB')

            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')

            for v in range(n_variants):
                augmented = augment_image(img, techniques=techniques, rng=rng)
                axes[i, v + 1].imshow(augmented)
                axes[i, v + 1].set_title(f"Augmented #{v + 1}")
                axes[i, v + 1].axis('off')

    plt.suptitle("Data Augmentation: Original vs Augmented Variants")
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"[SUCCESS] เซฟภาพเปรียบเทียบ Original/Augmented ที่: {save_path}")



#เทียบจำนวนข้อมูล "ก่อน-หลัง Clean" แยกตาม category (ไม่ใช่แค่ตัวเลขรวม) เพราะ:

#1. ถ้าดูแค่ยอดรวม อาจพลาดปัญหา class imbalance ที่เกิดขึ้นจากการ clean เช่น
#   บาง breed อาจถูกกรองออกไปเยอะผิดปกติ (ภาพเบลอ/ไฟล์เสียเยอะ) จนเหลือข้อมูลน้อยกว่า class อื่นมาก
#   ซึ่งจะกระทบตอนเทรนโมเดลโดยตรง (โมเดลจะ bias ไปทาง class ที่มีข้อมูลเยอะกว่า)

#2. removed_pct (เปอร์เซ็นต์ที่หายไป) สำคัญกว่า removed_count เฉย ๆ เพราะ breed ที่มีข้อมูลน้อยอยู่แล้ว
#   ถ้าโดน clean ออกไป 20 รูปจาก 30 รูป (66%) จะกระทบหนักกว่า breed ที่มี 500 รูปแล้วโดนออก 20 รูป (4%)
#   แม้ removed_count จะเท่ากันก็ตาม

#3. เก็บแถว TOTAL ไว้ด้วยเพื่อดูภาพรวมทั้ง dataset ควบคู่กับรายละเอียดแต่ละ class ในตารางเดียว

def summarize_clean_counts(raw_dir: str, cleaned_dir: str) -> pd.DataFrame:
    raw_df = load_images_from_folder(raw_dir)
    cleaned_df = load_images_from_folder(cleaned_dir)

    raw_counts = raw_df['category'].value_counts().rename('raw_count')
    cleaned_counts = cleaned_df['category'].value_counts().rename('cleaned_count')

    summary = pd.concat([raw_counts, cleaned_counts], axis=1).fillna(0).astype(int)
    summary['removed_count'] = summary['raw_count'] - summary['cleaned_count']
    summary['removed_pct'] = (
        (summary['removed_count'] / summary['raw_count'].replace(0, np.nan)) * 100
    ).round(2).fillna(0.0)
    summary = summary.sort_values('raw_count', ascending=False)

    total_row = pd.DataFrame({
        'raw_count': [summary['raw_count'].sum()],
        'cleaned_count': [summary['cleaned_count'].sum()],
        'removed_count': [summary['removed_count'].sum()],
        'removed_pct': [round(summary['removed_count'].sum() / summary['raw_count'].sum() * 100, 2)],
    }, index=['TOTAL'])
    summary = pd.concat([summary, total_row])

    print("[INFO] สรุปจำนวนข้อมูล ก่อน-หลัง Clean (แยกตาม category):")
    print(summary.to_string())

    return summary


def show_clean_summary(summary_df: pd.DataFrame,
                        save_name: str = "clean_summary.png"):
    plot_df = summary_df.drop(index='TOTAL', errors='ignore')

    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.2), 6))
    # หมายเหตุ: ใช้ text ภาษาอังกฤษล้วนบนกราฟ (title/label/legend) เพราะ matplotlib
    # ไม่มีฟอนต์ไทย default ติดมาให้ ถ้าใส่ข้อความไทยตรงนี้จะขึ้นเป็นกล่องสี่เหลี่ยม (tofu box)
    # แทนตัวอักษร -- ต่างจาก print() ที่ terminal render ภาษาไทยได้ปกติอยู่แล้ว
    ax.bar(x - width / 2, plot_df['raw_count'], width, label='Raw (before clean)', color='steelblue')
    ax.bar(x + width / 2, plot_df['cleaned_count'], width, label='Cleaned (after clean)', color='orange')

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=45, ha='right')
    ax.set_ylabel("Image count")
    ax.set_title("Image Count per Category: Before vs After Clean")
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"[SUCCESS] เซฟกราฟสรุป Clean ที่: {save_path}")


if __name__ == "__main__":
    import sys

    # หา RAW_DIR อัตโนมัติจาก Kaggle cache (ใช้ฟังก์ชันเดียวกับ preprocessing.py)
    # kagglehub จะคืน path เดิมจาก cache ทันทีถ้าเคยโหลดแล้ว ไม่โหลดซ้ำ
    # ถ้า import ไม่ได้ (ไม่มีไฟล์ preprocessing.py อยู่ข้าง ๆ หรือไม่ได้ลง kagglehub)
    # ให้ fallback เป็น RAW_DIR=None แล้วข้ามขั้นตอนสรุป ก่อน-หลัง Clean ไปเฉย ๆ
    RAW_DIR = None
    try:
        from preprocessing import find_kaggle_cache_path
        _kaggle_root = find_kaggle_cache_path("nikolasgegenava/cat-breeds")
        RAW_DIR = os.path.join(_kaggle_root, "cat-breeds")
    except Exception as e:
        print(f"[INFO] หา RAW_DIR จาก Kaggle cache ไม่ได้ ({e}) จะข้ามขั้นตอนสรุปก่อน-หลัง Clean")

    # หมายเหตุ path: preprocessing.py ใช้ copy_valid_images(TEST_DIR, OUTPUT_DIR) โดย
    # TEST_DIR = <kaggle_root>/cat-breeds และ OUTPUT_DIR = "data/cleaned"
    # เพราะใช้ os.path.relpath(src_path, TEST_DIR) ไฟล์เลยไปอยู่ที่ data/cleaned/<breed>/...
    # ไม่ใช่ data/cleaned/cat-breeds/<breed>/... จึงต้องแก้ INPUT_DIR ตรงนี้ให้ตรงกับของจริง
    INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/cleaned"
    RESIZED_DIR = "data/resized"
    TARGET_SIZE = (224, 224)

    if RAW_DIR and os.path.isdir(RAW_DIR):
        clean_summary_df = summarize_clean_counts(RAW_DIR, INPUT_DIR)
        show_clean_summary(clean_summary_df)
    else:
        print(f"[INFO] ไม่พบ RAW_DIR จะข้ามขั้นตอนสรุปก่อน-หลัง Clean")

    df = load_images_from_folder(INPUT_DIR)

    if len(df) == 0:
        print(f"[ERROR] ไม่พบไฟล์ภาพใน {INPUT_DIR}")
        sys.exit(1)

    resized_df = resize_dataset(df, RESIZED_DIR, target_size=TARGET_SIZE)
    show_resize_before_after(df, target_size=TARGET_SIZE)

    DENOISED_DIR = "data/denoised"
    denoised_df = denoise_dataset(resized_df, DENOISED_DIR, method="median", kernel_size=3)
    show_denoise_before_after(resized_df, method="median", kernel_size=3)

    mean, std = compute_dataset_mean_std(denoised_df)
    show_normalize_before_after(denoised_df, method="imagenet")

    AUGMENTED_DIR = "data/augmented"
    augmented_df = augment_dataset(denoised_df, AUGMENTED_DIR, n_augments_per_image=3)
    show_augment_before_after(denoised_df)