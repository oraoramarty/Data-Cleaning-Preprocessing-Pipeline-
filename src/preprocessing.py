import kagglehub
import os
import shutil
import random
from PIL import Image
import imagehash

VALID_EXT = (".jpg", ".jpeg", ".png")


# =========================================================================
# ฟีเจอร์ 0: หา path ของ dataset ใน Kaggle cache อัตโนมัติ
# =========================================================================
def find_kaggle_cache_path(dataset_handle="nikolasgegenava/cat-breeds"):
    #เรียก kagglehub เพื่อหา path ของ dataset ใน cache
    #ถ้าเคยโหลดแล้ว kagglehub จะคืน path เดิมจาก cache ทันที (ไม่โหลดซ้ำ)
    #ถ้ายังไม่เคยโหลด จะดาวน์โหลดให้อัตโนมัติ

    #คืนค่า: path (str) ของโฟลเดอร์ dataset ใน cache
    
    #import kagglehub  # import ในฟังก์ชันเพื่อไม่บังคับให้ต้องลงไลบรารีนี้ถ้าไม่ได้ใช้ฟังก์ชันนี้

    print(f"[Kaggle Cache] กำลังเช็ค/ดึง dataset: {dataset_handle}")
    dataset_path = kagglehub.dataset_download(dataset_handle)
    print(f"[Kaggle Cache] พบ path: {dataset_path}")
    return dataset_path


# =========================================================================
# ฟีเจอร์ 3: ดึงไฟล์ภาพทั้งหมดออกมาวางในโฟลเดอร์ทำงานใหม่
# =========================================================================
# หมายเหตุ: ย้ายฟีเจอร์นี้มาไว้เป็นขั้นตอนแรกสุด (ก่อนฟีเจอร์ 1/2)
# เพื่อไม่ให้ remove_corrupted_images / remove_duplicate_images ไปลบไฟล์
# ใน raw dataset (Kaggle cache) โดยตรง -- ต้อง copy มาไว้ที่ output_dir
# ก่อนเสมอ แล้วค่อยลบไฟล์เสีย/ซ้ำ "บนไฟล์ copy" เท่านั้น
# เหตุผล: raw data ต้องเก็บไว้ครบสำหรับให้ EDA (คนที่ 2) เทียบ
# before/after ได้ และไม่ต้องดาวน์โหลด dataset ใหม่ทุกครั้งที่ต้องการรีเซ็ต
def copy_valid_images(input_dir, output_dir):

    #Copy ไฟล์ภาพทั้งหมดจาก input_dir (raw data) ไปยัง output_dir
    #โดยรักษาโครงสร้าง subfolder เดิมไว้ (เช่น แยกตาม class)

    #เรียกฟังก์ชันนี้เป็นขั้นตอนแรกสุด ก่อน remove_corrupted_images และ
    #remove_duplicate_images เสมอ เพื่อให้ทุกการลบไฟล์เกิดขึ้นบน output_dir
    #(โฟลเดอร์ทำงาน) เท่านั้น ไม่ใช่บน raw dataset ต้นฉบับ

    #คืนค่า: จำนวนไฟล์ที่ copy สำเร็จ
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    copied_count = 0
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(VALID_EXT):
                continue

            src_path = os.path.join(root, fname)

            # รักษาโครงสร้างโฟลเดอร์ย่อย (เช่น class ของแมว) เหมือนต้นฉบับ
            rel_path = os.path.relpath(src_path, input_dir)
            dst_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            shutil.copy2(src_path, dst_path)  # copy2 = คัดลอกพร้อม metadata
            copied_count += 1

    print(f"[Copy Valid Images] คัดลอกไฟล์ทั้งหมด {copied_count} ไฟล์ -> {output_dir}")
    return copied_count


# =========================================================================
# ฟีเจอร์ 1: ลบไฟล์ที่เสียหายหรือใช้งานไม่ได้ (Corrupted Images)
# =========================================================================
def is_corrupted(filepath):
    
    #เช็คว่าไฟล์ภาพเปิดได้จริงหรือไม่
    #ใช้ img.verify() เพื่อตรวจสอบโครงสร้างไฟล์ (ไม่ได้โหลดข้อมูลภาพทั้งหมดเข้า memory)
    #คืนค่า True ถ้าไฟล์เสีย, False ถ้าไฟล์ปกติ
    
    try:
        with Image.open(filepath) as img:
            img.verify()
        return False
    except Exception:
        return True


def remove_corrupted_images(input_dir, dry_run=False):
    #สแกนทุกไฟล์ภาพใน input_dir (รวม subfolder) หาไฟล์ที่เสีย

    #ต้องเรียกด้วย OUTPUT_DIR (โฟลเดอร์ copy) เท่านั้น ห้ามเรียกด้วย
    #TEST_DIR (raw cache) ตรง ๆ เพราะฟังก์ชันนี้ลบไฟล์จริงถ้า dry_run=False

    #dry_run=True  -> แค่รายงานว่าจะลบอะไรบ้าง ไม่ลบจริง (ใช้เช็คก่อนได้)
    #dry_run=False -> ลบไฟล์เสียจริง

    #คืนค่า list ของ path ไฟล์ที่เสีย (หรือถูกลบไปแล้ว)
    
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {input_dir}")

    corrupted_files = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(VALID_EXT):
                path = os.path.join(root, fname)
                if is_corrupted(path):
                    corrupted_files.append(path)
                    if not dry_run:
                        os.remove(path)

    action = "พบ (dry run ไม่ได้ลบจริง)" if dry_run else "ลบไปแล้ว"
    print(f"[Corrupted Images] {action} ทั้งหมด {len(corrupted_files)} ไฟล์")
    for f in corrupted_files:
        print(f"   - {f}")

    return corrupted_files


# =========================================================================
# ฟีเจอร์ 2: ตรวจจับและจัดการรูปภาพซ้ำ (Duplicate Detection)
# =========================================================================
def find_duplicate_images(input_dir, hash_size=8):
    
    #ใช้ Perceptual Hash (phash) หารูปที่ "หน้าตาเหมือนกัน" แม้ชื่อไฟล์ต่างกัน
    #หรือถูก resize/บีบอัดมาต่างกันเล็กน้อย (ต่างจากการเทียบไฟล์แบบ byte-to-byte)

    #!! ต้องเรียกด้วย OUTPUT_DIR (โฟลเดอร์ copy) เท่านั้น เช่นเดียวกับ
    #remove_corrupted_images

    #คืนค่า dict {hash: [path1, path2, ...]} เฉพาะกลุ่มที่มีมากกว่า 1 ไฟล์ (ซ้ำกัน)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {input_dir}")

    hashes = {}
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(VALID_EXT):
                path = os.path.join(root, fname)
                try:
                    with Image.open(path) as img:
                        h = str(imagehash.phash(img, hash_size=hash_size))
                    hashes.setdefault(h, []).append(path)
                except Exception:
                    continue  # ไฟล์เสียถูกจัดการไปแล้วในขั้นตอนก่อนหน้า (remove_corrupted_images)

    duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    total_dup_files = sum(len(v) - 1 for v in duplicates.values())  # -1 เพราะเก็บตัวแทนไว้ 1 ไฟล์ต่อกลุ่ม

    print(f"[Duplicate Images] พบภาพซ้ำทั้งหมด {total_dup_files} ไฟล์ ({len(duplicates)} กลุ่ม)")
    for h, paths in duplicates.items():
        print(f"   กลุ่ม {h[:8]}...: {paths}")

    return duplicates


def remove_duplicate_images(duplicates, dry_run=False):
    
    #รับผลลัพธ์จาก find_duplicate_images() มาลบไฟล์ซ้ำออก
    #เก็บไฟล์แรก [0] ของแต่ละกลุ่มไว้เป็นตัวแทน ลบไฟล์ที่เหลือ

    #dry_run=True  -> รายงานว่าจะลบอะไรบ้าง ไม่ลบจริง
    #dry_run=False -> ลบไฟล์ซ้ำจริง
    removed = []
    for _, paths in duplicates.items():
        for p in paths[1:]:  # เก็บไฟล์แรกไว้ ลบที่เหลือ
            removed.append(p)
            if not dry_run:
                os.remove(p)

    action = "พบ (dry run ไม่ได้ลบจริง)" if dry_run else "ลบไปแล้ว"
    print(f"[Duplicate Images] {action} ทั้งหมด {len(removed)} ไฟล์")

    return removed


# =========================================================================
# ฟีเจอร์ 4: จัดการ Class Imbalance
# =========================================================================
def find_class_folders(input_dir):
    
    #หาโฟลเดอร์ที่เป็น "class จริง" แบบ recursive (เหมือนวิธีที่ eda.py ใช้)

    #นิยาม class = โฟลเดอร์ leaf ที่มีไฟล์ภาพอยู่ข้างในโดยตรง
    #(ไม่ใช่แค่ subfolder ชั้นแรกของ input_dir อย่างเดียว) เพราะ dataset นี้
    #มีโฟลเดอร์ wrapper ซ้อนอยู่อีกชั้นก่อนถึงโฟลเดอร์ breed จริง
    #เช่น data/cleaned/cat-breeds/persian/*.jpg
    #                  ^^^^^^^^^^ wrapper   ^^^^^^^ class จริง

    #คืนค่า dict {class_name: folder_path}
    
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {input_dir}")

    class_dirs = {}
    for root, _, files in os.walk(input_dir):
        image_files = [f for f in files if f.lower().endswith(VALID_EXT)]
        if image_files:
            class_name = os.path.basename(root)
            if class_name in class_dirs and class_dirs[class_name] != root:
                print(f"[WARNING] พบชื่อ class '{class_name}' ซ้ำกันคนละ path: "
                      f"{class_dirs[class_name]} และ {root}")
            class_dirs[class_name] = root

    return class_dirs


def check_class_distribution(input_dir):
    
    #นับจำนวนรูปในแต่ละ class (หา class folder แบบ recursive ด้วย find_class_folders)

    class_dirs = find_class_folders(input_dir)

    distribution = {}
    for class_name, class_path in class_dirs.items():
        count = len([f for f in os.listdir(class_path) if f.lower().endswith(VALID_EXT)])
        distribution[class_name] = count

    print("[Class Distribution]")
    for class_name, count in sorted(distribution.items(), key=lambda x: -x[1]):
        print(f"   {class_name}: {count} ไฟล์")

    return distribution


def oversample_class(class_dir, target_count, dry_run=False):

    #เพิ่มไฟล์ใน class_dir ให้ครบ target_count โดย copy ไฟล์เดิมซ้ำแบบสุ่ม
    #(ไม่ได้สร้างภาพใหม่ แค่ duplicate ไฟล์ที่มีอยู่ เพื่อให้จำนวนเท่ากันทุก class)

    #วิธีอื่นที่เลือกใช้แทนได้ (แล้วแต่ความเหมาะสมของ dataset):
    # - Undersampling: ตัดไฟล์ class ที่เยอะเกินให้เหลือเท่า class ที่น้อยที่สุด
    # - Weighted Sampling: ไม่ต้องเพิ่ม/ลดไฟล์จริง แต่ปรับ weight ตอน train แทน

    #dry_run=True  -> รายงานว่าจะเพิ่มกี่ไฟล์ ไม่ copy จริง
    #dry_run=False -> copy ไฟล์ซ้ำจริง

    files = [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXT)]
    current_count = len(files)

    if current_count == 0:
        print(f"[Oversample] {class_dir} ไม่มีไฟล์เลย ข้าม")
        return 0

    need = target_count - current_count
    if need <= 0:
        print(f"[Oversample] {class_dir} มี {current_count} ไฟล์ครบแล้ว (target={target_count}) ไม่ต้องเพิ่ม")
        return 0

    action = "จะเพิ่ม (dry run ไม่ได้ copy จริง)" if dry_run else "เพิ่มแล้ว"
    for i in range(need):
        src_name = random.choice(files)
        src_path = os.path.join(class_dir, src_name)
        dst_path = os.path.join(class_dir, f"aug_{i}_{src_name}")
        if not dry_run:
            shutil.copy2(src_path, dst_path)

    print(f"[Oversample] {class_dir}: {action} {need} ไฟล์ ({current_count} -> {target_count})")
    return need


def balance_classes(input_dir, dry_run=False):
    #เช็ค class distribution แล้ว oversample ทุก class ที่มีไฟล์น้อยกว่า
    #ให้เท่ากับ class ที่มีไฟล์เยอะที่สุด

    class_dirs = find_class_folders(input_dir)
    if not class_dirs:
        print("[Balance Classes] ไม่พบ class ใดเลยใน", input_dir)
        return

    distribution = {name: len([f for f in os.listdir(path) if f.lower().endswith(VALID_EXT)])
                     for name, path in class_dirs.items()}

    max_count = max(distribution.values())
    print(f"[Balance Classes] จะปรับทุก class ให้มีไฟล์เท่ากับ class ที่เยอะที่สุด ({max_count} ไฟล์)")

    for class_name, class_path in class_dirs.items():
        oversample_class(class_path, max_count, dry_run=dry_run)


# =========================================================================
# ฟีเจอร์ 5: แปลง Format และ Color Space ให้เป็นมาตรฐานเดียวกัน
# =========================================================================
def standardize_image(
    filepath,
    output_path=None,
    target_format="JPEG",
    target_mode="RGB",
    quality=95,
):
    
    #เปิดไฟล์ภาพ 1 ไฟล์ แล้วแปลงให้เป็นมาตรฐานเดียวกัน:
      #1. Color Space -> target_mode (ค่าเริ่มต้น "RGB")
      #รองรับกรณีภาพเป็น Grayscale (L), มี Alpha channel (RGBA/LA/P), หรือ CMYK
      #2. Format -> target_format (ค่าเริ่มต้น "JPEG")
      #เช่นถ้า dataset ปนกันทั้ง .jpg/.jpeg/.png ให้แปลงเป็น .jpg ทั้งหมด

    #filepath     : path ไฟล์ภาพต้นฉบับ
    #output_path  : path ปลายทาง ถ้าไม่ระบุ (None) จะ overwrite ไฟล์เดิม
    #               (เปลี่ยนนามสกุลไฟล์ให้ตรงกับ target_format อัตโนมัติ)
    #target_format: "JPEG" หรือ "PNG" (ตาม Pillow format name)
    #target_mode  : "RGB" (ค่ามาตรฐานสำหรับ dataset ทั่วไป) หรือ mode อื่นของ Pillow
    #quality      : คุณภาพตอน save (ใช้เฉพาะกรณี JPEG, ช่วง 1-95)

    #คืนค่า: path ของไฟล์ผลลัพธ์ที่ save ไปแล้ว หรือ None ถ้าแปลงไม่สำเร็จ

    ext_map = {"JPEG": ".jpg", "PNG": ".png"}
    target_ext = ext_map.get(target_format.upper(), ".jpg")

    if output_path is None:
        base, _ = os.path.splitext(filepath)
        output_path = base + target_ext
    else:
        base, _ = os.path.splitext(output_path)
        output_path = base + target_ext

    try:
        with Image.open(filepath) as img:
            # กรณีภาพมี Alpha channel (RGBA, LA, P ที่มี transparency)
            # ต้อง flatten ด้วยพื้นหลังสีขาวก่อน ไม่งั้นแปลงเป็น RGB ตรง ๆ
            # จะทำให้บริเวณโปร่งใสกลายเป็นสีดำ/สีเพี้ยน
            if img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            ):
                img = img.convert("RGBA")
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # ใช้ alpha channel เป็น mask
                img = background
            elif img.mode != target_mode:
                # กรณีอื่น เช่น Grayscale (L), CMYK, P (ไม่มี transparency)
                img = img.convert(target_mode)

            save_kwargs = {}
            if target_format.upper() == "JPEG":
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = True

            img.save(output_path, format=target_format.upper(), **save_kwargs)

        return output_path

    except Exception as e:
        print(f"   [ERROR] แปลงไฟล์ไม่สำเร็จ: {filepath} -> {e}")
        return None


def standardize_dataset(
    input_dir,
    target_format="JPEG",
    target_mode="RGB",
    quality=95,
    dry_run=False,
):
    #สแกนทุกไฟล์ภาพใน input_dir (รวม subfolder) แล้วแปลง format + color space
    #ให้เป็นมาตรฐานเดียวกันทั้ง dataset (เรียกใช้ standardize_image ทีละไฟล์)

    #ถ้าไฟล์ต้นฉบับมีนามสกุลไม่ตรงกับ target_format (เช่นเดิมเป็น .png แต่แปลงเป็น .jpg)
    #จะ save ไฟล์ใหม่แล้วลบไฟล์เดิมทิ้ง เพื่อไม่ให้เหลือไฟล์ซ้ำซ้อนสองสกุล

    #dry_run=True  -> แค่รายงานว่าจะแปลงไฟล์ไหนบ้าง (บอก mode เดิม) ไม่แปลงจริง
    #dry_run=False -> แปลงไฟล์จริงทั้งหมด

    #คืนค่า: dict สรุปผล {"converted": [...], "failed": [...], "skipped": [...]}
    
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {input_dir}")

    target_ext = {"JPEG": ".jpg", "PNG": ".png"}.get(target_format.upper(), ".jpg")

    result = {"converted": [], "failed": [], "skipped": []}

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(VALID_EXT):
                continue

            src_path = os.path.join(root, fname)

            try:
                with Image.open(src_path) as img:
                    current_mode = img.mode
                    current_ext = os.path.splitext(fname)[1].lower()
            except Exception as e:
                result["failed"].append(src_path)
                print(f"   [ERROR] เปิดไฟล์ไม่ได้ (ข้าม): {src_path} -> {e}")
                continue

            # ถ้าทั้ง mode และ นามสกุล ตรงมาตรฐานอยู่แล้ว ไม่ต้องแปลง
            already_standard = (
                current_mode == target_mode and current_ext == target_ext
            )
            if already_standard:
                result["skipped"].append(src_path)
                continue

            if dry_run:
                print(
                    f"   [DRY RUN] จะแปลง: {src_path} "
                    f"(mode เดิม={current_mode}, นามสกุลเดิม={current_ext} "
                    f"-> mode={target_mode}, นามสกุล={target_ext})"
                )
                result["converted"].append(src_path)
                continue

            new_path = standardize_image(
                src_path,
                target_format=target_format,
                target_mode=target_mode,
                quality=quality,
            )

            if new_path is None:
                result["failed"].append(src_path)
                continue

            # ถ้านามสกุลเปลี่ยน (เช่น .png -> .jpg) ให้ลบไฟล์เดิมทิ้ง
            # กันไม่ให้เหลือทั้งสองไฟล์ค้างอยู่
            if os.path.abspath(new_path) != os.path.abspath(src_path):
                os.remove(src_path)

            result["converted"].append(new_path)

    action = "จะแปลง (dry run ไม่ได้แปลงจริง)" if dry_run else "แปลงแล้ว"
    print(
        f"[Standardize Dataset] {action} {len(result['converted'])} ไฟล์, "
        f"ข้าม (มาตรฐานอยู่แล้ว) {len(result['skipped'])} ไฟล์, "
        f"ล้มเหลว {len(result['failed'])} ไฟล์"
    )

    return result


if __name__ == "__main__":
    # 0. หา path ของ dataset ใน Kaggle cache อัตโนมัติ (ไม่ต้องพิมพ์ path เอง)
    raw_dir = find_kaggle_cache_path("nikolasgegenava/cat-breeds")

    # dataset นี้มีรูปอยู่ใน subfolder ชื่อ "cat-breeds" ตามที่ data_collection.py ของทีมระบุไว้
    TEST_DIR = os.path.join(raw_dir, "cat-breeds")  # raw data (ห้ามแก้ไข/ลบไฟล์ในนี้)
    OUTPUT_DIR = "data/cleaned"                     # โฟลเดอร์ทำงาน (ลบ/แก้ไขได้อิสระ)

    # ---- ฟีเจอร์ 3: copy raw data ทั้งหมดไป OUTPUT_DIR ก่อนเป็นอันดับแรก ----
    # ทำก่อนเสมอ เพื่อไม่ให้ remove_corrupted_images/remove_duplicate_images
    # ไปแก้ไข raw dataset ใน Kaggle cache (TEST_DIR) โดยตรง
    copy_valid_images(TEST_DIR, OUTPUT_DIR)

    # ---- ฟีเจอร์ 1: ไฟล์เสีย (ทำงานบน OUTPUT_DIR เท่านั้น) ----
    remove_corrupted_images(OUTPUT_DIR, dry_run=False)

    # ---- ฟีเจอร์ 2: รูปซ้ำ (ทำงานบน OUTPUT_DIR เท่านั้น) ----
    duplicates = find_duplicate_images(OUTPUT_DIR)
    remove_duplicate_images(duplicates, dry_run=False)

    # ---- ฟีเจอร์ 4: จัดการ Class Imbalance ----
    check_class_distribution(OUTPUT_DIR)
    balance_classes(OUTPUT_DIR, dry_run=False)

    # ---- ฟีเจอร์ 5: แปลง Format + Color Space ให้เป็นมาตรฐานเดียวกัน ----
    standardize_dataset(OUTPUT_DIR, target_format="JPEG", target_mode="RGB", dry_run=False)