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
    """
    เรียก kagglehub เพื่อหา path ของ dataset ใน cache
    ถ้าเคยโหลดแล้ว kagglehub จะคืน path เดิมจาก cache ทันที (ไม่โหลดซ้ำ)
    ถ้ายังไม่เคยโหลด จะดาวน์โหลดให้อัตโนมัติ

    คืนค่า: path (str) ของโฟลเดอร์ dataset ใน cache
    """
    import kagglehub  # import ในฟังก์ชันเพื่อไม่บังคับให้ต้องลงไลบรารีนี้ถ้าไม่ได้ใช้ฟังก์ชันนี้

    print(f"[Kaggle Cache] กำลังเช็ค/ดึง dataset: {dataset_handle}")
    dataset_path = kagglehub.dataset_download(dataset_handle)
    print(f"[Kaggle Cache] พบ path: {dataset_path}")
    return dataset_path


# =========================================================================
# ฟีเจอร์ 1: ลบไฟล์ที่เสียหายหรือใช้งานไม่ได้ (Corrupted Images)
# =========================================================================
def is_corrupted(filepath):
    """
    เช็คว่าไฟล์ภาพเปิดได้จริงหรือไม่
    ใช้ img.verify() เพื่อตรวจสอบโครงสร้างไฟล์ (ไม่ได้โหลดข้อมูลภาพทั้งหมดเข้า memory)
    คืนค่า True ถ้าไฟล์เสีย, False ถ้าไฟล์ปกติ
    """
    try:
        with Image.open(filepath) as img:
            img.verify()
        return False
    except Exception:
        return True


def remove_corrupted_images(input_dir, dry_run=False):
    """
    สแกนทุกไฟล์ภาพใน input_dir (รวม subfolder) หาไฟล์ที่เสีย
    dry_run=True  -> แค่รายงานว่าจะลบอะไรบ้าง ไม่ลบจริง (ใช้เช็คก่อนได้)
    dry_run=False -> ลบไฟล์เสียจริง

    คืนค่า list ของ path ไฟล์ที่เสีย (หรือถูกลบไปแล้ว)
    """
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
    """
    ใช้ Perceptual Hash (phash) หารูปที่ "หน้าตาเหมือนกัน" แม้ชื่อไฟล์ต่างกัน
    หรือถูก resize/บีบอัดมาต่างกันเล็กน้อย (ต่างจากการเทียบไฟล์แบบ byte-to-byte)

    คืนค่า dict {hash: [path1, path2, ...]} เฉพาะกลุ่มที่มีมากกว่า 1 ไฟล์ (ซ้ำกัน)
    """
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
    """
    รับผลลัพธ์จาก find_duplicate_images() มาลบไฟล์ซ้ำออก
    เก็บไฟล์แรก [0] ของแต่ละกลุ่มไว้เป็นตัวแทน ลบไฟล์ที่เหลือ

    dry_run=True  -> รายงานว่าจะลบอะไรบ้าง ไม่ลบจริง
    dry_run=False -> ลบไฟล์ซ้ำจริง
    """
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
# ฟีเจอร์ 3: ดึงไฟล์ภาพที่ถูกต้อง (ผ่านการคัดกรองแล้ว) ออกมาวางในโฟลเดอร์ใหม่
# =========================================================================
def copy_valid_images(input_dir, output_dir):
    """
    เรียกหลังจากลบไฟล์เสีย/ซ้ำออกจาก input_dir แล้ว (ด้วย remove_corrupted_images
    และ remove_duplicate_images ที่ dry_run=False)

    Copy ไฟล์ภาพที่เหลือทั้งหมดใน input_dir ไปยัง output_dir
    โดยรักษาโครงสร้าง subfolder เดิมไว้ (เช่น แยกตาม class)

    คืนค่า: จำนวนไฟล์ที่ copy สำเร็จ
    """
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

    print(f"[Copy Valid Images] คัดลอกไฟล์ที่ผ่านเกณฑ์แล้ว {copied_count} ไฟล์ -> {output_dir}")
    return copied_count


# =========================================================================
# ฟีเจอร์ 4: จัดการ Class Imbalance
# =========================================================================
def check_class_distribution(input_dir):
    """
    นับจำนวนรูปในแต่ละ class (แต่ละ subfolder ของ input_dir)
    ใช้ดูก่อนว่า class ไหนมีรูปน้อย/มากผิดปกติ (Class Imbalance)

    คืนค่า dict {class_name: จำนวนไฟล์}
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {input_dir}")

    distribution = {}
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(VALID_EXT)])
            distribution[class_name] = count

    print("[Class Distribution]")
    for class_name, count in sorted(distribution.items(), key=lambda x: -x[1]):
        print(f"   {class_name}: {count} ไฟล์")

    return distribution


def oversample_class(class_dir, target_count, dry_run=False):
    """
    เพิ่มไฟล์ใน class_dir ให้ครบ target_count โดย copy ไฟล์เดิมซ้ำแบบสุ่ม
    (ไม่ได้สร้างภาพใหม่ แค่ duplicate ไฟล์ที่มีอยู่ เพื่อให้จำนวนเท่ากันทุก class)

    วิธีอื่นที่เลือกใช้แทนได้ (แล้วแต่ความเหมาะสมของ dataset):
      - Undersampling: ตัดไฟล์ class ที่เยอะเกินให้เหลือเท่า class ที่น้อยที่สุด
      - Weighted Sampling: ไม่ต้องเพิ่ม/ลดไฟล์จริง แต่ปรับ weight ตอน train แทน

    dry_run=True  -> รายงานว่าจะเพิ่มกี่ไฟล์ ไม่ copy จริง
    dry_run=False -> copy ไฟล์ซ้ำจริง
    """
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
    """
    เช็ค class distribution แล้ว oversample ทุก class ที่มีไฟล์น้อยกว่า
    ให้เท่ากับ class ที่มีไฟล์เยอะที่สุด (ใช้วิธี oversampling เป็นค่าเริ่มต้น)

    เรียกฟังก์ชันนี้หลังจาก copy_valid_images() แล้ว
    (ทำงานกับข้อมูลที่ผ่านการคัดกรองแล้วเท่านั้น ไม่ใช่ raw data)
    """
    distribution = check_class_distribution(input_dir)
    if not distribution:
        print("[Balance Classes] ไม่พบ class ใดเลยใน", input_dir)
        return

    max_count = max(distribution.values())
    print(f"[Balance Classes] จะปรับทุก class ให้มีไฟล์เท่ากับ class ที่เยอะที่สุด ({max_count} ไฟล์)")

    for class_name in distribution:
        class_dir = os.path.join(input_dir, class_name)
        oversample_class(class_dir, max_count, dry_run=dry_run)
        


if __name__ == "__main__":
    # 0. หา path ของ dataset ใน Kaggle cache อัตโนมัติ (ไม่ต้องพิมพ์ path เอง)
    raw_dir = find_kaggle_cache_path("nikolasgegenava/cat-breeds")

    # dataset นี้มีรูปอยู่ใน subfolder ชื่อ "cat-breeds" ตามที่ data_collection.py ของทีมระบุไว้
    TEST_DIR = os.path.join(raw_dir, "cat-breeds")
    OUTPUT_DIR = "data/cleaned"

    # ---- ฟีเจอร์ 1: ไฟล์เสีย ----
    # ขั้นแรกลอง dry_run=True ก่อน เพื่อดูว่าจะลบอะไรบ้างโดยยังไม่ลบจริง
    remove_corrupted_images(TEST_DIR, dry_run=True)
    # ถ้าเช็คแล้วโอเค ค่อยรันจริงโดยเปลี่ยนเป็น dry_run=False
    # remove_corrupted_images(TEST_DIR, dry_run=False)

    # ---- ฟีเจอร์ 2: รูปซ้ำ ----
    # ควรรันหลังลบไฟล์เสียแล้ว จะได้ไม่เสียเวลา hash ไฟล์ที่เสียอยู่แล้ว
    duplicates = find_duplicate_images(TEST_DIR)
    remove_duplicate_images(duplicates, dry_run=True)
    # remove_duplicate_images(duplicates, dry_run=False)

    # ---- ฟีเจอร์ 3: ดึงไฟล์ที่ผ่านเกณฑ์ไปโฟลเดอร์ใหม่ ----
    # รันหลังจากลบไฟล์เสีย/ซ้ำจริงแล้วเท่านั้น (dry_run=False ทั้งสองขั้นตอนข้างบน)
    # copy_valid_images(TEST_DIR, OUTPUT_DIR)

    # ---- ฟีเจอร์ 4: จัดการ Class Imbalance ----
    # ทำงานกับ OUTPUT_DIR (ข้อมูลที่คัดกรองแล้ว) ไม่ใช่ TEST_DIR (raw)
    # ขั้นแรกลอง dry_run=True เพื่อดูว่าจะเพิ่มไฟล์กี่ไฟล์ต่อ class ก่อน
    # check_class_distribution(OUTPUT_DIR)
    # balance_classes(OUTPUT_DIR, dry_run=True)
    # balance_classes(OUTPUT_DIR, dry_run=False)