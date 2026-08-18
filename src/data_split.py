"""
data_split.py
=============
รับผิดชอบโดย: คนที่ 4 (feature/data-split)

หน้าที่:
  1. อ่านไฟล์ภาพทั้งหมดจาก output สุดท้ายของ pipeline (data/augmented/)
     ซึ่งมาจาก image_processing.py ของคนที่ 3
  2. แบ่งข้อมูลเป็น Train / Validation / Test แบบ Stratified
     (รักษาสัดส่วน class ในทุก subset)
  3. ป้องกัน Data Leakage: ไฟล์ที่มาจากภาพต้นฉบับเดียวกัน (ทั้งจาก
     oversampling ในขั้นตอน balance_classes() และจาก data augmentation
     ในขั้นตอน augment_dataset()) ต้องถูกจัดอยู่ใน Split เดียวกันเสมอ
     ไม่งั้นโมเดลจะ "แอบเห็น" ข้อมูลที่คล้ายกันมากข้าม Train/Test
  4. บันทึกผลลัพธ์เป็น Manifest CSV เพื่อให้ตรวจสอบย้อนหลังได้

เหตุผลที่ต้อง group ก่อน split (สำคัญมาก อย่าลบ):
  - balance_classes() ใน preprocessing.py จะ copy ไฟล์ต้นฉบับซ้ำเพื่อ oversample
    ตั้งชื่อไฟล์แบบ: aug_<i>_<ชื่อเดิม>.jpg   (เป็นไฟล์ที่เหมือนต้นฉบับ 100%)
  - augment_dataset() ใน image_processing.py จะสร้างภาพแปรผัน (flip/rotate/
    brightness/crop) จากทุกภาพใน data/cleaned (รวมถึงภาพที่ oversample มา)
    ตั้งชื่อไฟล์แบบ: <ชื่อเดิม>_aug<0/1/2>.jpg
  - ผลคือ 1 ภาพต้นฉบับอาจมี "ญาติ" ที่หน้าตาคล้ายกันมากหลายไฟล์ใน
    data/augmented/ ถ้าไม่ group ไว้ด้วยกันก่อน split จะเกิด Data Leakage
    (ภาพที่คล้ายกันมากไปอยู่คนละ Split)
"""

import os
import re
import sys
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
VALID_EXT = (".jpg", ".jpeg", ".png")


# =========================================================================
# 1. โหลดรายชื่อไฟล์ทั้งหมดจากโฟลเดอร์ output สุดท้ายของ pipeline
# =========================================================================
def load_images_from_folder(input_dir: str) -> pd.DataFrame:
    """
    สแกน input_dir (เช่น data/augmented) แบบ recursive
    คืนค่า DataFrame: file_path, category (= ชื่อ subfolder ที่เก็บไฟล์)
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์: {input_dir} "
            f"(ต้องรัน image_processing.py ให้เสร็จก่อน เพื่อสร้าง data/augmented/)"
        )

    records = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(VALID_EXT):
                records.append({
                    "file_path": os.path.join(root, fname),
                    "category": os.path.basename(root),
                    "file_name": fname,
                })

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise ValueError(f"ไม่พบไฟล์ภาพใน {input_dir} เลย กรุณาตรวจสอบว่ารัน pipeline ครบหรือยัง")

    print(f"[INFO] โหลดไฟล์จาก {input_dir} ได้ {len(df)} ไฟล์, {df['category'].nunique()} class")
    return df


# =========================================================================
# 2. หา "group_id" = อัตลักษณ์ของภาพต้นฉบับ (ตัด prefix/suffix ที่ pipeline สร้างเพิ่ม)
# =========================================================================
_OVERSAMPLE_PREFIX = re.compile(r"^aug_\d+_")   # จาก balance_classes() ใน preprocessing.py
_AUGMENT_SUFFIX = re.compile(r"_aug\d+$")       # จาก augment_dataset() ใน image_processing.py


def extract_group_id(file_name: str) -> str:
    """
    ตัดชื่อไฟล์กลับไปหา "ภาพต้นฉบับ" เพื่อใช้เป็น group key ตอน split
    ตัวอย่าง:
      persian_012.jpg              -> persian_012.jpg
      persian_012_aug0.jpg         -> persian_012.jpg
      aug_1_persian_012.jpg        -> persian_012.jpg
      aug_1_persian_012_aug2.jpg   -> persian_012.jpg
    """
    name, ext = os.path.splitext(file_name)
    name = _OVERSAMPLE_PREFIX.sub("", name)
    name = _AUGMENT_SUFFIX.sub("", name)
    return name + ext


# =========================================================================
# 3. Stratified Split ที่ระดับ "กลุ่ม" (ป้องกัน Data Leakage)
# =========================================================================
def stratified_group_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "สัดส่วน train+val+test ต้องรวมกันเป็น 1.0"

    df = df.copy()
    df["group_id"] = df["file_name"].apply(extract_group_id)

    # ทำ 1 แถวต่อ 1 กลุ่ม (แทนภาพต้นฉบับ) เพื่อ split ที่ระดับกลุ่มเท่านั้น
    # หมายเหตุ: ทุกไฟล์ในกลุ่มเดียวกันต้องมี category เดียวกันเสมออยู่แล้ว
    # เพราะ oversample/augment ไม่เปลี่ยน class ของภาพ
    group_df = df.drop_duplicates(subset=["category", "group_id"])[["category", "group_id"]]

    # split ครั้งที่ 1: แยก train ออกจาก (val + test) ที่ระดับกลุ่ม
    train_groups, temp_groups = train_test_split(
        group_df,
        train_size=train_size,
        stratify=group_df["category"],
        random_state=seed,
    )

    # split ครั้งที่ 2: แยก val กับ test จากส่วนที่เหลือ ที่ระดับกลุ่มเช่นกัน
    relative_val_size = val_size / (val_size + test_size)
    val_groups, test_groups = train_test_split(
        temp_groups,
        train_size=relative_val_size,
        stratify=temp_groups["category"],
        random_state=seed,
    )

    group_to_split = {}
    for gid in train_groups["group_id"]:
        group_to_split[gid] = "train"
    for gid in val_groups["group_id"]:
        group_to_split[gid] = "val"
    for gid in test_groups["group_id"]:
        group_to_split[gid] = "test"

    # map กลับไปยังทุกไฟล์ (ไฟล์ในกลุ่มเดียวกันได้ split เดียวกันเสมอ)
    df["split"] = df["group_id"].map(group_to_split)

    n_groups = len(group_df)
    print(f"[INFO] จำนวนภาพต้นฉบับ (groups) ทั้งหมด: {n_groups}")
    print(f"[INFO]   -> train groups: {len(train_groups)} | val groups: {len(val_groups)} | test groups: {len(test_groups)}")
    print(f"[INFO] จำนวนไฟล์ทั้งหมดหลังขยายกลับ (รวม oversample+augment): {len(df)}")

    return df


# =========================================================================
# 4. ตรวจสอบว่าไม่มี group_id ใดหลุดข้าม split (sanity check ป้องกัน leakage)
# =========================================================================
def assert_no_leakage(df: pd.DataFrame):
    leak = df.groupby("group_id")["split"].nunique()
    leaked_groups = leak[leak > 1]
    if len(leaked_groups) > 0:
        raise AssertionError(
            f"[LEAKAGE DETECTED] พบ {len(leaked_groups)} กลุ่มที่กระจายไปมากกว่า 1 split: "
            f"{list(leaked_groups.index[:5])} ..."
        )
    print("[CHECK PASSED] ไม่มี group ใดหลุดข้าม split — ไม่มี Data Leakage")


# =========================================================================
# 5. สรุปสัดส่วน class ในแต่ละ split (เอาไปทำกราฟใน slide หัวข้อ 7.6.2 ได้)
# =========================================================================
def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["split", "category"])
        .size()
        .unstack(fill_value=0)
        .T
    )
    summary = summary[[c for c in ["train", "val", "test"] if c in summary.columns]]
    print("\n[SUMMARY] จำนวนไฟล์ต่อ Class ในแต่ละ Split:")
    print(summary.to_string())
    return summary


def plot_split_balance(summary_df: pd.DataFrame, save_path: str = "reports/figures/split_balance.png"):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    proportions = summary_df.div(summary_df.sum(axis=1), axis=0)

    ax = proportions.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis")
    ax.set_ylabel("Proportion within class")
    ax.set_title("Train / Val / Test Proportion per Class (Stratified Check)")
    ax.legend(title="Split", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[SUCCESS] เซฟกราฟยืนยันสัดส่วน Split ที่: {save_path}")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset เป็น Train/Val/Test แบบไม่มี Data Leakage")
    parser.add_argument("--input-dir", default="data/augmented",
                         help="โฟลเดอร์ผลลัพธ์สุดท้ายจาก pipeline (default: data/augmented)")
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--output-csv", default="reports/split_manifest.csv")
    args = parser.parse_args()

    df = load_images_from_folder(args.input_dir)

    result_df = stratified_group_split(
        df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    assert_no_leakage(result_df)

    summary_df = summarize_split(result_df)
    plot_split_balance(summary_df)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    manifest = result_df[["file_path", "category", "group_id", "split"]]
    manifest.to_csv(args.output_csv, index=False)
    print(f"\n[SUCCESS] บันทึก Manifest ที่: {args.output_csv} ({len(manifest)} แถว)")
