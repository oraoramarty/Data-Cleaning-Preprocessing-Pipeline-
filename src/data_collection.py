import os
import pandas as pd
import kagglehub

DATASET_HANDLE = "nikolasgegenava/cat-breeds"

def load_dataset():
    """
    ดึง Dataset จาก Kaggle ผ่าน kagglehub เข้าสู่ระบบโดยตรง
    """
    print(f"[INFO] กำลังเชื่อมต่อและโหลดข้อมูลจาก Kaggle ({DATASET_HANDLE})...")
    
    try:
        # 1. โหลด Dataset เข้า Cache
        dataset_path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f"[SUCCESS] เชื่อมต่อ Dataset สำเร็จ!")
        print(f"[INFO] Dataset Path: {dataset_path}")
        
        # 2. ระบุ Path ของโฟลเดอร์รูปภาพ และไฟล์ CSV
        images_dir = os.path.join(dataset_path, "cat-breeds")
        csv_path = os.path.join(dataset_path, "dataset_stats.csv")
        
        # 3. โหลด Metadata เข้า Pandas DataFrame (ถ้ามี)
        df = None
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"[SUCCESS] โหลดไฟล์ Metadata (dataset_stats.csv) เรียบร้อย! จำนวน {len(df)} รายการ")
            print(df.head(3))
        
        return images_dir, df

    except Exception as e:
        print(f"[ERROR] ไม่สามารถดึงข้อมูลได้: {e}")
        return None, None

if __name__ == "__main__":
    images_dir, df = load_dataset()