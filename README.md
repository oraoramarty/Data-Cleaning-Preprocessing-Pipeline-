# Data-Cleaning-Preprocessing-Pipeline

โปรเจกต์ Data Cleaning & Preprocessing Pipeline สำหรับชุดข้อมูลรูปภาพ **Cat Breeds**
ดึงข้อมูลจาก Kaggle ผ่าน API โดยอัตโนมัติ (ไม่มีการแนบไฟล์รูปภาพไว้ใน Repository)
ครอบคลุมตั้งแต่ EDA, Data Cleaning, Image Processing ไปจนถึง Train/Val/Test Split
พร้อม Manifest สำหรับตรวจสอบย้อนหลัง

## 1. ที่มาของ Dataset

- **Kaggle Dataset:** [nikolasgegenava/cat-breeds](https://www.kaggle.com/datasets/nikolasgegenava/cat-breeds)
- ดึงข้อมูลผ่านไลบรารี [`kagglehub`](https://pypi.org/project/kagglehub/) โดยอัตโนมัติเมื่อรัน `src/data_collection.py`, `src/preprocessing.py`, หรือ `src/eda.py` (ไม่ต้องดาวน์โหลดเองด้วยมือ)

### วิธี Setup Kaggle API Key (ต้องทำก่อนรัน Code ทุกครั้ง)

1. เข้าสู่ระบบ [Kaggle](https://www.kaggle.com/) → ไปที่ **Account Settings**
2. เลื่อนไปยังหัวข้อ **API** แล้วกด **Create New API Token** (จะได้ไฟล์ `kaggle.json`)
3. นำไฟล์ `kaggle.json` ไปวางตาม Path ของระบบปฏิบัติการดังนี้:
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
   - **Mac/Linux:** `~/.kaggle/kaggle.json`
4. (สำหรับ Mac/Linux) กำหนดสิทธิ์ไฟล์ด้วยคำสั่ง:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

> ⚠️ **ห้าม commit ไฟล์ `kaggle.json` หรือฝัง API Key ไว้ใน Code ที่ push ขึ้น GitHub โดยเด็ดขาด**

## 2. วิธีติดตั้งและวิธีรัน Code

### ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

หรือบน Windows ใช้ไฟล์ `install_requirements.bat` ได้เลย

### รัน Pipeline ตามลำดับ (สำคัญ: ต้องรันจาก root ของ repo)

```bash
# 1. EDA เชิงปริมาณและเชิงคุณภาพ (ดึง dataset อัตโนมัติ + เซฟกราฟใน reports/figures/)
python -m src.eda

# 2. Data Cleaning: ลบไฟล์เสีย/ซ้ำ, จัดการ Class Imbalance, แปลง format/color space
#    -> output: data/cleaned/<breed>/*.jpg
python src/preprocessing.py

# 3. Image Processing: resize, denoise, normalize, data augmentation
#    -> output: data/augmented/<breed>/*.jpg
python src/image_processing.py

# 4. Data Splitting: แบ่ง Train/Val/Test แบบ Stratified และป้องกัน Data Leakage
#    -> output: reports/split_manifest.csv
python src/data_split.py
```

แต่ละขั้นตอนต้องรันสำเร็จก่อนไปขั้นถัดไป เพราะแต่ละ script ใช้ output ของขั้นก่อนหน้าเป็น input

## 3. โครงสร้างโฟลเดอร์ของ Repository

```
Data-Cleaning-Preprocessing-Pipeline/
├── README.md
├── requirements.txt
├── install_requirements.bat
├── .gitignore
├── src/
│   ├── data_collection.py     # ดึง dataset จาก Kaggle ผ่าน kagglehub
│   ├── eda.py                 # Exploratory Data Analysis (เชิงปริมาณ+คุณภาพ)
│   ├── preprocessing.py       # Data Cleaning (corrupted, duplicate, class imbalance, standardize)
│   ├── image_processing.py    # Resize, Denoise, Normalize, Augmentation
│   └── data_split.py          # Train/Val/Test Split (Stratified, Leakage-safe)
├── reports/
│   ├── eda_summary.md
│   ├── split_manifest.csv
│   └── figures/                # กราฟสรุปทุกขั้นตอน (ไม่ใช่ dataset จริง)
└── slides/                     # ไฟล์นำเสนอ (PDF/PPTX)
```

> หมายเหตุ: โฟลเดอร์ `data/` (raw/cleaned/resized/denoised/augmented) จะถูกสร้างขึ้นอัตโนมัติตอนรัน Code และ**ไม่ถูก commit ขึ้น GitHub** (ควบคุมด้วย `.gitignore`)

## 4. รายชื่อสมาชิกกลุ่มและหน้าที่รับผิดชอบ

| สมาชิก | Branch | ขอบเขตงาน |
|---|---|---|
| คนที่ 1 | `feature/data-collection` | เขียน Script ดึง Dataset จาก Kaggle API + จัดโครงสร้างโฟลเดอร์ข้อมูล |
| คนที่ 2 | `feature/eda` | EDA เชิงปริมาณและเชิงคุณภาพ พร้อมสรุปผล |
| คนที่ 3 | `feature/preprocessing` | Data Cleaning, Image Processing (resize, denoise, augment ฯลฯ) |
| คนที่ 4 | `feature/data-split` | Train/Val/Test Split ตามหลักการ + จัดทำ Report สรุปทั้งหมด (README/Slide) |

## 5. หมายเหตุเทคนิค: การป้องกัน Data Leakage ในขั้นตอน Split

Pipeline นี้มีการ **Oversample** (ใน `preprocessing.py`) และ **Data Augmentation** (ใน `image_processing.py`)
ซึ่งสร้างไฟล์ที่คล้ายกันมากจากภาพต้นฉบับเดียวกัน (เช่น `cat_01.jpg`, `cat_01_aug0.jpg`, `aug_0_cat_01.jpg`)
`data_split.py` จึงทำการ **Group ไฟล์ที่มาจากภาพต้นฉบับเดียวกันไว้ด้วยกันก่อน Split** เพื่อไม่ให้ภาพที่คล้ายกันมาก
หลุดไปอยู่คนละ Subset (ซึ่งจะทำให้ผลประเมินโมเดลสูงเกินจริง) พร้อมมีการตรวจสอบอัตโนมัติ (`assert_no_leakage`)
ทุกครั้งที่รัน
