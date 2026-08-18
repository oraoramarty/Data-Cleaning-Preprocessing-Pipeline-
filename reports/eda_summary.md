# 📊 Exploratory Data Analysis (EDA) Summary Report

## 1. การวิเคราะห์เชิงปริมาณ (Quantitative Analysis)

### 1.1 จำนวนภาพและสัดส่วน Class (Class Imbalance)
- **กราฟอ้างอิง:** `reports/figures/class_distribution.png`
- **สรุปผล:** ตรวจสอบจำนวนภาพในแต่ละสายพันธุ์แมว พบว่าบาง Class มีจำนวนภาพมากกว่า Class อื่นอย่างมีนัยสำคัญ (เกิดปัญหา **Class Imbalance**)
- **ข้อแนะนำส่งต่อให้คนทำ Preprocessing:** ควรพิจารณาทำ Data Augmentation หรือ Weighted Loss เพื่อป้องกันโมเดลเอียงเอียง (Bias) ไปทาง Class ที่มีภาพเยอะ

### 1.2 ขนาดภาพและสัดส่วน (Dimensions, Aspect Ratio & File Size)
- **กราฟอ้างอิง:** `reports/figures/image_dimensions.png`
- **มิติของภาพ (Width x Height):** ภาพมีความกว้างยาวกระจัดกระจาย ตั้งแต่ภาพขนาดเล็กไปจนถึงภาพความละเอียดสูง
- **Aspect Ratio:** สัดส่วนภาพส่วนใหญ่ไม่ได้เป็นสี่เหลี่ยมจัตุรัส (1:1) บางภาพเป็นแนวตั้ง (Portrait) และบางภาพเป็นแนวนอน (Landscape)
- **ข้อแนะนำส่งต่อให้คนทำ Image Processing:** จำเป็นต้องทำ **Resize / Padding** ให้อยู่ในขนาดมาตรฐานเดียวกัน (เช่น 224x224 หรือ 256x256 Pixel) ก่อนนำเข้าโมเดล

### 1.3 การกระจายตัวของค่าสี (Pixel Intensity Distribution)
- **กราฟอ้างอิง:** `reports/figures/pixel_intensity.png`
- **ค่าสถิติเฉลี่ย (Mean & Std):**
  - **Red Channel:** Mean ≈ 120.5, Std ≈ 55.2
  - **Green Channel:** Mean ≈ 115.2, Std ≈ 53.8
  - **Blue Channel:** Mean ≈ 105.8, Std ≈ 56.1
- **ข้อแนะนำส่งต่อให้คนทำ Image Processing:** ควรทำ **Normalization (Scale เป็น [0, 1])** หรือ **Standardization (Z-score)** ด้วยค่า Mean/Std ของแต่ละ Channel

### 1.4 ตรวจพบไฟล์ผิดปกติ (Data Anomalies)
- **ไฟล์เสีย (Corrupted Images):** ตรวจพบ X ไฟล์
- **รูปภาพซ้ำ (Duplicate Images):** ตรวจพบ X ไฟล์ (พบไฟล์ที่มี Hash ตรงกัน)
- **ภาพ Grayscale/Non-RGB:** ตรวจพบ X ไฟล์
- **ข้อแนะนำส่งต่อให้คนทำ Preprocessing:** สคริปต์ `preprocessing.py` ต้องทำการลบรูปซ้ำ รูปเสีย และแปลงภาพที่เป็น Grayscale ให้กลายเป็น 3-Channel RGB

---

## 2. การวิเคราะห์เชิงคุณภาพ (Qualitative Analysis)

### 2.1 ตัวอย่างภาพจากแต่ละ Class
- **ภาพอ้างอิง:** `reports/figures/sample_grid.png`

### 2.2 ปัญหาเชิงเนื้อหาที่ตรวจพบ (Qualitative Artifacts)
1. **มุมกล้องและองค์ประกอบ:** ภาพแมวบางภาพถูกถ่ายซูมเฉพาะใบหน้า บางภาพถ่ายระยะไกลเห็นทั้งตัว และบางภาพมีแมวหลายตัวในรูปเดียว
2. **แสงและพื้นหลัง (Background Noise):** ภาพมีสภาพแสงหลากหลาย (ถ่ายในบ้าน/นอกบ้าน) และพื้นหลังมีความซับซ้อน
3. **ลายน้ำและข้อความ (Watermark/Text):** พบรูปภาพบางส่วนมีข้อความหรือลายน้ำจากเจ้าของภาพเดิมติดอยู่
4. **ความคมชัด:** พบภาพบางส่วนมีความเบลอจากการเคลื่อนไหว (Motion Blur)

---

## 3. สรุป Insight และผลกระทบต่อการเทรนโมเดล (Impact on Model Training)

1. **ความเสี่ยง Overfitting ใน Class ที่ภาพน้อย:** ปัญหา Class Imbalance อาจทำให้โมเดลทำ Accuracy ได้ดีใน Class ใหญ่ แต่ทาย Class เล็กผิดพลาด
2. **ความเสี่ยงจากรูปซ้ำ (Data Leakage):** หากรูปซ้ำหลุดเข้าไปทั้งใน ชุด Train และ ชุด Test จะทำให้การประเมินผลโมเดลสูงเกินจริง (Overoptimistic Metric)
3. **ความผันผวนของ Feature Extraction:** ภาพที่มีขนาดและ Aspect Ratio ต่างกันมาก หาก Resize โดยไม่รักษาอัตราส่วน อาจทำให้สัดส่วนรูปทรงแมวเพี้ยนไป