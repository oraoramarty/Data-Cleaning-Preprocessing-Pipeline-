## 4.1 Data Collection & Setup Instructions

### ขั้นตอนการตั้งค่า Kaggle API Key
1. เข้าสู่ระบบ [Kaggle](https://www.kaggle.com/) -> ไปที่ **Account Settings**
2. เลื่อนไปยังหัวข้อ **API** แล้วกด **Create New API Token** (จะได้ไฟล์ `kaggle.json`)
3. นำไฟล์ `kaggle.json` ไปวางตาม Path ของระบบปฏิบัติการดังนี้:
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
   - **Mac/Linux:** `~/.kaggle/kaggle.json`
4. (สำหรับ Mac/Linux) กำหนดสิทธิ์ไฟล์ด้วยคำสั่ง:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json