import pandas as pd

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # 1. หาคอลัมน์เป้าหมาย (y) ที่เก็บข้อมูลคำว่า 'Dog' หรือ 'Cat'
    target_col = None
    for col in df.columns:
        # เช็คว่าคอลัมน์นี้มีคำว่า Dog อยู่ไหม
        if df[col].astype(str).str.contains('Dog').any():
            target_col = col
            break
            
    # ถ้าหาไม่เจอจริงๆ ให้ดึงคอลัมน์สุดท้ายมาแทน
    if target_col is None:
        target_col = df.columns[-1] 
        
    y = df[target_col]
    
    # 2. เตรียมข้อมูลฟีเจอร์ (X)
    # ลบคอลัมน์คำตอบทิ้งไป และเลือกดึงมาเฉพาะคอลัมน์ที่เป็น "ตัวเลข" เท่านั้น (ป้องกัน Error String)
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=['number'])
    
    return X, y

if __name__ == "__main__":
    X, y = load_and_preprocess_data('data-animal/animal_dataset.csv')
    print("โหลดข้อมูลสำเร็จ!")
    print(f"จำนวนข้อมูลทั้งหมด: {len(X)} แถว")