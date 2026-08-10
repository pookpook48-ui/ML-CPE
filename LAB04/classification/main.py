import os
from sklearn.model_selection import train_test_split
import data_loader
import knn_tf
import evaluate

def main():
    # 1. กำหนดโฟลเดอร์สำหรับเซฟผลลัพธ์ (จะสร้างโฟลเดอร์ outputs ให้อัตโนมัติ)
    output_dir = 'classification/outputs'
    
    # 2. โหลดข้อมูล (ดึงฟังก์ชันมาจากไฟล์ data_loader.py)
    print("กำลังโหลดข้อมูล...")
    # อ้างอิง path จากโฟลเดอร์หลัก
    X, y = data_loader.load_and_preprocess_data('data-animal/animal_dataset.csv')
    
    # 3. แบ่งข้อมูล Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. ทดสอบหาค่า K ที่ดีที่สุด
    k_values = [3, 5, 7, 9, 11]
    accuracies = []
    
    best_k = 3
    best_acc = 0
    best_y_pred = None
    
    print("\nกำลังเทรนและประเมินโมเดล...")
    for k in k_values:
        # เรียกฟังก์ชันเทรนจากไฟล์ knn_tf.py
        knn, scaler = knn_tf.train_knn(X_train, y_train, k=k)
        
        # นำข้อมูล Test ไป Standardize ด้วยสเกลเดิมก่อนพยากรณ์
        X_test_scaled = scaler.transform(X_test)
        y_pred = knn.predict(X_test_scaled)
        
        # ประเมินความแม่นยำจากไฟล์ evaluate.py
        acc = evaluate.evaluate_accuracy(y_test, y_pred)
        accuracies.append(acc)
        print(f"Accuracy (k={k}): {acc:.4f}")
        
        # เก็บค่าที่ดีที่สุดไว้สร้างกราฟ
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_y_pred = y_pred
            
    print(f"\nค่า K ที่ดีที่สุดคือ: {best_k} (ความแม่นยำ {best_acc:.4f})")
    
    # 5. สร้างกราฟและเซฟไฟล์ลงใน outputs/ (เรียกใช้ฟังก์ชันจาก evaluate.py)
    print("\nกำลังบันทึกไฟล์ผลลัพธ์...")
    evaluate.save_k_curve(k_values, accuracies, output_dir)
    evaluate.save_confusion_matrix(y_test, best_y_pred, output_dir)
    evaluate.save_predictions(X_test, y_test, best_y_pred, output_dir)
    
    print(f"บันทึกไฟล์ทั้งหมดลงในโฟลเดอร์ '{output_dir}' เรียบร้อยแล้ว! ปิดจ๊อบ Classification 🎉")

if __name__ == "__main__":
    main()