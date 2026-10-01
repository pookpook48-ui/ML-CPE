from data_loader import load_data
from preprocessing import preprocess_features
from split_data import preprocess_labels
from cnn_model import create_model
from evaluate import plot_training_history, plot_confusion_matrix, plot_prediction_samples

# 1. นำเข้าคำสั่ง EarlyStopping เพิ่มเติม
from tensorflow.keras.callbacks import EarlyStopping

X_train, y_train, X_test, y_test = load_data()
X_train, X_test = preprocess_features(X_train, X_test)
y_train, y_test = preprocess_labels(y_train, y_test)

model = create_model()

# 2. ตั้งค่าการหยุดเทรนอัตโนมัติ (หยุดเมื่อ val_loss ไม่ลดลงติดกัน 3 รอบ และดึงค่าน้ำหนักที่ดีที่สุดกลับมาใช้)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 3. ใส่ callbacks=[early_stop] เข้าไปตอนสั่งเทรน 
# (สังเกตว่าผมแกล้งตั้ง epochs=20 ไว้เลย แต่เดี๋ยวมันจะหยุดเทรนให้เองก่อนถึง 20 แน่นอนครับ)
history = model.fit(X_train, y_train, 
                    epochs=20, 
                    validation_data=(X_test, y_test), 
                    batch_size=64,
                    callbacks=[early_stop])

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

print("\n--- แสดงกราฟ Training History ---")
plot_training_history(history)

print("--- แสดง Confusion Matrix ---")
plot_confusion_matrix(model, X_test, y_test, class_names)

print("--- แสดง Prediction Samples ---")
plot_prediction_samples(model, X_test, y_test, class_names)