from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json

def build_and_train_nn(X_train, y_train, X_val, y_val, input_shape, save_path="outputs/nn_model.keras"):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32, verbose=1)
    model.save(save_path)
    
    with open("outputs/history.json", "w") as f:
        json.dump(history.history, f)
        
    return model, history.history