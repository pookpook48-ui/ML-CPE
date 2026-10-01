from tensorflow.keras.utils import to_categorical

def preprocess_labels(y_train, y_test):
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return y_train, y_test