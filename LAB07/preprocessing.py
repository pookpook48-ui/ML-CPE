def preprocess_features(X_train, X_test):
    X_train = X_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
    X_test = X_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
    return X_train, X_test