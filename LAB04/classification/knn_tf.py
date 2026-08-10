from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def train_knn(X_train, y_train, k=5):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    return knn, scaler