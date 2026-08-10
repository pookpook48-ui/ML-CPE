import os
import data_loader
import knn_tools
import kmeans_tf
import visualize

def main():
    output_dir = 'clustering/outputs'
    filepath = 'data-animal/animal_dataset.csv'
    
    print("Loading data...")
    X = data_loader.load_data(filepath)
    
    print("Calculating WCSS for Elbow Method...")
    wcss = knn_tools.calculate_wcss(X, max_k=10)
    visualize.plot_elbow(wcss, output_dir)
    
    best_k = 3
    print(f"Training K-Means with k={best_k}...")
    kmeans, scaler, X_scaled = kmeans_tf.train_kmeans(X, n_clusters=best_k)
    labels = kmeans.labels_
    
    print("Saving visualizations and CSV outputs...")
    visualize.plot_clusters(X_scaled, labels, output_dir)
    visualize.save_csv_outputs(X, labels, output_dir)
    
    print(f"Clustering complete! All files saved to '{output_dir}'.")

if __name__ == "__main__":
    main()