import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA

def plot_elbow(wcss, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
    plt.savefig(os.path.join(output_dir, '01_elbow.png'))
    plt.close()

def plot_clusters(X_scaled, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.savefig(os.path.join(output_dir, '02_clusters.png'))
    plt.close()

def save_csv_outputs(df, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df_out = df.copy()
    df_out['Cluster'] = labels
    df_out.to_csv(os.path.join(output_dir, 'clustered_animals.csv'), index=False)
    summary = df_out.groupby('Cluster').mean()
    summary.to_csv(os.path.join(output_dir, 'cluster_summary.csv'))