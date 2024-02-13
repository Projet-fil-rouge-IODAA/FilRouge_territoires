from build_parameters import find_num_clusters
import seaborn as sns

data = sns.load_dataset('iris').iloc[:,0:4]

# Find the optimal number of clusters for the iris dataset

k = find_num_clusters(data, 1, 10, 'KMedoids', plot_elbow=True, max_iter=300, random_state=42)
k = find_num_clusters(data, 1, 10, 'KMeans', plot_elbow=True, n_init=10, max_iter=300, random_state=42)
k = find_num_clusters(data, 1, 10, 'Hierarchical', plot_elbow=True, method='ward')
k = find_num_clusters(data, 1, 10, 'Spectral', plot_elbow=True, random_state=42)

print(k)

