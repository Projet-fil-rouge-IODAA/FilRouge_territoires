from build_parameters import find_num_clusters
import seaborn as sns

data = sns.load_dataset('iris').iloc[:,0:4]

# Find the optimal number of clusters for the iris dataset

# k = find_num_clusters(data, 1, 10, 'KMedoids', max_iter=300, random_state=42)
# k = find_num_clusters(data, 1, 10, 'KMeans', n_init=10, max_iter=300, random_state=42)
# k = find_num_clusters(data, 1, 10, 'Hierarchical', method='ward')
k = find_num_clusters(data, 1, 10, 'Spectral', random_state=42)

print(k)

