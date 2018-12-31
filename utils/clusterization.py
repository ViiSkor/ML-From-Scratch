import numpy as np

def init_centroids(data, n_centroids, mode="random_sample"):
        """ Initialize the centroids.
        Has two mode: take n random samples of data as the centroids and 
        random sharing.
        
        Read about random sharing:
        https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html
        
        Args:
            data: numpy array
                2d dataset array, where a horizontal axis is equal to the 
                number of features, a vertical axis is equal to the number of 
                dataset samples.
            n: int
                The number of clusters the algorithm will form.
                
        Returns:
            centroids: numpy 0d array
                An array stores the init coordinates of the centroids.
        """
        
        if mode == "random_sample":
            index = np.random.choice(data.shape[0], n_centroids, replace=False)
            return data[index]
        elif mode == "sharding_init":
            attr_sum = np.sort(np.sum(data, axis=1))
            clusters_idx = np.array_split([*range(len(attr_sum))], n_centroids)
            centroids = []
            for idx in clusters_idx:
                centroids.append(np.mean(data[idx], axis=0))      
            return np.array(centroids)
        else:
            raise Exception('No such init!')