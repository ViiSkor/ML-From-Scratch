import numpy as np

def euclidean(x1, x2):
        """ Finds the Euclidean distance between two samples.
        
        Args:
            x1: numpy.array (1, n_features)
                An array of the coordinates of the first sample, where 
                n_features is the number 
            of features.
            x1: numpy.array 1, n_features)
                An array of the coordinates of the second sample, where 
                n_features is the number 
                
        Returns:
            numpy.float: Euclidean distance.
        """
     
        diff = x1 - x2
        return np.sqrt(np.dot(diff.T, diff))