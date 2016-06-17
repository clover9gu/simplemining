

from mlxtend.preprocessing import one_hot
import numpy as np

# defaults
y = np.array([0, 1, 2, 1, 2])
print one_hot(y)


# python lists
y = [0, 1, 2, 1, 2]
print one_hot(y)

# integer arrays
y = [0, 1, 2, 1, 2]
print one_hot(y, dtype='int')

# arbitray numbers of class labels
y = [0, 1, 2, 1, 2]
print one_hot(y, num_labels=10)