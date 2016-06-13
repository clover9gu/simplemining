
from mlxtend.classifier import NeuralNetMLP

from mlxtend.data import iris_data
from mlxtend.evaluate import plot_decision_regions

import matplotlib.pyplot as plt
import numpy as np

X, y = iris_data()

X = X[:, [0, 3]]
X = X[0:100]
y = y[0:100]

X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

nn = NeuralNetMLP(n_output = len(np.unique(y)),
                  n_features = X.shape[1],
                  n_hidden = 50,
                  l2 = 0.00,
                  l1 = 0.0,
                  epochs  = 300,
                  eta = 0.01,
                  alpha = 0.0,
                  decrease_const = 0.0,
                  minibatches = 1,
                  shuffle_init = False,
                  shuffle_epoch = False,
                  random_seed = 1,
                  print_progress = 3)

nn = nn.fit(X, y)

y_pred = nn.predict(X)

acc = np.sum(y == y_pred, axis=0) / X.shape[0]
print('Accuracy: %.2f%%' % (acc * 100))


plot_decision_regions(X, y, clf=nn, legend=2)
plt.title("Logistic regression - gd")
plt.show()

plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 300])
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()

