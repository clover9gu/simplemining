
import matplotlib
matplotlib.use('TKAgg')
from mlxtend.tf_classifier import TfSoftmaxRegression
from mlxtend.data import iris_data
from mlxtend.evaluate import plot_decision_regions
import matplotlib.pyplot as plt


X, y = iris_data()
X = X[:, [0, 3]]

X[:,0] = (X[:,0] - X[:,0].mean()) / X[:, 0].std()
X[:,1] = (X[:,1] - X[:,1].mean()) / X[:, 1].std()


# Gradient Descent

lr = TfSoftmaxRegression(eta = 0.75,
                         epochs=20,
                         print_progress=True,
                         minibatches=len(y),
                         random_seed=1)

lr.fit(X, y)

plt.plot(range(len(lr.cost_)), lr.cost_)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
