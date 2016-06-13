
from mlxtend.data import iris_data
from mlxtend.evaluate import plot_decision_regions
from mlxtend.classifier import Adaline
import matplotlib.pyplot as plt

X, y = iris_data()
X = X[:, [0, 3]]
X = X[0:100]
y = y[0:100]

X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


# Closed Form Solution
ada = Adaline(epochs=30,
              eta=0.01,
              minibatches=None,
              random_seed=1)

ada.fit(X, y)
plot_decision_regions(X, y, clf=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.show()



# (Stochastic) Gradient Descent
ada2 = Adaline(epochs=30,
               eta=0.01,
               minibatches=1, # 1 for GD learning
               #minibatches=len(y), # len(y) for SGD learning
               #minibatches=5, # for SGD learning w. minibatch size 20
               random_seed=1,
               print_progress=3)
ada2.fit(X, y)
plot_decision_regions(X, y, clf=ada2)
plt.title('Adaline - SGD')
plt.show()

plt.plot(range(len(ada2.cost_)), ada2.cost_)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

