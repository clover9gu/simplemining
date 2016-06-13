
from mlxtend.data import iris_data
from mlxtend.data import mnist_data
from mlxtend.classifier import MultiLayerPerceptron as MLP
from mlxtend.evaluate import plot_decision_regions
from mlxtend.preprocessing import shuffle_arrays_unison, standardize

import matplotlib.pyplot as plt


X, y = iris_data()
X = X[:, [0, 3]]

# standardize training data
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

print X_std


# Gradient Descent

nn1 = MLP(hidden_layers=[50],
          l2 = 0.00,
          l1 = 0.0,
          epochs=150,
          eta=0.05,
          momentum=0.1,
          decrease_const=0.0,
          minibatches=1,
          random_seed=1,
          print_progress=3)

nn1 = nn1.fit(X_std, y)
fig = plot_decision_regions(X=X_std, y=y, clf=nn1, legend=2)
plt.title('Multi-layer perception w. 1 hidden layer (logistic sigmod)')
plt.show()

plt.plot(range(len(nn1.cost_)), nn1.cost_)
plt.ylabel("Cost")
plt.xlabel("Epochs")
plt.show()

print 'Accuracy: %.2f%%' % (100 * nn1.score(X_std, y))


# Stochastic Gradient Descent

nn2 = MLP(hidden_layers=[50],
          l2=0.00,
          l1=0.0,
          epochs=5,
          eta=0.005,
          momentum=0.1,
          decrease_const=0.0,
          minibatches=len(y),
          random_seed=1,
          print_progress=3)
nn2.fit(X_std, y)

fig = plot_decision_regions(X=X_std, y=y, clf=nn2, legend=2)
plt.title('Multi-layer perception w. 1 hidden layer (logistic sigmod)')
plt.show()

plt.plot(range(len(nn2.cost_)), nn2.cost_)
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()


# Classify handwritten digits from a 10% mnist subset

from mlxtend.data import mnist_data
from mlxtend.preprocessing import shuffle_arrays_unison, standardize


X, y = mnist_data()
X, y = shuffle_arrays_unison((X, y), random_seed=1)
X_train, y_train = X[:500], y[:500]
X_test, y_test = X[500:], y[500:]

def plot_digit(X, y, idx):
  img = X[idx].reshape(28, 28) # 784 => 28 * 28
  plt.imshow(img, cmap='Greys', interpolation='nearest')
  plt.title('true label: %d' % y[idx])
  plt.show()

plot_digit(X, y, 3500)

X_train_std, params = standardize(X_train, columns=range(X_train.shape[1]), return_params=True)
X_test_std = standardize(X_test, columns=range(X_test.shape[1]), params=params)

nn1 = MLP(hidden_layers=[150],
          l2=0.00,
          l1=0.0,
          epochs=100,
          eta=0.005,
          momentum=0.0,
          decrease_const=0.0,
          minibatches=100,
          random_seed=1,
          print_progress=3)

nn1.fit(X_train_std, y_train)
plt.plot(range(len(nn1.cost_)), nn1.cost_)
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

print 'Train Accuracy: %.2f%%' % (100 * nn1.score(X_train_std, y_train))
print 'Test Accuracy: %.2f%%' % (100 * nn1.score(X_test_std, y_test))
