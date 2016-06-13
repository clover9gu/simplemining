
from mlxtend.data import iris_data
from mlxtend.evaluate import plot_decision_regions
from mlxtend.classifier import LogisticRegression

import matplotlib.pyplot as plt

X, y = iris_data()

X = X[:, [0, 3]]
X = X[0:100]
y = y[0:100]

X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

lr = LogisticRegression(eta = 0.1,
                        l2_lambda=0.0,
                        epochs=500,
                        #minibatches=1, # 1 for Gradient Descent
                        #minibatches=len(y), #  len(y) for SGD learning
                        minibatches=5, # 100/5 = 20 -> minibatch-s
                        random_seed=1,
                        print_progress=3)
lr.fit(X, y)


y_pred = lr.predict(X)
print('Last 3 Class Labels: %s' % y_pred[-3:])
y_pred = lr.predict_proba(X)
print('Last 3 Class Labels: %s' % y_pred[-3:])


plot_decision_regions(X, y, clf=lr)
plt.title("Logistic regression - gd")
plt.show()

plt.plot(range(len(lr.cost_)), lr.cost_)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()




