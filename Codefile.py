import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def sgd(X, y, learning_rate, iterations):
    m = len(y)
    theta = np.random.randn(2, 1)
    for iteration in range(iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradients
    return theta

X_b = np.c_[np.ones((len(X_train), 1)), X_train]
theta = sgd(X_b, y_train, learning_rate=0.01, iterations=1000)

X_new_b = np.c_[np.ones((len(X_test), 1)), X_test]
y_predict = X_new_b.dot(theta)

mse = mean_squared_error(y_test, y_predict)
print("Mean Squared Error:", mse)

plt.plot(X_test, y_test, "b.")
plt.plot(X_test, y_predict, "r-")
plt.show()

#END OF CODE
#COPY MAT KARNA NA PLEASE, UMMMM...... NHI NHI COPY KARLENA PAR EK BAAR BATA DENA CONNECT KARKE IF YOU WISH TO PLEASE PLEASE, THIS IS THE LEAST I CAN EXPECT!
