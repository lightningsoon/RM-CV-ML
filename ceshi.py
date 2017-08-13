import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

X_train = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y_train = [[7,6], [9,9], [13,10], [17.5,18.3], [18,19.3]]
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11,13], [8.5,9], [15,16], [18,19.9], [11,12]]
# 建立线性回归，并用训练的模型绘图
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# 多项式回归
quadratic_featurizer = PolynomialFeatures(degree=2)
quadratic_featurizer.fit(X_train,y_train)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)
# 多项式回归器的数据进行运算
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)

print(X_train)
print(X_train_quadratic)
print(X_test)
print(X_test_quadratic)
print('1 r-squared', regressor.score(X_test, y_test))
print('2 r-squared', regressor_quadratic.score(X_test_quadratic, y_test))