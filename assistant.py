
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
'''
读取数据
训练测试保存模型
预测输出
'''
def readData(file):
    X = list()
    y = list()
    with open(file, 'r') as f:
        for line in f.readlines():
            li = line.split(' ')
            X.append([int(li[0]), int(li[1])])
            y.append([float(li[2]), float(li[3])])
        # print(X,y)
        return X, y
    pass

def train(x,y):
    model = PolynomialFeatures(degree=2)
    x = model.fit_transform(x)
    x_test= model.transform(x)
    model = LinearRegression()
    model.fit(x, y)
    joblib.dump(model, 'fuck.m')
    print('多项式回归正确率', model.score(x_test, y))

def prepare():
    return joblib.load('fuck.m')

if __name__ == '__main__':
    data='data.txt'
    x,y=readData(data)
    train(x,y)