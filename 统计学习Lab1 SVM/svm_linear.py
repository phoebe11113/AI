import numpy as np
import matplotlib.pyplot as plt
import random

# Load dataset
def load_data(fname):
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)

# Split data into training and testing sets
def split_data(data, train_ratio=0.7):
    indices = list(range(len(data)))
    random.shuffle(indices)
    train_size = int(len(data) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return data[train_indices], data[test_indices]

# Calculate classification accuracy
def eval_acc(label, pred):
    return np.sum(label == pred) / len(pred)


# Visualization
def show_data(data):
    fig, ax = plt.subplots()
    cls = data[:, 2]
    ax.scatter(data[:, 0][cls==1], data[:, 1][cls==1])
    ax.scatter(data[:, 0][cls==-1], data[:, 1][cls==-1])
    ax.grid(False)
    fig.tight_layout()
    plt.show()


def plot_decision_boundary(data, svm):
    fig, ax = plt.subplots()
    cls = data[:, 2]

    # 绘制数据点
    ax.scatter(data[:, 0][cls == 1], data[:, 1][cls == 1], c='r', label='Class 1')
    ax.scatter(data[:, 0][cls == -1], data[:, 1][cls == -1], c='b', label='Class -1')

    # 获取权重和偏置
    w = svm.w
    b = svm.b

    # 计算超平面的x坐标范围
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1

    # 计算对应的y坐标（即超平面上的点）
    y_vals = (-w[0] * x_min - b) / w[1]
    x_vals = np.linspace(x_min, x_max, 100)
    y_boundary = (-w[0] * x_vals - b) / w[1]

    # 绘制超平面
    ax.plot(x_vals, y_boundary, 'k--', label='Decision Boundary')

    ax.grid(False)
    ax.legend()
    fig.tight_layout()
    plt.show()

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate  # 学习率
        self.lambda_param = lambda_param  # 正则化参数（C 的倒数）
        self.n_iters = n_iters  # 迭代次数
        self.w = None  # 权重向量
        self.b = None  # 偏置量

    """
           训练SVM模型。

           参数:
           X : ndarray, shape (m, n)
               训练数据，其中 m 是样本数量，n 是特征数量。
           y : ndarray, shape (m,)
               训练标签，必须是 +1 或 -1。
           """
    def train(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # 确保标签是 +1 或 -1

        # 初始化权重和偏置
        self.w = np.zeros(n_features)
        self.b = 0

        # 梯度下降
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    """
            使用训练好的模型进行预测。

            参数:
            X : ndarray, shape (m, n)
                测试数据，其中 m 是样本数量，n 是特征数量。

            返回:
            predictions : ndarray, shape (m,)
                预测标签。
            """
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


if __name__ == '__main__':
    # Load dataset
    train_file = 'data/train_linear.txt'
    # train_file = 'data/train_linear_intersect.txt'
    # train_file = 'data/train_kernel.txt'
    # test_file = 'data/test_linear.txt'
    data = load_data(train_file)  # dataset format [x1, x2, t], shape (N * 3)
    # data_test = load_data(test_file)
    data_train, data_test = split_data(data, train_ratio=0.7)

    show_data(data)
    show_data(data_train)
    show_data(data_test)

    # train SVM
    svm = SVM()
    X_train = data_train[:, :2]  # 取所有行，除了最后一列（标签）
    y_train = data_train[:, 2]  # 取所有行的最后一列（标签）
    svm.train(X_train, y_train)

    plot_decision_boundary(data_train, svm)
    plot_decision_boundary(data_test, svm)

    # predict
    x_train = data_train[:, :2]  # features [x1, x2]
    t_train = data_train[:, 2]  # ground truth labels
    t_train_pred = svm.predict(x_train)  # predicted labels
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # evaluate
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
