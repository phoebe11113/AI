import numpy as np
import matplotlib.pyplot as plt
import random
from cvxopt import matrix, solvers

## Load dataset
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

def plot_decision_boundary(data, svm, is_train=True):
    fig, ax = plt.subplots()
    cls = data[:, 2]

    ax.scatter(data[:, 0][cls == 1], data[:, 1][cls == 1], c='r', label='Class 1')
    ax.scatter(data[:, 0][cls == -1], data[:, 1][cls == -1], c='b', label='Class -1')

    w = svm.w
    b = svm.b

    # 绘制支持向量
    # if is_train:
    #     support_vectors = svm.support_vectors
    #     ax.scatter(data[support_vectors, 0], data[support_vectors, 1], facecolors='none', edgecolors='k', s=100, label='Support Vectors')

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = (-w[0] * x_vals - b) / w[1]

    ax.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

    ax.grid(False)
    ax.legend()
    fig.tight_layout()
    plt.show()


# SVM 类
class SVM:
    def __init__(self, C=1):
        self.C = C  # 正则化参数C
        self.w = None
        self.b = None
        self.support_vectors = None

    def train(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # 确保标签是 +1 或 -1

        # 计算核矩阵
        K = np.dot(X, X.T)  # 计算内积矩阵

        # 构建二次规划的矩阵
        P = matrix(np.outer(y_, y_) * K)  # QP中的P矩阵
        q = matrix(-np.ones(n_samples))  # QP中的q矩阵
        G = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]))  # G矩阵
        h = matrix(np.hstack([np.zeros(n_samples), np.ones(n_samples) * self.C]))  # h矩阵

        # 求解二次规划问题
        sol = solvers.qp(P, q, G, h)

        # 获取alpha值
        alphas = np.ravel(sol['x'])

        # 计算权重 w 和偏置 b
        self.w = np.dot((alphas * y_), X)  # 权重
        support_vector_indices = np.where(alphas > 1e-5)[0]  # 支持向量
        self.b = np.mean(y_[support_vector_indices] - np.dot(X[support_vector_indices], self.w))  # 偏置

        # 保存支持向量的索引
        self.support_vectors = support_vector_indices

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

    # 交叉验证 + 网格搜索
    def grid_search(self, X, y, C_values, n_folds=5):
        best_C = None
        best_acc = 0

        fold_size = len(X) // n_folds
        indices = np.arange(len(X))

        for C in C_values:
            accuracies = []
            # 将数据分成 n_folds 个子集进行交叉验证
            for i in range(n_folds):
                val_indices = indices[i * fold_size: (i + 1) * fold_size]
                train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

                X_train, X_val = X[train_indices], X[val_indices]
                y_train, y_val = y[train_indices], y[val_indices]

                # 训练 SVM
                self.C = C
                self.train(X_train, y_train)

                # 在验证集上进行预测
                y_pred = self.predict(X_val)
                acc = np.mean(y_pred == y_val)  # 计算验证集的准确率
                accuracies.append(acc)

            # 计算C值对应的平均准确率
            mean_acc = np.mean(accuracies)
            print(f"Average accuracy for C={C}: {mean_acc * 100:.2f}%")

            # 如果该C值的平均准确率更高，则更新最佳C值
            if mean_acc > best_acc:
                best_C = C
                best_acc = mean_acc

        print(f"Best C value: {best_C} with accuracy: {best_acc * 100:.2f}%")
        self.C = best_C  # 使用最佳C值
        return self  # 直接返回当前SVM实例


if __name__ == '__main__':
    # Load dataset
    # train_file = 'data/train_linear.txt'
    train_file = 'data/train_linear_intersect.txt'
    data = load_data(train_file)  # dataset format [x1, x2, t], shape (N * 3)
    data_train, data_test = split_data(data, train_ratio=0.7)
    show_data(data)
    # 获取训练集和测试集的特征和标签
    X_train = data_train[:, :2]  # 取所有行，除了最后一列（标签）
    y_train = data_train[:, 2]  # 取所有行的最后一列（标签）
    X_test = data_test[:, :2]
    y_test = data_test[:, 2]

    # 初始化SVM
    svm = SVM(C=1)  # 初始C值为1

    # 使用网格搜索调优C值
    best_svm = svm.grid_search(X_train, y_train, C_values=[0.003,0.007,0.009,0.01,0.05,0.1], n_folds=5)

    # 使用最佳模型进行预测和评估
    y_train_pred = best_svm.predict(X_train)
    y_test_pred = best_svm.predict(X_test)

    # 计算并输出训练和测试准确率
    acc_train = eval_acc(y_train, y_train_pred)
    acc_test = eval_acc(y_test, y_test_pred)
    print("Train accuracy: {:.1f}%".format(acc_train * 100))
    print("Test accuracy: {:.1f}%".format(acc_test * 100))

    # Plot decision boundary and support vectors
    plot_decision_boundary(data_train, best_svm, is_train=True)
    plot_decision_boundary(data_test, best_svm, is_train=False)
