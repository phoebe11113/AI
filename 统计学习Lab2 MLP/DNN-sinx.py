import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def relu(z):
    return np.maximum(0, z)


def RMSProp(eta, gamma=0.9, eps=1e-6):

    def optimizer(grads, vars):
        states = {key: np.zeros_like(vars[key]) for key in vars}

        def step():
            for key in vars:
                if key in grads:
                    g = grads[key]
                    r = gamma * states[key] + (1 - gamma) * g ** 2
                    vars[key] -= eta * g / (np.sqrt(r) + eps)
                    states[key] = r

        return step

    return optimizer


class DNNRegressor:
    """
    深度神经网络回归模型
    """

    def __init__(self, shape):
        assert len(shape) > 1
        self.__shape = shape
        self.__weight = {}
        self.__grad = {}
        for i in range(1, len(shape)):
            self.__weight['w' + str(i)] = np.random.randn(shape[i - 1], shape[i]) * 0.01  # 小的初始化
            self.__weight['b' + str(i)] = np.zeros(shape[i])

    def __forward(self, x):
        cache_a = [x]
        for i in range(1, len(self.__shape)):
            z = np.matmul(cache_a[-1], self.__weight['w' + str(i)]) + self.__weight['b' + str(i)]
            # 对于中间层，使用ReLU激活函数
            if i != len(self.__shape) - 1:
                cache_a.append(relu(z))
            else:
                cache_a.append(z)  # 最后一层没有激活函数（回归任务）
        return cache_a

    def __call__(self, x):
        output = self.__forward(x)[-1]
        return output

    def __backward(self, cache_a, y):
        batch_size = cache_a[-1].shape[0]
        dz = (cache_a[-1] - y) / batch_size  # 损失函数的梯度
        for i in reversed(range(1, len(self.__shape))):
            a = cache_a[i - 1]
            self.__grad['w' + str(i)] = np.matmul(a.T, dz)
            self.__grad['b' + str(i)] = np.mean(dz, axis=0)
            w = self.__weight['w' + str(i)]
            dz = np.matmul(dz, w.T) * (a > 0)

    def train(self, train_x, train_y, val_x, val_y, batch_size, n_epochs, patience, optimizer):
        optimize = optimizer(self.__grad, self.__weight)
        n_train, n_val = train_x.shape[0], val_x.shape[0]
        epoch, best_epoch, best_error = 0, 0, float('inf')
        best_weight = None
        while epoch < n_epochs:
            epoch += 1
            idx = np.arange(n_train)
            np.random.shuffle(idx)
            loss = 0
            for k in range(0, n_train, batch_size):
                batch_idx = idx[k:min(k + batch_size, n_train)]
                cache_a = self.__forward(train_x[batch_idx])
                batch_y = train_y[batch_idx]
                self.__backward(cache_a, batch_y)
                optimize()
                loss += np.sum((cache_a[-1] - batch_y) ** 2)
            y_pred = self(val_x)
            val_loss = np.mean((y_pred - val_y) ** 2)
            print(f'Epoch {epoch}, Train loss: {loss:.6f}, Validation loss: {val_loss:.6f}')

            if val_loss < best_error:
                best_error = val_loss
                best_epoch = epoch
                best_weight = deepcopy(self.__weight)

            if epoch - best_epoch >= patience:
                print(f"Early stopping: restoring weights from epoch {best_epoch}")
                self.__weight = best_weight
                break
        return self


def generate_sine_data(n_samples=1000, x_min=-np.pi, x_max=np.pi):
    """
    生成数据集，目标是拟合y = sin(x)
    """
    x = np.random.uniform(x_min, x_max, n_samples).reshape(-1, 1)
    y = np.sin(x)
    return x, y


if __name__ == '__main__':
    np.random.seed(1)

    # 生成训练集、验证集和测试集
    train_x, train_y = generate_sine_data(1000)
    val_x, val_y = generate_sine_data(200)
    test_x, test_y = generate_sine_data(200)

    # 定义并训练模型
    model = DNNRegressor([1, 512, 1]) \
        .train(train_x, train_y, val_x, val_y,
               batch_size=100, n_epochs=1000, patience=20,
               optimizer=RMSProp(eta=1e-3))

    # 在测试集上进行预测
    y_pred = model(test_x)

    # 输出回归误差（均方误差）（偷懒直接用了sklearn里面的均方误差计算函数）
    test_loss = mean_squared_error(test_y, y_pred)
    print(f"Test MSE: {test_loss:.6f}")

    # 绘制真实值与预测值的对比图
    plt.scatter(test_x, test_y,  s=5, label="True data points")  # 绘制真实数据点
    plt.scatter(test_x, y_pred,  s=5, label="Predicted data points")  # 绘制预测数据点
    plt.legend()
    plt.show()
