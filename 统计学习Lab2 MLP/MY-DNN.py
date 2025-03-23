import numpy as np
from copy import deepcopy
import pandas as pd
from scipy.stats import alpha

from dataset import N_CLASSES, get_train_dataset, get_test_dataset
import os

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """
        计算PCA的相关量，包括特征向量（主成分）和特征值（解释方差）
        """
        # Step 1: 数据标准化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: 使用SVD进行奇异值分解
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Step 3: 计算解释方差比例
        total_variance = np.sum(S**2)
        explained_variance = (S**2) / total_variance
        self.explained_variance_ratio_ = explained_variance

        # Step 4: 选择主成分
        self.components_ = Vt.T

        if self.n_components is None:
            cumulative_variance = np.cumsum(explained_variance)
            self.n_components = np.where(cumulative_variance >= 0.85)[0][0] + 1
            print(f"选择降维到 {self.n_components} 维，以保留 85% 的方差。")

    def transform(self, X):
        """
        将数据投影到已选择的主成分上
        """
        X_centered = X - self.mean_
        return X_centered.dot(self.components_[:, :self.n_components])

    def fit_transform(self, X):
        """
        拟合PCA并返回降维后的数据
        """
        self.fit(X)
        return self.transform(X)

class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for i in range(self.n_splits):
            start = current
            stop = current + fold_sizes[i]
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop

def sigmoid(z):
    r = np.zeros_like(z)
    pos_mask = (z >= 0)
    r[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
    neg_mask = (z < 0)
    e = np.exp(z[neg_mask])
    r[neg_mask] = e / (e + 1)
    return r


def softmax(z):
    m = np.max(z, axis=-1, keepdims=True)
    z = np.exp(z - m)
    return z / np.sum(z, axis=-1, keepdims=True)


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


class DNNClassifier:
    def __init__(self, shape):
        assert len(shape) > 1
        self.__shape = shape
        self.__weight = {}
        self.__grad = {}
        for i in range(1, len(shape)):
            self.__weight['w' + str(i)] = np.random.randn(shape[i - 1], shape[i])
            self.__weight['b' + str(i)] = np.zeros(shape[i])

    def __forward(self, x):
        cache_a = [x]
        for i in range(1, len(self.__shape)):
            z = np.matmul(cache_a[-1], self.__weight['w' + str(i)]) + self.__weight['b' + str(i)]
            cache_a.append(sigmoid(z) if i != len(self.__shape) - 1 else softmax(z))
        return cache_a

    def __call__(self, x):
        return self.__forward(x)[-1]

    def __backward(self, cache_a, y):
        batch_size = cache_a[-1].shape[0]
        dz = (cache_a[-1] - y) / self.__shape[-1]
        for i in reversed(range(1, len(self.__shape))):
            a = cache_a[i - 1]
            self.__grad['w' + str(i)] = np.matmul(a.T, dz) / batch_size
            self.__grad['b' + str(i)] = np.mean(dz, axis=0)
            w = self.__weight['w' + str(i)]
            dz = np.matmul(dz, w.T) * a * (1 - a)

    def train(self, train_x, train_y, val_x, val_y, batch_size, n_epochs, patience, optimizer):
        optimize = optimizer(self.__grad, self.__weight)
        n_train, n_val = train_x.shape[0], val_x.shape[0]
        y_true = np.argmax(val_y, axis=-1)
        idx = np.arange(n_train)
        epoch, best_epoch, best_correct = [0] * 3
        best_weight = None
        while epoch < n_epochs:
            epoch += 1
            np.random.shuffle(idx)
            loss, correct = [0] * 2
            for k in range(0, n_train, batch_size):
                batch_idx = idx[k:min(k + batch_size, n_train)]
                cache_a = self.__forward(train_x[batch_idx])
                batch_y = train_y[batch_idx]
                self.__backward(cache_a, batch_y)
                optimize()
                loss += np.sum(-batch_y * np.log(cache_a[-1]))
                correct += np.sum(np.argmax(cache_a[-1], axis=-1) == np.argmax(batch_y, axis=-1))
            y_pred = np.argmax(self(val_x), axis=-1)
            val_correct = np.sum(y_true == y_pred)
            print(f'epoch: {epoch}, train loss: {loss:.6f}, '
                  f'train accuracy: {correct / n_train:.6f}, val accuracy: {val_correct / n_val:.6f}')
            if val_correct > best_correct:
                best_correct = val_correct
                best_epoch = epoch
                best_weight = deepcopy(self.__weight)
            if epoch - best_epoch >= patience:
                print(f'early stopping: restoring weights from epoch {best_epoch}')
                self.__weight = best_weight
                break
        # Calculate error and alpha (weight)
        y_pred = np.argmax(self(val_x), axis=-1)
        err = np.sum(y_pred != y_true) / len(y_true)
        alpha = np.log((1 - err) / (err + 1e-8)) + np.log(N_CLASSES - 1)
        return self, err, alpha

    def save_parameters(self, filename, err=None, alpha=None):
        np.savez(filename, **self.__weight, err=err, alpha=alpha)
        print(f"Model parameters and additional info saved to {filename}")

    def load_parameters(self, filename):
        if os.path.exists(filename):
            # 加载 .npz 文件
            params = np.load(filename)
            for key in self.__weight:
                if key in params:
                    self.__weight[key] = params[key]

            err = params.get('err', None)
            alpha = params.get('alpha', None)
            print(f"Parameters loaded from {filename}")
            print(f"Error (err): {err}, Alpha: {alpha}")
            return err, alpha
        else:
            print(f"File {filename} not found!")
            return None, None


def save_predictions_to_csv(image_ids, predictions, output_file='predictions.csv'):
    df = pd.DataFrame({
        'ID': image_ids,
        'Predicted': predictions
    })
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


def majority_vote(predictions):
    return np.array([np.bincount(pred).argmax() for pred in predictions.T])


def predict_multiple_models(models, test_x):
    predictions = np.array([np.argmax(model(test_x), axis=-1) for model in models])
    return predictions


def predict_with_adaboost(models, test_x):
    """
    使用AdaBoost的加权投票方法进行多个模型的预测。
    每个模型产生一个权重的预测结果，最终根据权重计算加权得分。
    """
    n_samples = test_x.shape[0]
    final_scores = np.zeros((n_samples, N_CLASSES))  # 存储每个模型的加权得分
    for model, _, alpha in models:
        model_predictions = model(test_x)
        for i, prediction in enumerate(model_predictions):
            prediction = np.argmax(prediction)  # 确保是类别索引
            final_scores[i, prediction] += alpha  # 加权得分
    # 对加权得分进行投票，选择得分最高的类别
    final_predictions = np.argmax(final_scores, axis=-1)
    return final_predictions

def load_model(model_idx):
    model = DNNClassifier(model_structure)  # 使用相同的模型结构
    err, alpha = model.load_parameters(f"model_{model_idx}.npz")  # 加载模型参数和附加信息
    return model, err, alpha  # 返回模型、误差和alpha


if __name__ == '__main__':
    np.random.seed(1)

    train_images, train_labels = get_train_dataset()
    test_images, test_labels = get_test_dataset()

    print(f"Training data shape: {train_images.shape}, Labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_images.shape}, Labels shape: {test_labels.shape}")

    test_ids = np.arange(len(test_images))
    y_true = np.argmax(test_labels, axis=-1)


    models = []
    pca = PCA()
    pca.fit(train_images)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.where(cumulative_variance >= 0.85)[0][0] + 1
    print(f"选择降维到 {n_components} 维，以保留 85% 的方差。")
    pca = PCA(n_components=n_components)

    train_x_pca = pca.fit_transform(train_images)
    test_x_pca = pca.transform(test_images)

    num_models = 30
    optimizer = RMSProp(2e-3)
    batch_size = 100
    n_epochs = 1000
    patience = 20
    model_idx = 0

    kf = KFold(n_splits=num_models, shuffle=True, random_state=42)
    #训练过程
    # for train_idx, val_idx in kf.split(train_images):
    #     train_x, val_x = train_x_pca[train_idx], train_x_pca[val_idx]
    #     train_y, val_y = train_labels[train_idx], train_labels[val_idx]
    #
    #     model_structure = [train_x.shape[1], 100, N_CLASSES]
    #     model, err, alpha = DNNClassifier(model_structure).train(train_x, train_y, val_x, val_y,
    #                                                              batch_size=batch_size, n_epochs=n_epochs, patience=patience,
    #                                                              optimizer=optimizer)
    #     model.save_parameters(f"model_{model_idx}.npz", err, alpha)
    #     models.append((model, err, alpha))
    #     model_idx += 1
    #
    # final_predictions = predict_with_adaboost(models, test_x_pca)
    #直接使用训练好的参数
    models_loaded = []
    for train_idx, val_idx in kf.split(train_images):
        train_x, val_x = train_x_pca[train_idx], train_x_pca[val_idx]
        train_y, val_y = train_labels[train_idx], train_labels[val_idx]
        model_structure = [train_x.shape[1], 100, N_CLASSES]

        model, err, alpha = load_model(model_idx)  # 加载模型并返回 err 和 alpha
        models_loaded.append((model, err, alpha))  # 将模型、err 和 alpha 存储为元组
        model_idx += 1

    # 使用加权投票方法进行预测
    final_predictions = predict_with_adaboost(models_loaded, test_x_pca)


    save_predictions_to_csv(test_ids, final_predictions, output_file='predictions.csv')

