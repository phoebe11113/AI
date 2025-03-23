import numpy as np
import random
import time
import matplotlib.pyplot as plt


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

# Visualization
def show_data(data):
    fig, ax = plt.subplots()
    cls = data[:, 2]
    ax.scatter(data[:, 0][cls==1], data[:, 1][cls==1])
    ax.scatter(data[:, 0][cls==-1], data[:, 1][cls==-1])
    ax.grid(False)
    fig.tight_layout()
    plt.show()

# Standardize the data (mean=0, std=1)
def standardize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


# Kernel function (linear, rbf)
def calc_kernel_value(matrix_x, sample_x, kernel_option):
    kernel_type = kernel_option[0]
    num_samples = matrix_x.shape[0]
    kernel_value = np.zeros((num_samples, 1))

    if kernel_type == 'linear':
        kernel_value = np.dot(matrix_x, sample_x.T)
    elif kernel_type == 'rbf':
        sigma = kernel_option[1]
        for i in range(num_samples):
            diff = matrix_x[i, :] - sample_x
            kernel_value[i] = np.exp(-np.sum(diff ** 2) / (2.0 * sigma ** 2))
    return kernel_value


# Calculate kernel matrix
def calc_kernel_matrix(train_x, kernel_option):
    num_samples = train_x.shape[0]
    kernel_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        kernel_matrix[:, i] = calc_kernel_value(train_x, train_x[i, :], kernel_option).flatten()
    return kernel_matrix


# Calculate error for a given alpha_k
def calc_error(svm, alpha_k):
    output_k = np.dot(np.multiply(svm.alphas, svm.train_y).T, svm.kernel_mat[:, alpha_k]) + svm.b
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k


# Update error cache
def update_error(svm, alpha_k):
    error = calc_error(svm, alpha_k)
    svm.error_cache[alpha_k] = [1, error]


# Select alpha_j (heuristic method for selecting the second alpha)
def select_alpha_j(svm, alpha_i, error_i):
    svm.error_cache[alpha_i] = [1, error_i]
    candidate_alpha_list = np.nonzero(svm.error_cache[:, 0])[0]
    max_step = 0
    alpha_j = 0
    error_j = 0

    if len(candidate_alpha_list) > 1:
        for alpha_k in candidate_alpha_list:
            if alpha_k == alpha_i:
                continue
            error_k = calc_error(svm, alpha_k)
            if abs(error_k - error_i) > max_step:
                max_step = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
    else:
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(random.uniform(0, svm.num_samples))
        error_j = calc_error(svm, alpha_j)

    return alpha_j, error_j


# Main optimization loop (SMO)
def inner_loop(svm, alpha_i):

    error_i = calc_error(svm, alpha_i)
    if isinstance(error_i, np.ndarray):
        error_i = error_i.item()  # Convert to scalar if it is an ndarray

    train_y_i = float(svm.train_y[alpha_i])

    # Check the KKT conditions
    if (train_y_i * error_i < -svm.toler and svm.alphas[alpha_i] < svm.C) or \
            (train_y_i * error_i > svm.toler and svm.alphas[alpha_i] > 0):
        alpha_j, error_j = select_alpha_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        # Calculate L and H
        if train_y_i != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])

        if L == H:
            return 0

        # Calculate eta
        eta = 2.0 * svm.kernel_mat[alpha_i, alpha_j] - svm.kernel_mat[alpha_i, alpha_i] - svm.kernel_mat[
            alpha_j, alpha_j]
        if eta >= 0:
            return 0

        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta
        svm.alphas[alpha_j] = np.clip(svm.alphas[alpha_j], L, H)

        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            update_error(svm, alpha_j)
            return 0

        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] * (alpha_j_old - svm.alphas[alpha_j])

        # Update b
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) * svm.kernel_mat[
            alpha_i, alpha_i] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) * svm.kernel_mat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) * svm.kernel_mat[
            alpha_i, alpha_j] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) * svm.kernel_mat[alpha_j, alpha_j]
        if 0 < svm.alphas[alpha_i] < svm.C:
            svm.b = b1
        elif 0 < svm.alphas[alpha_j] < svm.C:
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        update_error(svm, alpha_j)
        update_error(svm, alpha_i)

        return 1
    else:
        return 0


def cross_validate(svm, X, y, K=5):
    num_samples = X.shape[0]  # 样本数
    fold_size = num_samples // K  # 每折的样本数量
    accuracies = []  # 用于保存每一折的准确率

    # 打乱数据集的顺序
    indices = np.random.permutation(num_samples)

    for fold in range(K):
        # 划分出当前折的验证集（validation）索引
        validation_indices = indices[fold * fold_size: (fold + 1) * fold_size]

        # 得到当前折的训练集（train）索引
        train_indices = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])

        # 根据训练集和验证集索引，获取训练数据和标签
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[validation_indices], y[validation_indices]

        svm.train(X_train, y_train)

        accuracy = svm.predict(X_val, y_val)

        # 将每一折的准确率记录下来
        accuracies.append(accuracy)

    return np.mean(accuracies)

# SVM class
class SVM:
    def __init__(self, data_set, labels, C=1.5, toler=0.001, kernel_option=('rbf', 0.3)):
        self.train_x = np.array(data_set)
        self.train_y = np.array(labels)
        self.C = C
        self.toler = toler
        self.num_samples = data_set.shape[0]
        self.alphas = np.zeros(self.num_samples)  # Use 1D array instead
        self.b = 0
        self.error_cache = np.zeros((self.num_samples, 2))
        self.kernel_option = kernel_option
        self.kernel_mat = calc_kernel_matrix(self.train_x, self.kernel_option)

    def train(self, train_x, train_y, max_iter=50):
        start_time = time.time()
        entire_set = True
        alpha_pairs_changed = 0
        iter_count = 0
        while iter_count < max_iter and (alpha_pairs_changed > 0 or entire_set):
            alpha_pairs_changed = 0
            if entire_set:
                for i in range(self.num_samples):
                    alpha_pairs_changed += inner_loop(self, i)
                iter_count += 1
            else:
                non_bound_alphas_list = np.nonzero((self.alphas > 0) & (self.alphas < self.C))[0]
                for i in non_bound_alphas_list:
                    alpha_pairs_changed += inner_loop(self, i)
                iter_count += 1

            if entire_set:
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True

        print(f'Congratulations, training complete! Took {time.time() - start_time:.2f} seconds!')
        return self

    def predict(self, test_x, test_y):
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        num_test_samples = test_x.shape[0]
        support_vectors_index = np.nonzero(self.alphas > 0)[0]
        support_vectors = self.train_x[support_vectors_index]
        support_vector_labels = self.train_y[support_vectors_index]
        support_vector_alphas = self.alphas[support_vectors_index]

        match_count = 0
        for i in range(num_test_samples):
            kernel_value = calc_kernel_value(support_vectors, test_x[i, :], self.kernel_option)
            predict = np.dot(kernel_value.T, np.multiply(support_vector_labels, support_vector_alphas)) + self.b
            if np.sign(predict) == np.sign(test_y[i]):  # 正确使用 test_y
                match_count += 1

        accuracy = float(match_count) / num_test_samples
        return accuracy

    def plot(self):
        fig, ax = plt.subplots()

        # Define the boundaries of the plot
        x_min, x_max = self.train_x[:, 0].min() - 1, self.train_x[:, 0].max() + 1
        y_min, y_max = self.train_x[:, 1].min() - 1, self.train_x[:, 1].max() + 1

        # Generate grid points
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        # Use the prediction function to evaluate the model on the grid points
        Z = np.array([self.predict(np.array([[xi, yi]]), self.train_y) for xi, yi in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary with a custom color map (e.g., 'coolwarm')
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

        # Plot the training data points
        scatter = ax.scatter(self.train_x[:, 0], self.train_x[:, 1], c=self.train_y, s=50, edgecolor='k', marker='o',
                             cmap=plt.cm.RdYlBu)  # Change cmap here for different data colors

        # Highlight support vectors with a different color and style
        support_vectors_index = np.nonzero(self.alphas > 0)[0]
        ax.scatter(self.train_x[support_vectors_index, 0], self.train_x[support_vectors_index, 1],
                   facecolors='none', edgecolors='black', s=100,
                   label="Support Vectors")  # Different color for support vectors


        # Show the legend for support vectors
        ax.legend()

        # Display the plot
        plt.show()


## 主函数中修改
if __name__ == '__main__':
    # Load dataset
    train_file = 'data/train_kernel.txt'
    test_file = 'data/test_kernel.txt'
    data_train = load_data(train_file)
    data_test = load_data(test_file)

    show_data(data_train)
    show_data(data_test)

    X_train = data_train[:, :2]
    y_train = data_train[:, 2]
    X_test = data_test[:, :2]
    y_test = data_test[:, 2]

    # Standardize the training data
    X_train = standardize_data(X_train)
    X_test = standardize_data(X_test)

    best_C = 1
    best_sigma = 1.0
    best_accuracy = 0
    #多次实验找到较好参数有C:1.5,2,2.5,3
    #sigma:0;3,0.6,0.9
    for C in [1.5,2,2.5,3]:
        for sigma in [0.3,0.6,0.9]:
            svm = SVM(X_train, y_train, C=C, kernel_option=('rbf', sigma))
            accuracy = cross_validate(svm, X_train, y_train)
            if accuracy > best_accuracy:
                best_C = C
                best_sigma = sigma
                best_accuracy = accuracy

    print(f"Best C: {best_C}, Best sigma: {best_sigma}, Best accuracy: {best_accuracy}")

    svm = SVM(X_train, y_train,C=best_C, kernel_option=('rbf', best_sigma))
    # svm = SVM(X_train, y_train)
    svm.train(X_train, y_train, max_iter=50)

    # Test predictions
    accuracy_train = svm.predict(X_train, y_train)  # 传入 y_train
    accuracy_test = svm.predict(X_test, y_test)  # 传入 y_test

    # Evaluate accuracy
    print(f'Train accuracy: {accuracy_train * 100:.2f}%')
    print(f'Test accuracy: {accuracy_test * 100:.2f}%')

    # Plot decision boundary
    svm.plot()