import os
import numpy as np
import pandas as pd
from PIL import Image

# 定义文件夹路径
train_folder = './train_2024'
test_folder = './test_2022'
label_file = './upload_sample.csv'

# 常量
SIZE = 28 * 28
N_CLASSES = 10


def load_images_from_folder(folder):
    """加载文件夹中的所有图片，并返回图像数据和标签"""
    images = []
    labels = []
    for label in range(N_CLASSES):
        folder_path = os.path.join(folder, str(label))  # 获取每个类的文件夹路径
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".bmp"):
                    image_path = os.path.join(folder_path, filename)
                    try:
                        img = Image.open(image_path).convert('L')
                        img = img.resize((28, 28))
                        img_array = np.array(img)
                        images.append(img_array)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error loading {image_path}: {e}")

    # 转换为numpy数组
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def load_test_images(test_folder, label_file):
    """加载测试图片和对应标签"""
    images = []
    # 读取CSV文件中的标签
    labels_df = pd.read_csv(label_file)

    # 获取图像ID和标签
    image_ids = labels_df['ID'].values
    labels = labels_df['GT'].values

    # 遍历测试图片文件夹
    for image_id, label in zip(image_ids, labels):
        # 修改为bmp格式
        image_filename = f"{image_id}.bmp"  # 假设文件名为ID.bmp
        image_path = os.path.join(test_folder, image_filename)
        try:
            # 使用PIL库打开图像并转换为灰度图
            img = Image.open(image_path).convert('L')
            img = img.resize((28, 28))  # 确保图像大小为28x28
            img_array = np.array(img)
            images.append(img_array)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")

    # 转换为numpy数组
    images = np.array(images)
    return images, labels


def preprocess_data(images, labels):
    """预处理数据：标准化并转换为适合神经网络输入的格式"""
    # 将图像数据归一化至[0, 1]
    images = images / 255.0

    # 重塑图像为二维向量（28x28 -> 784维）
    images = images.reshape(-1, SIZE)

    # 将标签转为one-hot编码
    labels_one_hot = np.eye(N_CLASSES)[labels]

    return images, labels_one_hot


def get_train_dataset():
    """加载训练数据"""
    # 读取训练数据
    train_images, train_labels = load_images_from_folder(train_folder)

    # 预处理训练数据
    train_images, train_labels = preprocess_data(train_images, train_labels)

    return train_images, train_labels


def get_test_dataset():
    """加载测试数据"""
    # 读取测试数据和标签
    test_images, test_labels = load_test_images(test_folder, label_file)

    # 预处理测试数据
    test_images, test_labels = preprocess_data(test_images, test_labels)

    return test_images, test_labels


