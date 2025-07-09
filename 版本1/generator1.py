# generator1.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, Sequence


class IMU_Pressure_Generator(Sequence):
    """数据生成器：同时加载IMU图像和压力图像，并输出标签。"""

    def __init__(self, img_dir, batch_size=32, shuffle=True):
        """
        img_dir: 包含 'healthy_images' 和 'parkinson_images' 子目录的总目录路径。
        batch_size: 批大小。
        shuffle: 是否在每轮开始时打乱样本。
        """
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples = []  # 存放 (imu_path, pressure_path, label) 的列表

        # 遍历目录加载样本路径
        # ⚠️【已修改】这里的文件夹名现在与 main1.py 中创建的文件夹名一致
        for label_name in ['healthy_images', 'parkinson_images']:
            # ⚠️【已修改】标签判断逻辑更稳健
            label = 0 if 'healthy' in label_name else 1
            imu_folder = os.path.join(img_dir, label_name, 'imu')
            pressure_folder = os.path.join(img_dir, label_name, 'pressure')

            # 增加路径存在性检查，避免因空文件夹报错
            if not os.path.exists(imu_folder):
                print(f"警告：路径不存在 {imu_folder}")
                continue

            imu_files = sorted(os.listdir(imu_folder))
            # 假设 imu_0001.npy, pressure_0001.npy 对应相同索引
            for imu_file in imu_files:
                idx = imu_file.split('_')[1]  # 如 '0001.npy'
                pressure_file = f'pressure_{idx}'
                imu_path = os.path.join(imu_folder, imu_file)
                pressure_path = os.path.join(pressure_folder, pressure_file)
                if os.path.exists(pressure_path):
                    self.samples.append((imu_path, pressure_path, label))

        self.on_epoch_end()

    def __len__(self):
        """每个 epoch 的批次数"""
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        """每轮结束或开始时随机打乱索引"""
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        """生成一个批次的数据"""
        batch_inds = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        imu_batch = []
        pressure_batch = []
        labels = []
        for i in batch_inds:
            imu_path, p_path, label = self.samples[i]
            imu_img = np.load(imu_path)  # 形状 (H, W, C)
            p_img = np.load(p_path)  # 形状 (H, W, C)
            imu_batch.append(imu_img)
            pressure_batch.append(p_img)
            labels.append(label)
        # 转换为 NumPy 数组并归一化 (可选)
        imu_batch = np.array(imu_batch, dtype=np.float32) / 255.0
        pressure_batch = np.array(pressure_batch, dtype=np.float32) / 255.0
        y_batch = to_categorical(labels, num_classes=2)
        return [imu_batch, pressure_batch], y_batch