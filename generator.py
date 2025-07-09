import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from scipy.interpolate import interp1d


class IMU_Pressure_Generator(Sequence):
    def __init__(self, imu_data, pressure_data, labels, imu_transformer, pressure_transformer,
                 batch_size=32, target_size=224, shuffle=True, augment=False):
        self.imu_data = imu_data
        self.pressure_data = pressure_data
        self.labels = labels
        self.imu_transformer = imu_transformer
        self.pressure_transformer = pressure_transformer
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.imu_data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.imu_data) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        imu_batch = self.imu_data[batch_indices]
        pressure_batch = self.pressure_data[batch_indices]
        labels_batch = self.labels[batch_indices]

        imu_images = self.preprocess_imu_batch(imu_batch)
        pressure_images = self.preprocess_pressure_batch(pressure_batch)

        imu_images = tf.cast(imu_images, tf.float32)
        pressure_images = tf.cast(pressure_images, tf.float32)

        if self.augment:
            imu_images = self.apply_augmentation(imu_images)
            pressure_images = self.apply_augmentation(pressure_images)

        return {'imu_input': imu_images, 'pressure_input': pressure_images}, labels_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def interpolate_batch(self, batch_data):
        """对批次数据进行插值，使其达到 target_size 时间步长。
        batch_data 形状可以是 (batch_size, original_timesteps, num_features) (3D)
        或 (batch_size, original_timesteps, N_sensors, N_axes) (4D)。
        """
        interpolated_batch = []
        for i in range(batch_data.shape[0]):  # 遍历批次中的每个样本
            original_timesteps = batch_data.shape[1]

            # 获取特征部分的形状，例如 (num_features,) 或 (N_sensors, N_axes)
            feature_shape = batch_data.shape[2:]
            num_total_features_flat = np.prod(feature_shape)  # 计算扁平化后的总特征数

            if original_timesteps != self.target_size:
                # 将当前样本的数据展平为 (original_timesteps, num_total_features_flat)
                flat_data_slice = batch_data[i].reshape(original_timesteps, num_total_features_flat)

                interp_data_flat = np.zeros((self.target_size, num_total_features_flat))
                for j in range(num_total_features_flat):  # 对每个扁平化的特征进行插值
                    original_time = np.linspace(0, 1, original_timesteps)
                    target_time = np.linspace(0, 1, self.target_size)
                    interp_func = interp1d(original_time, flat_data_slice[:, j], kind='linear')
                    interp_data_flat[:, j] = interp_func(target_time)

                # 插值完成后，恢复原始特征形状
                interpolated_batch.append(interp_data_flat.reshape(self.target_size, *feature_shape))
            else:
                interpolated_batch.append(batch_data[i])
        return np.array(interpolated_batch)

    def preprocess_imu_batch(self, imu_batch):
        """
        预处理IMU批次数据，将其转换为GAF图像。
        imu_batch 的形状应为 (batch_size, window_size, num_sensors, num_axes)
        例如 (32, 224, 4, 3)
        """
        imu_batch_interp = self.interpolate_batch(imu_batch)

        batch_imu_images = []
        for single_imu_segment in imu_batch_interp:  # 遍历批次中的每个样本
            # single_imu_segment 的形状是 (window_size, num_sensors, num_axes) 例如 (224, 4, 3)
            imu_images_for_one_segment = []

            num_sensors = single_imu_segment.shape[1]
            num_axes = single_imu_segment.shape[2]

            for sensor_idx in range(num_sensors):
                for axis_idx in range(num_axes):
                    # 提取单个 1D 时间序列
                    channel_data_1d = single_imu_segment[:, sensor_idx, axis_idx]
                    # pyts 期望 (n_samples, n_timestamps)，所以这里需要 reshape 为 (1, window_size)
                    transformed = self.imu_transformer.fit_transform(channel_data_1d.reshape(1, -1))[0].astype(
                        np.float32)
                    imu_images_for_one_segment.append(transformed)
            # 堆叠为 (H, W, C) 的单一样本图像 (e.g., 224, 224, 12)
            batch_imu_images.append(np.stack(imu_images_for_one_segment, axis=-1))
        # 返回批次图像 (N, H, W, C)
        return np.array(batch_imu_images)

    def preprocess_pressure_batch(self, pressure_batch):
        """
        预处理压力批次数据，将其转换为RP图像。
        pressure_batch 的形状应为 (batch_size, window_size, num_pressure_features)
        例如 (32, 224, 2)
        """
        pressure_batch_interp = self.interpolate_batch(pressure_batch)

        batch_pressure_images = []
        for single_pressure_segment in pressure_batch_interp:
            # single_pressure_segment 的形状是 (window_size, num_pressure_features)
            pressure_images_for_one_segment = []
            for i in range(single_pressure_segment.shape[1]):  # 遍历压力数据的每个特征/通道 (2个通道)
                channel_data_1d = single_pressure_segment[:, i]
                transformed = self.pressure_transformer.fit_transform(channel_data_1d.reshape(1, -1))[0].astype(
                    np.float32)
                pressure_images_for_one_segment.append(transformed)
            batch_pressure_images.append(np.stack(pressure_images_for_one_segment, axis=-1))
        return np.array(batch_pressure_images)

    def apply_augmentation(self, images):
        """对输入图像批量增强"""
        images_aug = tf.image.random_flip_left_right(images)
        images_aug = tf.image.random_brightness(images_aug, max_delta=0.1)
        return images_aug