import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical # 添加 to_categorical
from pyts.image import GramianAngularField
# 注意：我们不再需要在生成器内部进行插值，所以可以移除 interp1d

class AdvancedDataGenerator(Sequence):
    """
    一个高级数据生成器，它在内部处理滑动窗口、动态图像生成和缩放。
    """
    def __init__(self, raw_data, raw_labels, raw_patient_ids,
                 window_size, step,
                 target_image_size=128,
                 feature_index=0,
                 batch_size=64,
                 shuffle=True,
                 is_rgb=True):
        """
        初始化生成器。
        :param raw_data: 完整的原始特征数据 (N, num_features)
        :param raw_labels: 完整的原始标签
        :param raw_patient_ids: 完整的原始病人ID
        :param window_size: 滑动窗口的大小 (例如 896)
        :param step: 滑动窗口的步长
        :param target_image_size: 模型最终输入的图像尺寸 (例如 224)
        :param feature_index: 要用于生成图像的特征列的索引
        :param batch_size: 批次大小
        :param shuffle: 是否在每个epoch后打乱数据
        :param is_rgb: 是否将灰度图转换为3通道RGB图
        """
        self.batch_size = batch_size
        self.target_image_size = target_image_size
        self.is_rgb = is_rgb
        self.shuffle = shuffle
        self.feature_index = feature_index

        # 初始化GAF转换器，用于生成高分辨率图像
        self.transformer = GramianAngularField(image_size=window_size, method='summation')

        # 核心步骤1：在初始化时，根据原始数据创建所有的时间窗口
        print("Generator: Creating time windows...")
        self.windows, self.labels, self.patient_ids = self._create_windows(
            raw_data, raw_labels, raw_patient_ids, window_size, step
        )
        print(f"Generator: Created {len(self.windows)} windows.")

        self.indices = np.arange(len(self.windows))
        self.on_epoch_end()

    def _create_windows(self, data, labels, patient_ids, window_size, step):
        """内部函数：使用滑动窗口创建数据切片。"""
        windows, window_labels, window_patient_ids = [], [], []
        unique_patients = np.unique(patient_ids)

        for patient in unique_patients:
            patient_mask = (patient_ids == patient)
            patient_data = data[patient_mask]
            patient_labels = labels[patient_mask]

            if len(patient_data) < window_size:
                continue

            for i in range(0, len(patient_data) - window_size + 1, step):
                window = patient_data[i : i + window_size]
                label = patient_labels[i + window_size - 1]
                windows.append(window)
                window_labels.append(label)
                window_patient_ids.append(patient)

        return np.array(windows), to_categorical(np.array(window_labels), num_classes=2), np.array(window_patient_ids)

    def __len__(self):
        """返回每个epoch的批次数。"""
        return int(np.floor(len(self.windows) / self.batch_size))

    def __getitem__(self, index):
        """生成一个批次的数据。"""
        # 1. 获取当前批次的索引
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # 2. 根据索引获取对应的窗口数据
        windows_batch = self.windows[batch_indices]
        labels_batch = self.labels[batch_indices]

        # 3. 对这一批次的窗口数据执行动态转换
        images_batch = self._transform_batch_to_images(windows_batch)

        return images_batch, labels_batch

    def on_epoch_end(self):
        """在每个epoch结束后，如果需要，则打乱索引。"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _transform_batch_to_images(self, windows_batch):
        """
        核心步骤2：将一批窗口数据动态转换为模型所需的图像格式。
        """
        # a. 从窗口中提取指定的单一特征序列
        # windows_batch shape: (batch_size, window_size, num_features)
        feature_sequence_batch = windows_batch[:, :, self.feature_index]
        # shape: (batch_size, window_size)

        # b. GAF变换，生成高分辨率图像
        # shape: (batch_size, window_size, window_size)
        gaf_high_res = self.transformer.fit_transform(feature_sequence_batch)

        # c. 添加通道维度，以符合图像处理函数的要求
        # shape: (batch_size, window_size, window_size, 1)
        gaf_high_res_with_channel = gaf_high_res[..., np.newaxis]

        # d. 缩放图像到目标尺寸
        # shape: (batch_size, target_size, target_size, 1)
        images_resized = tf.image.resize(gaf_high_res_with_channel, [self.target_image_size, self.target_image_size])

        # e. (可选) 转换为3通道RGB图像
        if self.is_rgb:
            images_final = tf.image.grayscale_to_rgb(images_resized)
        else:
            images_final = images_resized

        return images_final