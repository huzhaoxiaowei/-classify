# preprocess1.py
import os
import numpy as np
from pyts.image import GramianAngularField, RecurrencePlot

# 参数设置
window_size = 224 # 保持为 224
step = 224 # 保持为 224

# 输入文件
files = {
    'healthy': {'imu': r'\DL-Project\实验数据\假设数据\处理后的数据\2Z_imu_data.npy', 'pressure': r'\DL-Project\实验数据\假设数据\处理后的数据\2Z_pressure_data.npy'},
    'parkinson': {'imu': r'\DL-Project\实验数据\假设数据\处理后的数据\3P_imu_data.npy', 'pressure': r'\DL-Project\实验数据\假设数据\处理后的数据\3P_pressure_data.npy'}
}

# 创建输出目录
for label in files:
    os.makedirs(f'{label}_images/imu', exist_ok=True)
    os.makedirs(f'{label}_images/pressure', exist_ok=True)

# 滑动窗口函数
def sliding_windows(data, size, step):
    """对时间序列 data 以窗口 size 和步长 step 划分子序列。"""
    for start in range(0, len(data) - size + 1, step):
        yield data[start:start+size]

# 初始化转换器
# ✅ 保持 GramianAngularField 的 image_size 参数
gaf = GramianAngularField(image_size=window_size, method='summation')
# ❌ 移除 RecurrencePlot 的 image_size 参数
rp = RecurrencePlot()

# 对每个类别的数据进行转换
for label, paths in files.items():
    # 读取 IMU 和压力数据 (假设形状为 [Timestamps, Channels])
    imu_data = np.load(paths['imu'])
    pressure_data = np.load(paths['pressure'])

    count = 1
    for imu_seg, p_seg in zip(sliding_windows(imu_data, window_size, step),
                              sliding_windows(pressure_data, window_size, step)):
        # IMU 数据：对每个通道分别做 GAF，再堆叠为多通道图像
        imu_imgs = []
        for ch in range(imu_seg.shape[1]):
            ts = imu_seg[:, ch].reshape(1, -1)
            img = gaf.fit_transform(ts)[0]
            imu_imgs.append(img)
        imu_img = np.stack(imu_imgs, axis=-1)

        # 压力数据：对每个通道分别做 Recurrence Plot，再堆叠
        p_imgs = []
        for ch in range(p_seg.shape[1]):
            ts = p_seg[:, ch].reshape(1, -1)
            img = rp.fit_transform(ts)[0]
            p_imgs.append(img)
        p_img = np.stack(p_imgs, axis=-1)

        # 保存图像为 .npy
        imu_path = f"{label}_images/imu/imu_{count:04d}.npy"
        p_path = f"{label}_images/pressure/pressure_{count:04d}.npy"
        np.save(imu_path, imu_img)
        np.save(p_path, p_img)
        count += 1