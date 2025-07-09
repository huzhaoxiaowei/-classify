# main1.py
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, optimizers

# ✅ 更明确地添加 generator1.py 所在的目录到 sys.path
# 直接指定 generator1.py 所在的目录，确保 Python 能够找到它
generator_dir = "/root/autodl-tmp/DL-Project/版本1"
if generator_dir not in sys.path:
    sys.path.append(generator_dir)

from generator1 import IMU_Pressure_Generator  # 现在应该能找到了


# ✅ 收集图像路径
def collect_image_paths(base_path):
    imu_paths, pressure_paths, labels = [], [], []

    for imu_dir, p_dir, label in [
        (os.path.join(base_path, 'healthy_images', 'imu'), os.path.join(base_path, 'healthy_images', 'pressure'), 0),
        (os.path.join(base_path, 'parkinson_images', 'imu'), os.path.join(base_path, 'parkinson_images', 'pressure'), 1)
    ]:
        if not os.path.exists(imu_dir) or not os.path.exists(p_dir):
            print(f"❌ 路径不存在：{imu_dir} 或 {p_dir}")
            continue

        for fname in sorted(os.listdir(imu_dir)):
            if fname.endswith('.npy'):
                idx = fname.split('_')[1]
                pressure_file = f'pressure_{idx}'
                imu_path = os.path.join(imu_dir, fname)
                pressure_path = os.path.join(p_dir, fname.replace("imu_", "pressure_"))
                if os.path.exists(pressure_path):
                    imu_paths.append(imu_path)
                    pressure_paths.append(pressure_path)
                    labels.append(label)

    return imu_paths, pressure_paths, to_categorical(labels, num_classes=2)


# ✅ 构建模型
def cbam_block(input_tensor, ratio=8):
    channel = input_tensor.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
    max_pool = layers.GlobalMaxPooling2D()(input_tensor)
    shared_dense = layers.Dense(channel // ratio, activation='relu')
    avg_out = shared_dense(avg_pool)
    max_out = shared_dense(max_pool)
    channel_att = layers.Dense(channel, activation='sigmoid')(layers.Add()([avg_out, max_out]))
    channel_att = layers.Reshape((1, 1, channel))(channel_att)
    x = layers.Multiply()([input_tensor, channel_att])
    spatial_att = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(x)
    x = layers.Multiply()([x, spatial_att])
    return x


def res_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x


def build_model(imu_input_shape, pressure_input_shape):
    imu_input = layers.Input(shape=imu_input_shape, name='imu_input')
    p_input = layers.Input(shape=pressure_input_shape, name='pressure_input')

    # IMU 分支 - 滤波器数量保持 16
    x1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(imu_input)
    x1 = layers.BatchNormalization()(x1)
    x1 = cbam_block(x1)
    x1 = res_block(x1, 16)
    x1 = layers.MaxPooling2D()(x1)
    x1 = layers.Flatten()(x1)

    # 压力分支 - 滤波器数量保持 16
    x2 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(p_input)
    x2 = layers.BatchNormalization()(x2)
    x2 = cbam_block(x2)
    x2 = res_block(x2, 16)
    x2 = layers.MaxPooling2D()(x2)
    x2 = layers.Flatten()(x2)

    merged = layers.concatenate([x1, x2])
    merged = layers.Dense(64, activation='relu')(merged)
    merged = layers.Dropout(0.5)(merged)
    output = layers.Dense(2, activation='softmax')(merged)

    model = models.Model(inputs=[imu_input, p_input], outputs=output)
    model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ✅ 主程序入口
if __name__ == '__main__':
    # 路径更新为您的服务器路径
    base_path = r"/root/autodl-tmp/DL-Project/版本1"

    # 先统计总样本数量（这里不区分 train/val，只用于打印确认）
    all_imu_paths = []
    for label_name in ['healthy_images', 'parkinson_images']:
        imu_dir = os.path.join(base_path, label_name, 'imu')
        if not os.path.exists(imu_dir):
            print(f"❌ 路径不存在: {imu_dir}")
            continue
        all_imu_paths.extend([f for f in os.listdir(imu_dir) if f.endswith('.npy')])
    print(f"✅ 成功读取图像样本数: {len(all_imu_paths)}")

    # 构建样本列表 (imu_path, pressure_path, label)
    all_samples = []
    first_imu_img_shape = None
    first_pressure_img_shape = None

    for label_name in ['healthy_images', 'parkinson_images']:
        label = 0 if 'healthy' in label_name else 1
        imu_folder = os.path.join(base_path, label_name, 'imu')
        pressure_folder = os.path.join(base_path, label_name, 'pressure')

        if not os.path.exists(imu_folder) or not os.path.exists(pressure_folder):
            print(f"警告: 目录 {imu_folder} 或 {pressure_folder} 不存在，跳过该类别。")
            continue

        for imu_file in sorted(os.listdir(imu_folder)):
            if imu_file.endswith('.npy'):
                idx = imu_file.split('_')[1]
                pressure_file = f'pressure_{idx}'
                imu_path = os.path.join(imu_folder, imu_file)
                pressure_path = os.path.join(pressure_folder, pressure_file)
                if os.path.exists(pressure_path):
                    all_samples.append((imu_path, pressure_path, label))

                    # 自动检测图像通道数
                    if first_imu_img_shape is None:
                        first_imu_img_shape = np.load(imu_path).shape
                        print(f"检测到 IMU 图像形状: {first_imu_img_shape}")
                    if first_pressure_img_shape is None:
                        first_pressure_img_shape = np.load(pressure_path).shape
                        print(f"检测到压力图像形状: {first_pressure_img_shape}")

    if not all_samples:
        print("错误: 未找到任何样本。请检查图像生成路径和文件是否存在。")
        exit()

    if first_imu_img_shape is None or first_pressure_img_shape is None:
        print("错误: 未能检测到图像形状。请确保有图像文件可供加载。")
        exit()

    # 提取通道数
    imu_channels = first_imu_img_shape[-1]
    pressure_channels = first_pressure_img_shape[-1]

    # 定义传递给 build_model 的输入形状
    imu_model_input_shape = (first_imu_img_shape[0], first_imu_img_shape[1], imu_channels)
    pressure_model_input_shape = (first_pressure_img_shape[0], first_pressure_img_shape[1], pressure_channels)

    # 确保宽度和高度是 224
    if imu_model_input_shape[0] != 224 or imu_model_input_shape[1] != 224:
        print(
            f"警告: IMU 图像尺寸 ({imu_model_input_shape[0]}x{imu_model_input_shape[1]}) 与模型期望的 224x224 不符。请检查 preprocess1.py。")
    if pressure_model_input_shape[0] != 224 or pressure_model_input_shape[1] != 224:
        print(
            f"警告: 压力图像尺寸 ({pressure_model_input_shape[0]}x{pressure_model_input_shape[1]}) 与模型期望的 224x224 不符。请检查 preprocess1.py。")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_samples), 1):
        print(f"\n--- Fold {fold} ---")


        # 训练集和验证集文件路径分别写入子目录
        def save_subset(samples, folder):
            import shutil
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)

            label_names = set('healthy_images' if label == 0 else 'parkinson_images' for _, _, label in samples)
            for label_name in label_names:
                os.makedirs(os.path.join(folder, label_name, 'imu'), exist_ok=True)
                os.makedirs(os.path.join(folder, label_name, 'pressure'), exist_ok=True)

            for imu_path, pressure_path, label in samples:
                label_name = 'healthy_images' if label == 0 else 'parkinson_images'
                imu_dst = os.path.join(folder, label_name, 'imu', os.path.basename(imu_path))
                p_dst = os.path.join(folder, label_name, 'pressure', os.path.basename(pressure_path))
                shutil.copy2(imu_path, imu_dst)
                shutil.copy2(pressure_path, p_dst)


        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]
        train_dir = os.path.join(base_path, f'_fold{fold}_train')
        val_dir = os.path.join(base_path, f'_fold{fold}_val')
        save_subset(train_samples, train_dir)
        save_subset(val_samples, val_dir)

        # 创建生成器 - batch_size 恢复为 32，因为显存充足
        train_gen = IMU_Pressure_Generator(train_dir, batch_size=32, shuffle=True)
        val_gen = IMU_Pressure_Generator(val_dir, batch_size=32, shuffle=False)

        # 模型构建与训练
        model = build_model(imu_model_input_shape, pressure_model_input_shape)

        model.fit(train_gen, epochs=20, validation_data=val_gen, verbose=2)

        # 模型评估
        y_true, y_score = [], []
        for x, y_batch in val_gen:
            preds = model.predict(x, verbose=0)
            y_true.extend(np.argmax(y_batch, axis=1))
            y_score.extend(preds[:, 1])
        auc = roc_auc_score(y_true, y_score)
        y_pred = [1 if s > 0.5 else 0 for s in y_score]
        cm = confusion_matrix(y_true, y_pred)
        fn, tp = cm[1, 0], cm[1, 1]
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"Fold {fold}: AUC={auc:.3f}, FNR={fnr:.3f}")

        # 绘制 ROC 曲线
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC Curve - Fold {fold}')
        plt.legend()
        plt.grid(True)
        plt.show()