import numpy as np
import sys
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, Activation, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from pyts.image import GramianAngularField, RecurrencePlot
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, roc_curve
from scipy.interpolate import interp1d
from sklearn.utils import shuffle, class_weight
from sklearn.preprocessing import StandardScaler
import re  # 导入正则表达式模块

# 更明确地添加 generator.py 所在的目录到 sys.path
generator_dir = "/root/autodl-tmp/DL-Project"  # 请确认这个路径是 generator.py 实际所在的目录
if generator_dir not in sys.path:
    sys.path.append(generator_dir)

from generator import IMU_Pressure_Generator

# ------------------ 数据加载 ------------------
# 定义数据根目录
data_root_dir = r'/root/autodl-tmp/DL-Project/实验数据/假设数据/处理后的数据/'


# ✅ 动态生成 patient_id_to_label 映射
def generate_patient_id_map(base_dir):
    """
    遍历目录，根据文件名模式自动生成病人ID到标签的映射。
    假定文件名格式为 'xZ_imu_data.npy' 或 'xP_imu_data.npy'。
    'Z' -> 0 (健康), 'P' -> 1 (帕金森)。
    """
    patient_id_map = {}

    # 查找所有 .npy 文件
    all_npy_files = [f for f in os.listdir(base_dir) if f.endswith('.npy')]

    # 使用正则表达式匹配文件名，提取病人ID和类型
    # 模式：数字 + Z/P + _imu_data.npy 或 _pressure_data.npy
    pattern = re.compile(r'(\d+[ZP])_(imu_data|pressure_data)\.npy')

    for fname in all_npy_files:
        match = pattern.match(fname)
        if match:
            patient_id = match.group(1)  # 'xZ' 或 'xP'

            if patient_id not in patient_id_map:  # 只添加一次
                if 'Z' in patient_id:
                    patient_id_map[patient_id] = 0  # 健康
                elif 'P' in patient_id:
                    patient_id_map[patient_id] = 1  # 帕金森

    # 按照病人ID进行排序，可选，但有助于查看
    sorted_patient_ids = sorted(patient_id_map.keys(), key=lambda x: (x[-1], int(x[:-1])))
    sorted_map = {pid: patient_id_map[pid] for pid in sorted_patient_ids}

    print("自动生成的病人ID映射:")
    for pid, label in sorted_map.items():
        print(f"  '{pid}': {label},")

    return sorted_map


# 自动生成病人ID到标签的映射
patient_id_to_label = generate_patient_id_map(data_root_dir)


def load_data_from_folders(base_dir, patient_id_map):
    """
    加载指定目录下所有 IMU 和 Pressure 文件的数据，并根据病人ID分配标签。
    base_dir: 数据文件所在的根目录。
    patient_id_map: 字典，映射病人ID（文件名前缀）到类别标签。
    """
    all_imu_segments_list = []
    all_pressure_segments_list = []
    all_labels_list = []

    # 收集所有 .npy 文件
    all_npy_files = [f for f in os.listdir(base_dir) if f.endswith('.npy')]

    # 对文件名进行分类和匹配
    imu_files_dict = {f.split('_')[0]: f for f in all_npy_files if '_imu_data.npy' in f}
    pressure_files_dict = {f.split('_')[0]: f for f in all_npy_files if '_pressure_data.npy' in f}

    # 遍历已知的病人ID映射
    for patient_id, label in patient_id_map.items():
        if patient_id in imu_files_dict and patient_id in pressure_files_dict:
            imu_fname = imu_files_dict[patient_id]
            pressure_fname = pressure_files_dict[patient_id]

            imu_path = os.path.join(base_dir, imu_fname)
            pressure_path = os.path.join(base_dir, pressure_fname)

            if os.path.exists(imu_path) and os.path.exists(pressure_path):
                imu_data = np.load(imu_path)
                pressure_data = np.load(pressure_path)

                all_imu_segments_list.append(imu_data)
                all_pressure_segments_list.append(pressure_data)
                all_labels_list.append(label)
            else:
                print(f"警告: 病人ID '{patient_id}' 的数据文件缺失 ({imu_path} 或 {pressure_path})。")
        else:
            print(f"警告: 病人ID '{patient_id}' 的 IMU 或 Pressure 文件未在目录中找到。")

    if not all_imu_segments_list:
        raise ValueError(f"在目录 {base_dir} 中未找到任何匹配的病人数据文件。请检查路径、文件命名和 patient_id_to_label 映射！")

    return all_imu_segments_list, all_pressure_segments_list, np.array(all_labels_list)


# 加载数据 (现在返回的是每个病人的完整时间序列数据列表)
imu_data_per_patient, pressure_data_per_patient, labels_per_patient = load_data_from_folders(data_root_dir,
                                                                                             patient_id_to_label)

print(f"总共加载了 {len(imu_data_per_patient)} 个病人样本。")
print(f"健康病人数量: {np.sum(labels_per_patient == 0)}")
print(f"帕金森病人数量: {np.sum(labels_per_patient == 1)}")

# --------------------------------------------------------------------------------------
# ⚠️ 重要的修改：
# 在 K-Fold 之前，您现在需要对 *每个病人的数据* 进行滑动窗口分段。
# IMU_Pressure_Generator 应该接收这些分段后的数据。
# --------------------------------------------------------------------------------------

# 定义滑动窗口参数 (与 preprocess.py 中的参数保持一致)
window_size = 224  # 从 preprocess.py 导入或手动设置
step = 224  # 从 preprocess.py 导入或手动设置


def sliding_windows(data, size, step):
    """对时间序列 data 以窗口 size 和步长 step 划分子序列。"""
    segments = []
    for start in range(0, len(data) - size + 1, step):
        segments.append(data[start:start + size])
    return np.array(segments)


all_imu_segments_combined = []
all_pressure_segments_combined = []
all_segment_labels_combined = []  # 每个窗口的标签

for i in range(len(imu_data_per_patient)):
    # 对每个病人的 IMU 数据进行分段
    imu_segments = sliding_windows(imu_data_per_patient[i], window_size, step)
    # 对每个病人的 Pressure 数据进行分段
    pressure_segments = sliding_windows(pressure_data_per_patient[i], window_size, step)

    # 每个分段的标签与病人本身的标签相同
    segment_labels = np.full(len(imu_segments), labels_per_patient[i])

    all_imu_segments_combined.append(imu_segments)
    all_pressure_segments_combined.append(pressure_segments)
    all_segment_labels_combined.append(segment_labels)

# 将所有病人的分段合并到一个大的 NumPy 数组中
all_imu_segments_combined = np.concatenate(all_imu_segments_combined, axis=0)
all_pressure_segments_combined = np.concatenate(all_pressure_segments_combined, axis=0)
all_segment_labels_combined = np.concatenate(all_segment_labels_combined, axis=0)

# 对所有分段进行洗牌 (重要！在K折交叉验证之前)
# 确保IMU分段、压力分段和分段标签以相同的方式洗牌
all_imu_segments_shuffled, all_pressure_segments_shuffled, all_segment_labels_shuffled = \
    shuffle(all_imu_segments_combined, all_pressure_segments_combined, all_segment_labels_combined, random_state=42)

# 将标签转换为one-hot编码
all_labels_one_hot = to_categorical(all_segment_labels_shuffled, num_classes=2)

print(f"总IMU分段数量: {all_imu_segments_shuffled.shape[0]}")
print(f"总压力分段数量: {all_pressure_segments_shuffled.shape[0]}")
print(f"总分段标签形状: {all_labels_one_hot.shape}")
print(f"健康分段数量: {np.sum(all_segment_labels_shuffled == 0)}")
print(f"帕金森分段数量: {np.sum(all_segment_labels_shuffled == 1)}")

# 标准化 (通常在数据加载后，图像转换前进行)
# 对于多通道时间序列数据，通常对所有时间点上的每个通道独立进行标准化
scaler_imu = StandardScaler()
# all_imu_segments_shuffled.shape: (num_segments, window_size, num_sensors, num_axes)
original_imu_segment_shape = all_imu_segments_shuffled.shape
# 展平为 (num_segments * window_size, num_sensors * num_axes) 进行标准化
reshaped_imu_data = all_imu_segments_shuffled.reshape(-1,
                                                      original_imu_segment_shape[-2] * original_imu_segment_shape[-1])
all_imu_segments_scaled = scaler_imu.fit_transform(reshaped_imu_data).reshape(original_imu_segment_shape)

scaler_pressure = StandardScaler()
# all_pressure_segments_shuffled.shape: (num_segments, window_size, num_pressure_features)
original_pressure_segment_shape = all_pressure_segments_shuffled.shape
all_pressure_segments_scaled = scaler_pressure.fit_transform(
    all_pressure_segments_shuffled.reshape(-1, original_pressure_segment_shape[-1])).reshape(
    original_pressure_segment_shape)

# GAF 和 RP 转换器初始化
# 这些转换器会被传递给生成器，由生成器在运行时进行图像转换
imu_transformer = GramianAngularField(image_size=window_size, method='summation')
pressure_transformer = RecurrencePlot(threshold='point', percentage=20)


# ------------------ 模型定义 (保持不变) ------------------
# ... (build_model 函数和你的ResNet模型定义保持不变)
def residual_block(x, filters, kernel_size, stride=1, conv_shortcut=False, name=None):
    bn_axis = -1  # 假设通道在最后一个维度

    # 主分支
    if conv_shortcut:
        shortcut = Conv2D(filters * 4, 1, strides=stride, name=name + '_0_conv')(x)  # ResNet 通常会在这里将通道数乘以4
        shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    # 1x1 卷积 (Bottleneck)
    x = Conv2D(filters, 1, strides=stride, use_bias=False, name=name + '_1_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    # 3x3 卷积
    x = Conv2D(filters, kernel_size, padding='same', use_bias=False, name=name + '_2_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    # 1x1 卷积 (Bottleneck 恢复)
    x = Conv2D(filters * 4, 1, use_bias=False, name=name + '_3_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = Add(name=name + '_add')([shortcut, x])
    x = Activation('relu', name=name + '_out')(x)
    return x


def build_model(imu_input_shape, pressure_input_shape):
    # IMU 分支 (ResNet)
    imu_input = Input(shape=imu_input_shape, name='imu_input')  # (224, 224, 12)
    x_imu = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False, name='imu_conv1')(imu_input)
    x_imu = BatchNormalization(name='imu_bn1')(x_imu)
    x_imu = Activation('relu', name='imu_relu1')(x_imu)
    x_imu = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='imu_pool1')(x_imu)

    # ResNet Blocks for IMU
    # filters=64 -> 输出 64 * 4 = 256 通道
    x_imu = residual_block(x_imu, 64, (3, 3), conv_shortcut=True, name='imu_res_block1_1')
    x_imu = residual_block(x_imu, 64, (3, 3), name='imu_res_block1_2')

    # filters=128 -> 输出 128 * 4 = 512 通道
    x_imu = residual_block(x_imu, 128, (3, 3), stride=2, conv_shortcut=True, name='imu_res_block2_1')
    x_imu = residual_block(x_imu, 128, (3, 3), name='imu_res_block2_2')

    x_imu = GlobalAveragePooling2D(name='imu_avg_pool')(x_imu)
    x_imu = Dense(256, activation='relu', name='imu_dense')(x_imu)
    x_imu = Dropout(0.5, name='imu_dropout')(x_imu)

    # 足底压力分支 (ResNet)
    pressure_input = Input(shape=pressure_input_shape, name='pressure_input')  # (224, 224, 2)
    x_pressure = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False, name='pressure_conv1')(
        pressure_input)
    x_pressure = BatchNormalization(name='pressure_bn1')(x_pressure)
    x_pressure = Activation('relu', name='pressure_relu1')(x_pressure)
    x_pressure = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pressure_pool1')(x_pressure)

    # ResNet Blocks for Pressure
    x_pressure = residual_block(x_pressure, 64, (3, 3), conv_shortcut=True, name='pressure_res_block1_1')
    x_pressure = residual_block(x_pressure, 64, (3, 3), name='pressure_res_block1_2')

    x_pressure = residual_block(x_pressure, 128, (3, 3), stride=2, conv_shortcut=True, name='pressure_res_block2_1')
    x_pressure = residual_block(x_pressure, 128, (3, 3), name='pressure_res_block2_2')

    x_pressure = GlobalAveragePooling2D(name='pressure_avg_pool')(x_pressure)
    x_pressure = Dense(256, activation='relu', name='pressure_dense')(x_pressure)
    x_pressure = Dropout(0.5, name='pressure_dropout')(x_pressure)

    # 合并两个分支
    combined = Concatenate()([x_imu, x_pressure])

    # 最终分类层
    output = Dense(2, activation='softmax', name='output_layer')(combined)

    model = Model(inputs=[imu_input, pressure_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])  # 添加auc指标
    return model


# ------------------ K折交叉验证 ------------------
num_folds = 5  # 保持 5 折，或者根据数据量增加到 10 折
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

fold_results = []

# ✅ 清理旧的临时目录 (此处不再需要，因为不再生成临时图片文件了)
# 如果您之前有生成临时图片文件的代码，并希望清理，请手动添加
# for i in range(num_folds):
#     temp_train_dir = os.path.join(base_for_temp_dirs, f'_fold{i}_train')
#     temp_val_dir = os.path.join(base_for_temp_dirs, f'_fold{i}_val')
#     if os.path.exists(temp_train_dir):
#         import shutil
#         shutil.rmtree(temp_train_dir)
#     if os.path.exists(temp_val_dir):
#         import shutil
#         shutil.rmtree(temp_val_dir)


# 确保模型输入形状与 generator 输出的图像形状一致
# 假设 imu_data 形状是 (num_segments, window_size, num_sensors, num_axes)
# 那么 imu_model_input_shape 是 (window_size, window_size, num_sensors * num_axes)
# 注意：这里从实际数据推断形状，更加稳健
# imu_segments_scaled 形状 (num_segments, window_size, num_sensors, num_axes)
imu_model_input_shape = (window_size, window_size, all_imu_segments_scaled.shape[2] * all_imu_segments_scaled.shape[3])
# pressure_segments_scaled 形状 (num_segments, window_size, num_pressure_features)
pressure_model_input_shape = (window_size, window_size, all_pressure_segments_scaled.shape[2])

print(f"模型IMU输入形状: {imu_model_input_shape}")
print(f"模型压力输入形状: {pressure_model_input_shape}")

# ✅ K折交叉验证循环现在直接在分段后的数据上操作
for fold, (train_idx, val_idx) in enumerate(
        kf.split(all_imu_segments_shuffled, all_segment_labels_shuffled)):  # ✅ 使用分段后的数据和标签进行划分
    print(f"\n--- Fold {fold + 1}/{num_folds} ---")

    # 划分训练集和验证集数据
    imu_train, imu_val = all_imu_segments_scaled[train_idx], all_imu_segments_scaled[val_idx]
    pressure_train, pressure_val = all_pressure_segments_scaled[train_idx], all_pressure_segments_scaled[val_idx]
    labels_train, labels_val = all_labels_one_hot[train_idx], all_labels_one_hot[val_idx]

    # ✅ 计算并应用类别权重
    y_integers_train = np.argmax(labels_train, axis=1)  # 从 one-hot 编码转换回整数标签
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_integers_train),
        y=y_integers_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Computed class weights for this fold: {class_weight_dict}")

    # 创建数据生成器 (直接传入 NumPy 数组，而不是目录路径)
    train_generator = IMU_Pressure_Generator(
        imu_train, pressure_train, labels_train,
        imu_transformer=imu_transformer,
        pressure_transformer=pressure_transformer,
        batch_size=32, shuffle=True, augment=True
    )
    val_generator = IMU_Pressure_Generator(
        imu_val, pressure_val, labels_val,
        imu_transformer=imu_transformer,
        pressure_transformer=pressure_transformer,
        batch_size=32, shuffle=False, augment=False
    )

    # 构建和编译模型
    model = build_model(imu_model_input_shape, pressure_model_input_shape)

    # 定义回调函数
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    # 模型训练
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks,
        verbose=2,
        class_weight=class_weight_dict  # ✅ 添加类别权重
    )

    # 验证集评估
    y_true_val = []
    y_pred_scores_val = []
    for x_batch, y_batch in val_generator:
        preds = model.predict(x_batch, verbose=0)
        y_true_val.extend(np.argmax(y_batch, axis=1))
        y_pred_scores_val.extend(preds[:, 1])

    y_true_val = np.array(y_true_val)
    y_pred_scores_val = np.array(y_pred_scores_val)
    y_pred_classes_val = (y_pred_scores_val > 0.5).astype(int)

    # 计算指标
    auc = roc_auc_score(y_true_val, y_pred_scores_val)
    cm = confusion_matrix(y_true_val, y_pred_classes_val)

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0

    print(f"Fold {fold + 1} - AUC: {auc:.4f}")
    print(f"Fold {fold + 1} - FNR: {fnr:.4f}")
    print(f"Fold {fold + 1} - TPR (Recall): {tpr:.4f}")
    print(f"Fold {fold + 1} - Confusion Matrix:\n{cm}")

    fold_results.append({'fold': fold + 1, 'auc': auc, 'fnr': fnr, 'confusion_matrix': cm})

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Parkinson'])
    disp.plot(cmap='Blues')
    plt.title(f"Fold {fold + 1} Confusion Matrix FNR: {fnr:.2f}")
    plt.savefig(f'confusion_matrix_fold_{fold + 1}.png')
    plt.close()

    fpr, tpr_curve, _ = roc_curve(y_true_val, y_pred_scores_val)
    plt.plot(fpr, tpr_curve, label=f"Fold {fold + 1} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC=0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold + 1}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'roc_curve_fold_{fold + 1}.png')
    plt.close()

# 汇总所有折叠的结果
print("\n--- 所有折叠结果 ---")
for res in fold_results:
    print(f"Fold {res['fold']}: AUC={res['auc']:.4f}, FNR={res['fnr']:.4f}")

avg_auc = np.mean([res['auc'] for res in fold_results])
avg_fnr = np.mean([res['fnr'] for res in fold_results])
print(f"\n平均 AUC: {avg_auc:.4f}")
print(f"平均 FNR: {avg_fnr:.4f}")