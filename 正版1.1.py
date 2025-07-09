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
generator_dir = "/root/autodl-tmp"  # 请确认这个路径是 generator.py 实际所在的目录
if generator_dir not in sys.path:
    sys.path.append(generator_dir)

# 尝试导入 generator，如果失败则打印提示信息
try:
    from generator import IMU_Pressure_Generator
except ImportError:
    print(f"⚠️ 无法导入 generator.py 中的 IMU_Pressure_Generator。请确保 '{generator_dir}' 路径正确，且 generator.py 存在。")
    # 如果 generator 不是核心部分，可以考虑在此处退出或提供默认行为

# ------------------ 数据加载 ------------------
# 定义数据根目录
data_root_dir = r'/root/autodl-tmp/实验数据/假设数据/处理后的数据'


# ✅ 动态生成 patient_id_to_label 映射 (根据你提供的新的函数)
def generate_patient_id_map(base_dir):
    """
    遍历目录，根据文件名模式自动生成病人ID到标签的映射。
    假定文件名格式为 'xxxZ_imu_data.npy' 或 'xxxP_imu_data.npy'。
    'Z' -> 0 (健康), 'P' -> 1 (帕金森)。
    例如: '1Z_imu_data.npy', '12P_pressure_data.npy'
    """
    patient_id_map = {}
    print(f"正在检查数据目录: {base_dir}")
    if not os.path.exists(base_dir):
        print(f"错误: 数据目录 '{base_dir}' 不存在。请检查路径。")
        return patient_id_map

    # 查找所有 .npy 文件
    all_npy_files = [f for f in os.listdir(base_dir) if f.endswith('.npy')]

    # 使用正则表达式匹配文件名，提取病人ID和类型
    # 模式：数字 + Z/P + _imu_data.npy 或 _pressure_data.npy
    # 捕获组1是患者ID，捕获组2是数据类型（imu_data或pressure_data）
    pattern = re.compile(r'(\d+[ZP])_(imu_data|pressure_data)\.npy')

    for fname in all_npy_files:
        match = pattern.match(fname)
        if match:
            patient_id_str = match.group(1)  # 例如 '1Z' 或 '12P'
            data_type = match.group(2) # 'imu_data' 或 'pressure_data'

            # 只有当找到 imu_data 文件时，才将其视为一个完整的患者ID，并为其分配标签
            # 避免重复为同一个ID分配标签
            if data_type == 'imu_data' and patient_id_str not in patient_id_map:
                if 'Z' in patient_id_str:
                    patient_id_map[patient_id_str] = 0  # 健康 (0)
                elif 'P' in patient_id_str:
                    patient_id_map[patient_id_str] = 1  # 帕金森 (1)
                else:
                    print(f"警告: 文件名 '{fname}' 中的ID '{patient_id_str}' 不包含 'Z' 或 'P'，跳过。")
            # print(f"  ✅ 匹配成功: 文件 '{fname}', 提取ID: {patient_id_str}, 类型: {data_type}") # 详细调试信息
        # else:
            # print(f"  ❌ 未匹配: 文件 '{fname}' 不符合模式。") # 详细调试信息

    # 按照病人ID进行排序，可选，但有助于查看
    # 假设 'Z' 的ID数字部分在前，'P'的ID数字部分在前
    # 例如 '1Z', '1P', '2Z', '2P' 排序
    # sorted_patient_ids = sorted(patient_id_map.keys(), key=lambda x: (x[-1], int(x[:-1]))) # 适用于 xZ/xP 这种ID
    # 但如果 ID 只是 1, 2, 3 且 Z/P 是标签，则不需要这么复杂
    # 在这里，patient_id_str 已经是 '1Z' 或 '1P' 这种形式了，直接排序可能需要更复杂的key
    # 简单起见，先按字符串排序，如果需要特定数字排序，再细化lambda函数
    sorted_patient_ids = sorted(patient_id_map.keys()) # 默认按字符串排序

    sorted_map = {pid: patient_id_map[pid] for pid in sorted_patient_ids}

    print("自动生成的病人ID到标签映射:")
    for pid, label in sorted_map.items():
        print(f"  '{pid}': {label},")

    return sorted_map

# 调用新的函数来生成映射
patient_id_to_label = generate_patient_id_map(data_root_dir)


# ✅ 检查 patient_id_to_label 是否为空
if not patient_id_to_label:
    print("❌ 警告：病人ID到标签映射为空。请检查 'data_root_dir' 是否包含 'xxxZ_imu_data.npy' 或 'xxxP_imu_data.npy' 文件，并确认文件名格式正确。")
    sys.exit("程序终止：未找到数据文件。")


# 修改 load_data 函数以适应新的文件名约定
def load_data(data_root_dir, patient_id_to_label):
    all_imu_data = []
    all_pressure_data = []
    all_labels = []
    all_patient_ids = []

    for patient_id_str, label in patient_id_to_label.items(): # patient_id_str 现在是 '1Z', '2P' 这样的字符串
        # 构建 IMU 和 Pressure 文件的完整路径
        imu_file = os.path.join(data_root_dir, f'{patient_id_str}_imu_data.npy')
        pressure_file = os.path.join(data_root_dir, f'{patient_id_str}_pressure_data.npy')

        if os.path.exists(imu_file) and os.path.exists(pressure_file):
            try:
                imu_data = np.load(imu_file)
                pressure_data = np.load(pressure_file)

                # 确保IMU和压力数据样本数一致
                min_samples = min(imu_data.shape[0], pressure_data.shape[0])
                imu_data = imu_data[:min_samples]
                pressure_data = pressure_data[:min_samples]

                all_imu_data.append(imu_data)
                all_pressure_data.append(pressure_data)
                all_labels.extend([label] * min_samples)
                all_patient_ids.extend([patient_id_str] * min_samples) # patient_id_str 作为ID
                print(f"  ✅ 成功加载患者 '{patient_id_str}' 的数据。样本数: {min_samples}, 标签: {label}")
            except Exception as e:
                print(f"❌ 加载或处理患者 '{patient_id_str}' 的数据时出错: {e}。文件: {imu_file}, {pressure_file}")
        else:
            print(f"警告：找不到患者 '{patient_id_str}' 的完整数据文件 ({imu_file} 或 {pressure_file} 缺失)，跳过。")

    # 检查是否成功加载了任何数据
    if not all_imu_data:
        print("❌ 错误：未能从指定目录加载任何有效的IMU或压力数据。请检查文件是否存在且格式正确。")
        sys.exit("程序终止：未加载到任何数据。")

    # np.concatenate 在所有数据都加载成功后执行
    return np.concatenate(all_imu_data, axis=0), np.concatenate(all_pressure_data, axis=0), np.array(all_labels), np.array(all_patient_ids)

# 调用修改后的 load_data
X_imu, X_pressure, y_labels, patient_ids = load_data(data_root_dir, patient_id_to_label)


scaler_imu = StandardScaler()
X_imu_scaled = scaler_imu.fit_transform(X_imu.reshape(-1, X_imu.shape[-1])).reshape(X_imu.shape)

scaler_pressure = StandardScaler()
X_pressure_scaled = scaler_pressure.fit_transform(X_pressure)

# 将IMU和压力数据合并，形成模型的输入
# 假设IMU数据是 (N, 4, 3)，压力数据是 (N, 2)
# 合并成 (N, 4, 5) 或 (N, 6, 3) 甚至 (N, 12 + 2) 都可以，取决于你的模型设计
# 这里简单地将压力数据扩展到IMU数据的最后一个维度，作为新的特征
# 假设每个时间步的IMU数据有12个特征（4*3），压力有2个
# 我们将其展平并拼接
X_imu_flat = X_imu_scaled.reshape(X_imu_scaled.shape[0], -1) # (N, 12)
X_combined = np.concatenate((X_imu_flat, X_pressure_scaled), axis=1) # (N, 14)

# 转换为图像表示（GAF）
# 选择 GAF 作为图像转换方法
image_size = 24  # 图像尺寸，可以根据数据长度调整
gaf = GramianAngularField(image_size=image_size, method='summation')

# 对合并后的数据进行GAF转换
# 需要确保 X_combined 的每个样本长度足够进行 GAF 转换 (即 >= image_size)
# 如果 X_combined 长度太短，可能需要调整 image_size 或插值
# 这里假设 X_combined.shape[1] (即14) 足够，或者我们将进行插值
if X_combined.shape[1] < image_size:
    # 如果特征数量小于目标图像尺寸，进行插值
    X_padded_combined = np.zeros((X_combined.shape[0], image_size))
    for i in range(X_combined.shape[0]):
        original_indices = np.linspace(0, X_combined.shape[1] - 1, num=X_combined.shape[1])
        new_indices = np.linspace(0, X_combined.shape[1] - 1, num=image_size)
        interp_func = interp1d(original_indices, X_combined[i], kind='linear')
        X_padded_combined[i] = interp_func(new_indices)
else:
    # 如果特征数量足够，直接使用
    X_padded_combined = X_combined

X_gaf = gaf.fit_transform(X_padded_combined)
X_gaf = X_gaf.reshape(X_gaf.shape[0], image_size, image_size, 1) # 添加通道维度

# 标签转换为独热编码
y_categorical = to_categorical(y_labels, num_classes=len(np.unique(y_labels)))

# 计算类别权重
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_labels),
    y=y_labels
)
class_weight_dict = dict(enumerate(class_weights))
print("类别权重:", class_weight_dict)

# ------------------ 模型定义 (ResNet 风格) ------------------
def resnet_block(input_tensor, filters, kernel_size=(3, 3), stage=1, block='a'):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters, kernel_size, padding='same', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)

    x = Add()([x, input_tensor]) # 残差连接
    x = Activation('relu')(x)
    return x

def build_resnet_model(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)

    x = Conv2D(32, (7, 7), strides=(2, 2), padding='same', name='conv1')(input_tensor)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x) # 暂时移除，如果图像太小可能不需要

    x = resnet_block(x, 32, stage=2, block='a')
    x = resnet_block(x, 32, stage=2, block='b')

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='conv_down1')(x) # 下采样
    x = BatchNormalization(name='bn_down1')(x)
    x = Activation('relu')(x)

    x = resnet_block(x, 64, stage=3, block='a')
    x = resnet_block(x, 64, stage=3, block='b')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x) # 添加 Dropout 防止过拟合
    output_tensor = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

# ------------------ K折交叉验证 ------------------
n_splits = 5  # 定义折数
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_results = []
all_true_labels = []
all_pred_scores = []
all_patient_ids_for_roc = [] # 用于收集ROC曲线的患者ID

for fold, (train_index, val_index) in enumerate(skf.split(X_padded_combined, y_labels)):
    print(f"\n--- 正在处理第 {fold + 1} 折 ---")

    # 打印训练集和验证集数据量
    print(f"  训练集样本数: {len(train_index)}")
    print(f"  验证集样本数: {len(val_index)}")

    X_train, X_val = X_gaf[train_index], X_gaf[val_index]
    y_train, y_val = y_categorical[train_index], y_categorical[val_index]
    y_true_val_binary = y_labels[val_index] # 用于FNR/AUC计算的二分类标签
    patient_ids_val = patient_ids[val_index] # 验证集的患者ID

    # 确保训练集和验证集中的类别分布
    print(f"  训练集类别分布: {np.bincount(np.argmax(y_train, axis=1))}")
    print(f"  验证集类别分布: {np.bincount(np.argmax(y_val, axis=1))}")

    model = build_resnet_model(input_shape=(image_size, image_size, 1), num_classes=len(np.unique(y_labels)))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 设置回调函数
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=150, # 可以根据需要调整
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stopping],
        class_weight=class_weight_dict, # 应用类别权重
        verbose=1 # 设置为0不打印每个epoch的训练过程，只打印回调信息
    )

    # 评估模型
    y_pred_scores_val = model.predict(X_val) # 获取预测概率
    y_pred_labels_val = np.argmax(y_pred_scores_val, axis=1) # 获取预测类别

    # 计算 AUC (对于二分类)
    if len(np.unique(y_true_val_binary)) > 1: # 确保验证集中至少有两个类别
        auc = roc_auc_score(y_true_val_binary, y_pred_scores_val[:, 1]) # 假设阳性类是索引1
    else:
        auc = 0.0 # 如果验证集只有一个类别，AUC无法计算
        print("警告: 验证集中只有一个类别，无法计算AUC。")

    # 计算混淆矩阵
    cm = confusion_matrix(y_true_val_binary, y_pred_labels_val)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0) # 处理混淆矩阵可能不是2x2的情况

    # 计算 FNR 和 TPR
    # FNR = FN / (FN + TP)
    # TPR (Recall) = TP / (TP + FN)
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

    fpr, tpr_curve, _ = roc_curve(y_true_val_binary, y_pred_scores_val[:, 1]) # 对于二分类
    plt.plot(fpr, tpr_curve, label=f"Fold {fold + 1} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--') # 随机分类器的对角线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_fold_{fold + 1}.png')
    plt.close()

    # 收集所有折叠的真实标签和预测分数，用于绘制总的ROC曲线
    all_true_labels.extend(y_true_val_binary)
    all_pred_scores.extend(y_pred_scores_val[:, 1]) # 只收集阳性类的分数
    all_patient_ids_for_roc.extend(patient_ids_val) # 收集患者ID

# ------------------ 结果汇总 ------------------
print("\n--- 所有折叠结果 ---")
for res in fold_results:
    print(f"Fold {res['fold']}: AUC={res['auc']:.4f}, FNR={res['fnr']:.4f}")

# 绘制所有折叠的平均ROC曲线
if len(np.unique(all_true_labels)) > 1: # 确保至少有两个类别
    overall_auc = roc_auc_score(all_true_labels, all_pred_scores)
    print(f"\n--- 整体结果 ---")
    print(f"平均AUC: {overall_auc:.4f}")

    fpr_overall, tpr_overall, _ = roc_curve(all_true_labels, all_pred_scores)
    plt.plot(fpr_overall, tpr_overall, label=f"Overall (AUC={overall_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Overall ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('overall_roc_curve.png')
    plt.close()
else:
    print("\n--- 整体结果 ---")
    print("警告: 整体数据中只有一个类别，无法计算整体AUC。")

# 可以进一步分析FNR为1.0000的原因：
# 1. 数据量极度不平衡：第五折验证集中的阳性样本（帕金森患者）数量极少。
# 2. 模型过拟合或欠拟合：模型没有学到如何识别阳性样本的特征。
# 3. 验证集划分问题： StratifiedKFold 应该能保证类别分布相对均衡，但如果总样本量太小，或者某些类别样本量极少，即使分层也可能导致某些折的验证集出现极端情况。
# 4. 数据预处理问题： 输入数据质量或处理方式可能导致模型无法学习。

# 建议：
# 1. 运行修改后的代码，检查第五折训练集和验证集的具体样本数，尤其是阳性样本数。
# 2. 检查 patient_id_to_label 映射是否正确，确保标签与实际数据对应。
# 3. 检查原始数据中是否存在数据缺失、异常值等问题，特别是在第五折对应的患者数据中。
# 4. 尝试增加数据集大小，或者采用更多样的交叉验证策略（例如，如果样本量非常小，留一交叉验证可能更合适，但计算成本高）。
# 5. 调整模型复杂度、学习率、批次大小等超参数。
# 6. 考虑使用 F1-score 等其他评估指标，它们在类别不平衡时可能比 AUC 提供更直观的信息。