# ✅ 主程序：单阶段训练 + 可选 GAF 分类主干结构（EffNetV2 / ConvNeXt / ResNeSt 等）
import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import re
from collections import defaultdict

# 【修改点1】将所有模型导入集中管理
# 1. 从 TensorFlow Keras 导入标准模型
from tensorflow.keras.applications import (
    MobileNetV2,
    EfficientNetV2S,
    EfficientNetV2M,
    ResNet50
)

# 2. 尝试从 keras-cv-attention-models 导入扩展模型
#    如果需要使用 ConvNeXt, ResNeSt, SEResNeXt 等模型，请先安装库: pip install keras-cv-attention-models
try:
    from keras_cv_attention_models import resnest, seresnext, convnext

    HAS_EXTRA_MODELS = True
    print("✅ 已成功加载 keras_cv_attention_models 扩展模型库。")
except ImportError:
    HAS_EXTRA_MODELS = False
    print("⚠️ 提示：无法导入 keras_cv_attention_models。ConvNeXt, ResNeSt, SEResNeXt 等模型将不可用。")
    # 定义占位符，以防代码因未定义而报错
    resnest, seresnext, convnext = None, None, None

# 3. 导入我们重构好的生成器
try:
    from generator1_4 import AdvancedDataGenerator
except ImportError:
    print("❌ 致命错误：无法从 generator1_4.py 导入 AdvancedDataGenerator。请确保文件名和路径正确。")
    sys.exit()

# ------------------ 🧪 参数配置区 (只改这里) ------------------
# --- 实验设置 ---
PHASE_TO_TRAIN = "向前直行阶段"  # 可选: "起立阶段", "拐弯阶段", "向前直行阶段", "向后直行阶段", "坐下阶段"
DATA_DIR = r'/root/autodl-tmp/project/实验数据/假设数据/处理后的数据'

# --- 模型选择 ---
# 支持的模型: 'efficientnetv2-s', 'efficientnetv2-m', 'mobilenetv2', 'resnet50'
# 如果安装了扩展库，还支持: 'convnext-tiny', 'convnext-small', 'resnest50', 'seresnext50'
BACKBONE_NAME = 'efficientnetv2-s'

# --- 数据处理参数 ---
WINDOW_SIZE = 256  # 截取的时间窗口大小 (原始论文为896)
STEP = 64  # 滑动窗口的步长
FEATURE_INDEX = 0  # 使用第几个特征来生成图像 (0代表第一个)

# --- 训练参数 ---
TARGET_IMAGE_SIZE = 224  # 模型最终输入的图像尺寸
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4


# ------------------ 数据加载与预处理 (保持不变) ------------------
def generate_patient_id_map(base_dir):
    # ...内容不变...
    patient_id_map = {}
    pattern = re.compile(r'(\d+[ZP])_(imu_data|pressure_data)\.npy')
    for fname in os.listdir(base_dir):
        match = pattern.match(fname)
        if match:
            pid = match.group(1)
            if match.group(2) == 'imu_data':
                patient_id_map[pid] = 0 if 'Z' in pid else 1
    return patient_id_map


def load_data(data_dir, pid_map):
    # ...内容不变...
    imu_list, p_list, label_list, pid_list = [], [], [], []
    for pid, label in pid_map.items():
        imu_path = os.path.join(data_dir, f"{pid}_imu_data.npy")
        p_path = os.path.join(data_dir, f"{pid}_pressure_data.npy")
        if os.path.exists(imu_path) and os.path.exists(p_path):
            imu = np.load(imu_path)
            pres = np.load(p_path)
            min_len = min(len(imu), len(pres))
            imu_list.append(imu[:min_len])
            p_list.append(pres[:min_len])
            label_list.extend([label] * min_len)
            pid_list.extend([pid] * min_len)
    return np.concatenate(imu_list), np.concatenate(p_list), np.array(label_list), np.array(pid_list)


def split_by_phase_five(pids, stand_ratio=0.1, walk_straight_ratio=0.3, turn_ratio=0.2, walk_back_ratio=0.3):
    # ...内容不变...
    grouped = defaultdict(list)
    for i, pid in enumerate(pids):
        grouped[pid].append(i)

    stand, forward, turn, back, sit = [], [], [], [], []
    for pid, idxs in grouped.items():
        n = len(idxs)
        s_end = int(n * stand_ratio)
        f_end = s_end + int(n * walk_straight_ratio)
        t_end = f_end + int(n * turn_ratio)
        b_end = t_end + int(n * walk_back_ratio)
        stand.extend(idxs[:s_end])
        forward.extend(idxs[s_end:f_end])
        turn.extend(idxs[f_end:t_end])
        back.extend(idxs[t_end:b_end])
        sit.extend(idxs[b_end:])
    return {"起立阶段": stand, "向前直行阶段": forward, "拐弯阶段": turn, "向后直行阶段": back, "坐下阶段": sit}

# ------------------ 【修改点2】完善的模型工厂 ------------------
def build_backbone(model_name, input_shape):
    name = model_name.lower()
    # 从 tf.keras.applications 加载
    if name == 'efficientnetv2-s':
        return EfficientNetV2S(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == 'efficientnetv2-m':
        return EfficientNetV2M(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == 'mobilenetv2':
        return MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif name == 'resnet50':
        return ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    # 从 keras_cv_attention_models 加载
    elif HAS_EXTRA_MODELS:
        if name == 'convnext-tiny':
            return convnext.ConvNeXtTiny(input_shape=input_shape, include_top=False, pretrained=True)
        elif name == 'convnext-small':
            return convnext.ConvNeXtSmall(input_shape=input_shape, include_top=False, pretrained=True)
        elif name == 'seresnext50':
            return seresnext.SEResNeXt50(input_shape=input_shape, include_top=False, pretrained=True)
        elif name == 'resnest50':
            return resnest.ResNeSt50(input_shape=input_shape, include_top=False, pretrained=True)
    # 如果找不到模型或未安装依赖，则报错
    raise ValueError(f"模型 '{model_name}' 无效或其依赖库未安装。请检查 BACKBONE_NAME 参数。")


def build_model(input_shape, num_classes=2, model_name='efficientnetv2-s'):
    base = build_backbone(model_name, input_shape)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=out)


# ------------------ 单阶段训练（已更新） ------------------
def train_single_phase(X_raw, y_raw, pid_raw, idx, phase_name):
    X_sel, y_sel, pid_sel = X_raw[idx], y_raw[idx], pid_raw[idx]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, accs = [], []

    for fold, (train_i, val_i) in enumerate(skf.split(X_sel, y_sel, groups=pid_sel)):
        print(f"\n📦 第 {fold + 1}/5 折 - {phase_name}")

        train_gen = AdvancedDataGenerator(
            raw_data=X_sel[train_i], raw_labels=y_sel[train_i], raw_patient_ids=pid_sel[train_i],
            window_size=WINDOW_SIZE, step=STEP, target_image_size=TARGET_IMAGE_SIZE,
            feature_index=FEATURE_INDEX, batch_size=BATCH_SIZE, shuffle=True, is_rgb=True
        )
        val_gen = AdvancedDataGenerator(
            raw_data=X_sel[val_i], raw_labels=y_sel[val_i], raw_patient_ids=pid_sel[val_i],
            window_size=WINDOW_SIZE, step=STEP, target_image_size=TARGET_IMAGE_SIZE,
            feature_index=FEATURE_INDEX, batch_size=BATCH_SIZE, shuffle=False, is_rgb=True
        )

        if len(train_gen) == 0 or len(val_gen) == 0:
            print("⚠️ 数据不足，无法生成至少一个批次，跳过此折。")
            continue

        # 【修改点3】修复BUG：确保模型和生成器的图像尺寸一致
        model_input_shape = (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3)
        model = build_model(input_shape=model_input_shape, num_classes=2, model_name=BACKBONE_NAME)
        model.compile(optimizer=Adam(LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
                  callbacks=[ReduceLROnPlateau(patience=5, factor=0.5),
                             EarlyStopping(patience=10, restore_best_weights=True)],
                  verbose=1)

        # 评估逻辑...
        y_true, y_score = [], []
        for i in range(len(val_gen)):
            Xb, yb = val_gen[i]
            preds = model.predict(Xb, verbose=0)
            y_score.extend(preds[:, 1])
            y_true.extend(np.argmax(yb, axis=1))

        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
            _, acc = model.evaluate(val_gen, verbose=0)
            aucs.append(auc)
            accs.append(acc)
            print(f"✅ Fold AUC = {auc:.4f}, Accuracy = {acc:.4f}")
        else:
            print("⚠️ 验证集标签单一，无法计算AUC。")

    if aucs:
        print(f"\n📊 【{phase_name}】平均 AUC = {np.mean(aucs):.4f}")
        print(f"📊 【{phase_name}】平均 Acc = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    else:
        print(f"⚠️ 【{phase_name}】无有效结果。")


# ------------------ 主运行入口 ------------------
if __name__ == "__main__":
    print(f"🧪 即将训练阶段: {PHASE_TO_TRAIN}，使用模型: {BACKBONE_NAME}")
    pid_map = generate_patient_id_map(DATA_DIR)
    X_imu, X_pressure, y_labels, pid_array = load_data(DATA_DIR, pid_map)

    X_imu_std = StandardScaler().fit_transform(X_imu.reshape(-1, X_imu.shape[-1])).reshape(X_imu.shape)
    X_pressure_std = StandardScaler().fit_transform(X_pressure)
    X_combined = np.concatenate([X_imu_std.reshape(X_imu_std.shape[0], -1), X_pressure_std], axis=1)

    phase_indices_dict = split_by_phase_five(pid_array)
    selected_indices = np.array(phase_indices_dict[PHASE_TO_TRAIN])

    train_single_phase(X_combined, y_labels, pid_array, selected_indices, PHASE_TO_TRAIN)