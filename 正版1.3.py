# ✅ 完整训练脚本：启用 EfficientNetV2 + Residual Blocks + CBAM + Fine-tuning + ModelCheckpoint
import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout, Conv2D,
                                     BatchNormalization, ReLU, Add, Multiply,
                                     GlobalMaxPooling2D, Reshape, Concatenate)
from tensorflow.keras.applications import EfficientNetV2S
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import re
from collections import defaultdict

# ✅ CBAM 模块
def cbam_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    avg_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = GlobalMaxPooling2D()(input_feature)
    shared_dense = Dense(channel // ratio, activation='relu')
    shared_out = Dense(channel)

    avg_out = shared_out(shared_dense(avg_pool))
    max_out = shared_out(shared_dense(max_pool))
    channel_attention = Add()([avg_out, max_out])
    channel_attention = tf.nn.sigmoid(channel_attention)
    channel_attention = Reshape((1, 1, channel))(channel_attention)
    channel_refined = Multiply()([input_feature, channel_attention])

    avg_pool_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
    max_pool_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
    spatial_attention = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(spatial_attention)

    refined_feature = Multiply()([channel_refined, spatial_attention])
    return refined_feature

# ✅ 残差模块
def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x

# ✅ 自定义模型构建函数
def build_model(input_shape, num_classes=2):
    base = EfficientNetV2S(input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = True
    x = base.output
    x = residual_block(x, x.shape[-1])
    x = residual_block(x, x.shape[-1])
    x = cbam_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=out)

# ✅ 加载数据相关函数
def generate_patient_id_map(base_dir):
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

# ✅ 模型训练逻辑
def train_single_phase(X_raw, y_raw, pid_raw, idx, phase_name, generator_cls, model_dir):
    X_sel, y_sel, pid_sel = X_raw[idx], y_raw[idx], pid_raw[idx]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_i, val_i) in enumerate(skf.split(X_sel, y_sel, groups=pid_sel)):
        print(f"\n📦 第 {fold + 1}/5 折 - {phase_name}")

        train_gen = generator_cls(X_sel[train_i], y_sel[train_i], pid_sel[train_i])
        val_gen = generator_cls(X_sel[val_i], y_sel[val_i], pid_sel[val_i], shuffle=False)

        if len(train_gen) == 0 or len(val_gen) == 0:
            print("⚠️ 数据不足，跳过此折。"); continue

        model = build_model((224, 224, 3))
        model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        checkpoint_path = os.path.join(model_dir, f"best_model_fold{fold + 1}_{phase_name}.h5")
        callbacks = [
            ReduceLROnPlateau(patience=5, factor=0.5),
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]

        model.fit(train_gen, validation_data=val_gen, epochs=50, callbacks=callbacks, verbose=1)

# ✅ 主运行逻辑
if __name__ == "__main__":
    PHASE_TO_TRAIN = "向前直行阶段"
    DATA_DIR = "/root/autodl-tmp/project/实验数据/假设数据/处理后的数据"
    MODEL_DIR = "./saved_models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        from generator1_4 import AdvancedDataGenerator
    except ImportError:
        print("❌ 无法导入生成器 AdvancedDataGenerator"); sys.exit()

    pid_map = generate_patient_id_map(DATA_DIR)
    X_imu, X_pressure, y_labels, pid_array = load_data(DATA_DIR, pid_map)

    X_imu_std = StandardScaler().fit_transform(X_imu.reshape(-1, X_imu.shape[-1])).reshape(X_imu.shape)
    X_pressure_std = StandardScaler().fit_transform(X_pressure)
    X_combined = np.concatenate([X_imu_std.reshape(X_imu_std.shape[0], -1), X_pressure_std], axis=1)

    phase_indices_dict = split_by_phase_five(pid_array)
    selected_indices = np.array(phase_indices_dict[PHASE_TO_TRAIN])

    train_single_phase(
        X_combined, y_labels, pid_array, selected_indices,
        PHASE_TO_TRAIN, AdvancedDataGenerator, MODEL_DIR
    )
