import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from pyts.image import GramianAngularField, RecurrencePlot
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.interpolate import interp1d
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 引入数据增强工具
from sklearn.preprocessing import StandardScaler  # 引入标准化工具

# ------------------ 数据预处理部分 ------------------

# 加载IMU数据和压力数据（从npy文件加载）
imu_data_healthy = np.load(r'E:\DL-Project\实验数据\假设数据\处理后的数据\2Z_imu_data.npy')  # 健康人IMU数据
pressure_data_healthy = np.load(r'E:\DL-Project\实验数据\假设数据\处理后的数据\2Z_pressure_data.npy')  # 健康人压力数据
imu_data_park = np.load(r'E:\DL-Project\实验数据\假设数据\处理后的数据\3P_imu_data.npy')  # 帕金森病人IMU数据
pressure_data_park = np.load(r'E:\DL-Project\实验数据\假设数据\处理后的数据\3P_pressure_data.npy')  # 帕金森病人压力数据

print(f"Healthy IMU data shape: {imu_data_healthy.shape}, Healthy Pressure data shape: {pressure_data_healthy.shape}")
print(f"Parkinson IMU data shape: {imu_data_park.shape}, Parkinson Pressure data shape: {pressure_data_park.shape}")

# 标准化IMU和压力数据
scaler_imu = StandardScaler()
scaler_pressure = StandardScaler()

# 对健康人和帕金森病人的IMU数据标准化
imu_data_healthy = scaler_imu.fit_transform(imu_data_healthy.reshape(-1, imu_data_healthy.shape[-1])).reshape(imu_data_healthy.shape)
imu_data_park = scaler_imu.transform(imu_data_park.reshape(-1, imu_data_park.shape[-1])).reshape(imu_data_park.shape)

# 对健康人和帕金森病人的压力数据标准化
pressure_data_healthy = scaler_pressure.fit_transform(pressure_data_healthy.reshape(-1, pressure_data_healthy.shape[-1])).reshape(pressure_data_healthy.shape)
pressure_data_park = scaler_pressure.transform(pressure_data_park.reshape(-1, pressure_data_park.shape[-1])).reshape(pressure_data_park.shape)

# Gramian Angular Field (GAF) 和 Recurrence Plot (RP) 转换器
image_size = 224
gaf_transformer = GramianAngularField(image_size=image_size, method='summation')
rp_transformer = RecurrencePlot(threshold='point', percentage=20)

# 数据插值函数
def interpolate_data(data, target_size):
    """对时间序列数据进行插值以达到目标长度"""
    interpolated_data = np.array([
        interp1d(np.arange(sample.shape[0]), sample, axis=0, kind='linear')(np.linspace(0, sample.shape[0] - 1, target_size))
        for sample in data
    ])
    return interpolated_data

# 数据预处理函数
def preprocess_data(data, transformer, target_size=224):
    interpolated_data = interpolate_data(data, target_size)
    if interpolated_data.ndim == 2:
        interpolated_data = interpolated_data[:, :, np.newaxis]
    images = []
    for i in range(interpolated_data.shape[-1]):
        channel_data = interpolated_data[:, :, i]
        transformed_images = transformer.fit_transform(channel_data)
        images.append(transformed_images)
    return np.stack(images, axis=-1)
'''def preprocess_data(data, transformer, target_size=224):
    """插值并转换为图像"""
    interpolated_data = interpolate_data(data, target_size)  # 插值到目标长度
    images = []
    for i in range(interpolated_data.shape[-1]):  # 遍历每个通道
        channel_data = interpolated_data[:, :, i]  # 提取单通道
        transformed_images = transformer.fit_transform(channel_data)  # 转换为图像
        images.append(transformed_images)
    return np.stack(images, axis=-1)  # 组合成 (样本数, 224, 224, 通道数)
'''
# 对健康人IMU数据应用GAF
healthy_imu_gaf = preprocess_data(imu_data_healthy, gaf_transformer)  # 健康人IMU数据
# 对健康人压力数据应用RP
healthy_pressure_rp = preprocess_data(pressure_data_healthy, rp_transformer)  # 健康人压力数据

# 对帕金森病人IMU数据应用GAF
parkinsons_imu_gaf = preprocess_data(imu_data_park, gaf_transformer)  # 帕金森病人IMU数据
# 对帕金森病人压力数据应用RP
parkinsons_pressure_rp = preprocess_data(pressure_data_park, rp_transformer)  # 帕金森病人压力数据

# 合并健康人和帕金森病人的数据（IMU + Pressure）
healthy_gaf_rp = np.concatenate([healthy_imu_gaf, healthy_pressure_rp], axis=-1)  # 健康人数据
parkinsons_gaf_rp = np.concatenate([parkinsons_imu_gaf, parkinsons_pressure_rp], axis=-1)  # 帕金森病人数据

# 合并健康人和帕金森病人数据
final_input = np.concatenate([healthy_gaf_rp, parkinsons_gaf_rp], axis=0)  # 合并健康人和帕金森病人数据

# 随机打乱数据和标签
labels_healthy = np.zeros(healthy_gaf_rp.shape[0])  # 健康人标签为0
labels_parkinsons = np.ones(parkinsons_gaf_rp.shape[0])  # 帕金森病人标签为1

labels = np.concatenate([labels_healthy, labels_parkinsons], axis=0)
final_input, labels = shuffle(final_input, labels, random_state=42)  # 随机打乱数据和标签

# 打印最终数据形状
print(f"Final Input Shape after shuffle: {final_input.shape}")
labels = to_categorical(labels, num_classes=2)  # 转为独热编码


# ------------------ 数据增强部分 ------------------

# 创建数据增强生成器
datagen = ImageDataGenerator(
    rotation_range=20,    # 随机旋转角度
    width_shift_range=0.2,  # 随机水平平移
    height_shift_range=0.2, # 随机垂直平移
    shear_range=0.2,      # 随机错切变换
    zoom_range=0.2,       # 随机缩放
    horizontal_flip=True, # 随机水平翻转
    fill_mode='nearest'   # 填充方式
)

# 增强数据的函数
def augment_data(X_train):
    """返回增强后的数据生成器"""
    datagen.fit(X_train)  # 使用数据增强器拟合训练数据
    return datagen.flow(X_train, batch_size=64)  # 直接传递增强后的数据生成器

# ------------------ 模型构建部分 ------------------

# CBAM块
def cbam_block(input_tensor, ratio=8):
    import tensorflow as tf  # 确保导入了 tf

    channel = input_tensor.shape[-1]

    # 通道注意力
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = tf.reduce_max(input_tensor, axis=[1, 2])

    shared_dense = Dense(channel // ratio, activation='relu')
    shared_out_avg = shared_dense(avg_pool)
    shared_out_max = shared_dense(max_pool)

    channel_attention = Dense(channel, activation='sigmoid')(
        Add()([shared_out_avg, shared_out_max])
    )

    # 通道维度扩展（匹配原始张量）
    channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, 1), 1)
    channel_out = input_tensor * channel_attention

    # 空间注意力（保持原实现即可）
    spatial_avg = Conv2D(1, (1, 1), activation='sigmoid')(BatchNormalization()(channel_out))
    spatial_out = channel_out * spatial_avg

    return spatial_out

# 残差块
def residual_block(input_tensor, filters, kernel_size=3, stride=1):
    x = Conv2D(filters, kernel_size, padding='same', strides=stride)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(filters, (1, 1), strides=stride)(input_tensor)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# 构建CNN模型
def build_model(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    x = residual_block(x, 64)
    x = cbam_block(x)
    x = residual_block(x, 128)
    x = cbam_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)  # 在全连接层之前增加Dropout
    x = Dense(64, activation='relu')(x)  # 添加额外的全连接层
    x = Dropout(0.3)(x)  # 添加第二个Dropout
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_tensor, outputs=output)

# 模型定义
model = build_model(final_input.shape[1:], num_classes=2)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'AUC'])
# ------------------ 五倍交叉验证部分 ------------------

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fnr_scores = []  # 用来存储每次交叉验证的假阴性率
auc_scores = []  # 用来存储每次交叉验证的AUC

for fold, (train_index, val_index) in enumerate(kf.split(final_input)):
    print(f"\n--- Fold {fold + 1} ---")

    # 拆分训练集和验证集
    X_train, X_val = final_input[train_index], final_input[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    # 重新构建并编译模型
    model = build_model(final_input.shape[1:], num_classes=2)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'AUC'])
    # 数据增强仅应用于训练集
    X_train_augmented = augment_data(X_train)

    # 回调函数,模型训练
    callbacks = [
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5),
        EarlyStopping(patience=10, restore_best_weights=True)
    ]

    # 训练时，验证集的数据不会进行增强，原样传入
    model.fit(
        X_train_augmented, y_train,
        validation_data=(X_val, y_val),  # 验证集保持原始数据
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=0  # 训练时不打印日志,设置为1查看进度，设置为0隐藏输出
    )

    # 评估模型
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_val_classes, y_pred_classes).ravel()
    fnr = fn / (fn + tp)  # 计算假阴性率
    fnr_scores.append(fnr)

    # 计算AUC
    auc = roc_auc_score(y_val, y_pred)  # 使用roc_auc_score来计算AUC
    auc_scores.append(auc)

    print(f"Fold FNR: {fnr:.4f}, Fold AUC: {auc:.4f}")

# 输出五折交叉验证的平均假阴性率和AUC
mean_fnr = np.mean(fnr_scores)
mean_auc = np.mean(auc_scores)
print(f"Mean False Negative Rate: {mean_fnr:.4f}")
print(f"Mean AUC: {mean_auc:.4f}")