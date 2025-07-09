import numpy as np
import re

def parse_and_save_imu_pressure(file_path, start_line=0, imu_output_path='park_imu_data.npy', pressure_output_path='park_pressure_data.npy'):
    imu_data = []
    pressure_data = []

    # 改进的正则表达式：
    # 匹配：
    #   - 可选的负号 (-?)
    #   - 一个或多个数字 (\d+)
    #   - 必须有一个小数点 (\.)
    #   - 必须有四位数字 (\d{4})
    # 这个模式专门针对 "X.XXXX" 形式的浮点数，且小数点后必须是四位
    float_pattern = re.compile(r'-?\d+\.\d{4}')

    # 读取文件，跳过空行
    with open(file_path, 'r') as f:
        raw_lines = f.readlines()
        # 去除前 start_line 行，剔除空行和换行
        lines = [line.strip() for line in raw_lines[start_line:] if line.strip()]

    # 每 6 行一组（去除空行后）
    if len(lines) % 6 != 0:
        print(f"⚠️ 数据行数 {len(lines)} 不是6的整数倍，最后一组可能会被忽略。")

    for i in range(0, len(lines) - 5, 6):
        try:
            # 修正数据顺序：
            # 第一行：imu腰部角速度的x，y，z数据 (waist_gyro)
            # 第二行：左腿部角速度x，y，z数据 (leg_gyro)
            # 第三行：腰部加速度的x，y，z数据 (waist_acc)
            # 第四行：左腿部加速度数据的x，y，z数据 (leg_acc)
            waist_gyro_str = lines[i]
            leg_gyro_str = lines[i+1]
            waist_acc_str = lines[i+2]
            leg_acc_str = lines[i+3]

            # 使用 re.findall 提取所有匹配的浮点数
            waist_gyro = [float(x) for x in float_pattern.findall(waist_gyro_str)]
            leg_gyro = [float(x) for x in float_pattern.findall(leg_gyro_str)]
            waist_acc = [float(x) for x in float_pattern.findall(waist_acc_str)]
            leg_acc = [float(x) for x in float_pattern.findall(leg_acc_str)]

            # 检查解析出的IMU数据长度是否正确
            if not (len(waist_gyro) == 3 and len(waist_acc) == 3 and
                    len(leg_gyro) == 3 and len(leg_acc) == 3):
                print(f"⚠️第 {i//6+1} 组IMU数据格式有误（解析到的IMU值数量不为3），跳过。数据行: {lines[i]} 到 {lines[i+3]}")
                continue

            imu_sample = [waist_gyro, waist_acc, leg_gyro, leg_acc]
            imu_data.append(imu_sample)

            rear_pressure = int(lines[i + 4])
            front_pressure = int(lines[i + 5])
            pressure_data.append([rear_pressure, front_pressure])

        except ValueError as ve:
            print(f"❌ 解析第 {i//6+1} 组数据时出错（数值转换错误）：{ve}。原始行: {lines[i]} - {lines[i+5]}")
            continue
        except IndexError as ie:
            print(f"❌ 解析第 {i//6+1} 组数据时出错（索引错误，可能是文件末尾数据不完整）：{ie}")
            break # 遇到索引错误通常表示文件已读完或数据不完整，可以跳出循环
        except Exception as e:
            print(f"❌ 解析第 {i//6+1} 组数据时出错：{e}")
            continue

    # 转换为 numpy 数组并保存
    imu_data = np.array(imu_data)
    pressure_data = np.array(pressure_data)

    print(f"✅ IMU数据 shape: {imu_data.shape}（期望: (样本数, 4, 3)）")
    print(f"✅ 压力数据 shape: {pressure_data.shape}（期望: (样本数, 2)）")

    np.save(imu_output_path, imu_data)
    np.save(pressure_output_path, pressure_data)
    print(f"数据已保存到 {imu_output_path} 和 {pressure_output_path}")

# 示例调用方式：
parse_and_save_imu_pressure(
    file_path=r'E:\DL-Project\实验数据\假设数据\原数据\P9.txt',
    start_line=4,                       # 根据你的p9.txt文件实际情况调整，如果前几行是注释或标题，可以设置为 >0
    imu_output_path=r'E:\DL-Project\实验数据\假设数据\处理后的数据\9P_imu_data.npy',
    pressure_output_path=r'E:\DL-Project\实验数据\假设数据\处理后的数据\9P_pressure_data.npy'
)