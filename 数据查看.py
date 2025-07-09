import os
import numpy as np
from glob import glob


def count_data_groups(data_dir):
    """统计指定目录下不同类型npy文件的数据组数"""
    # 定义文件匹配模式
    patterns = {
        'Z_imu': '*Z_imu_data.npy',
        'P_imu': '*P_imu_data.npy',
        'Z_pressure': '*Z_pressure_data.npy',
        'P_pressure': '*P_pressure_data.npy'
    }

    results = {}

    # 遍历每种文件类型
    for key, pattern in patterns.items():
        file_paths = glob(os.path.join(data_dir, pattern))

        # 统计文件数量
        file_count = len(file_paths)

        # 统计每组数据的样本数（行数）
        sample_counts = []
        for file_path in file_paths:
            try:
                data = np.load(file_path)
                sample_counts.append(data.shape[0] if data.ndim > 0 else 1)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                sample_counts.append(0)

        # 保存结果
        results[key] = {
            'file_count': file_count,
            'total_samples': sum(sample_counts),
            'samples_per_file': sample_counts
        }

    return results


def print_statistics(results):
    """打印统计结果"""
    print("=" * 50)
    print("数据统计结果")
    print("=" * 50)

    for key, stats in results.items():
        group_type, data_type = key.split('_')
        print(f"\n[{group_type}组-{data_type}数据]")
        print(f"  文件数量: {stats['file_count']}")
        print(f"  总样本数: {stats['total_samples']}")

        if stats['file_count'] > 0:
            print(f"  各文件样本数:")
            for i, count in enumerate(stats['samples_per_file']):
                print(f"    文件{i + 1}: {count}")


if __name__ == "__main__":
    # 指定数据目录，请修改为实际路径
    data_dir = "实验数据/假设数据/处理后的数据"

    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 目录 '{data_dir}' 不存在")
    else:
        # 统计数据
        results = count_data_groups(data_dir)

        # 打印统计结果
        print_statistics(results)