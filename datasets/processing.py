import numpy as np
from PIL import Image
import os
from pathlib import Path
import random
import shutil

# --- 1. 配置参数 ---
# 您可以在这里设置测试集中需要的样本数量
X_NORMAL_TEST_SAMPLES = 2000  # 测试集中需要的【正常】样本数量
Y_ABNORMAL_TEST_SAMPLES = 2000 # 测试集中需要的【异常】样本数量

# --- 2. 定义路径 ---
# 原始数据输入路径
raw_data_path = Path("datasets/raw_data")

# 新数据集的根路径
output_root = Path("datasets/combustion_dataset/chamber")

# 清理旧数据 (可选，但推荐，以防重复运行导致数据混淆)
if output_root.exists():
    print(f"警告：输出目录 '{output_root}' 已存在，将删除并重建...")
    shutil.rmtree(output_root)
    print("旧目录已删除。")

# 定义所有需要的输出路径
paths = {
    "train_images": output_root / "train/good",
    "train_currents": output_root / "current_features/train/good",
    "test_images_good": output_root / "test/good",
    "test_images_burn": output_root / "test/burn",
    "test_currents_good": output_root / "current_features/test/good",
    "test_currents_burn": output_root / "current_features/test/burn",
    "gt_good": output_root / "ground_truth/good",
    "gt_burn": output_root / "ground_truth/burn",
}

# --- 3. 创建所有输出文件夹 ---
print("正在创建新的数据集文件结构...")
for key, path in paths.items():
    path.mkdir(parents=True, exist_ok=True)
print("文件结构创建完毕。")


# --- 4. 加载原始 .npy 文件 ---
print("正在加载原始数据文件...")
try:
    video_data = np.load(raw_data_path / "video_data.npy")
    current_data = np.load(raw_data_path / "current_data.npy")
    labels_data = np.load(raw_data_path / "labels.npy")
except FileNotFoundError as e:
    print(f"错误：找不到文件 - {e}")
    exit()
print("数据加载成功。")


# --- 5. 分类：区分正常与异常样本 ---
print("正在分类所有样本 (正常/异常)...")
normal_indices = []
abnormal_indices = []

num_total_samples = labels_data.shape[0]
for i in range(num_total_samples):
    label_slice = labels_data[i]
    # np.all(array == 0) 会检查数组中的所有元素是否都为0
    if np.all(label_slice == 0):
        normal_indices.append(i)
    else:
        abnormal_indices.append(i)

print(f"分类完成：找到 {len(normal_indices)} 个正常样本和 {len(abnormal_indices)} 个异常样本。")


# --- 6. 分配训练集与测试集 ---
# 随机打乱索引列表以确保抽样是随机的
random.shuffle(normal_indices)
random.shuffle(abnormal_indices)

# 验证样本数量是否足够
if len(normal_indices) < X_NORMAL_TEST_SAMPLES:
    print(f"错误：正常样本不足 {X_NORMAL_TEST_SAMPLES} 个，无法创建测试集。请减少 X 的值。")
    exit()
if len(abnormal_indices) < Y_ABNORMAL_TEST_SAMPLES:
    print(f"警告：异常样本数量 ({len(abnormal_indices)}) 少于请求的 {Y_ABNORMAL_TEST_SAMPLES} 个。")
    print("将使用所有可用的异常样本作为测试集。")
    Y_ABNORMAL_TEST_SAMPLES = len(abnormal_indices)

# 分配索引
test_normal_indices = normal_indices[:X_NORMAL_TEST_SAMPLES]
train_normal_indices = normal_indices[X_NORMAL_TEST_SAMPLES:]
test_abnormal_indices = abnormal_indices[:Y_ABNORMAL_TEST_SAMPLES]

print("-" * 30)
print("数据集分配计划：")
print(f"  - 训练集 (正常): {len(train_normal_indices)} 个")
print(f"  - 测试集 (正常): {len(test_normal_indices)} 个")
print(f"  - 测试集 (异常): {len(test_abnormal_indices)} 个")
print("-" * 30)


# --- 7. 处理并保存所有文件 ---
def process_and_save_files(indices, data_type):
    """一个辅助函数，用于处理并保存指定索引和类型的数据"""
    print(f"正在处理和保存 {len(indices)} 个 {data_type} 样本...")
    for index in indices:
        # a. 处理图像
        img_array_chw = video_data[index]
        img_array_hwc = np.transpose(img_array_chw, (1, 2, 0))
        if img_array_hwc.max() <= 1.0 and img_array_hwc.min() >= 0.0:
            img_array_hwc = (img_array_hwc * 255).astype(np.uint8)
        else:
            img_array_hwc = img_array_hwc.astype(np.uint8)
        img = Image.fromarray(img_array_hwc)

        # b. 处理电流数据
        current_array = current_data[index]

        # c. 根据类型保存到不同位置
        if data_type == "训练集-正常":
            img.save(paths["train_images"] / f"{index:04d}.png")
            np.save(paths["train_currents"] / f"{index:04d}.npy", current_array)

        elif data_type == "测试集-正常":
            img.save(paths["test_images_good"] / f"{index:04d}.png")
            np.save(paths["test_currents_good"] / f"{index:04d}.npy", current_array)
            # 正常样本的标签是纯黑的
            label_img = Image.fromarray(np.zeros_like(labels_data[index]), mode='L')
            label_img.save(paths["gt_good"] / f"{index:04d}.png")

        elif data_type == "测试集-异常":
            img.save(paths["test_images_burn"] / f"{index:04d}.png")
            np.save(paths["test_currents_burn"] / f"{index:04d}.npy", current_array)
            # 异常样本的标签是黑白的
            binary_label_array = (labels_data[index] > 0).astype(np.uint8)
            visual_label_array = binary_label_array * 255
            label_img = Image.fromarray(visual_label_array, mode='L')
            label_img.save(paths["gt_burn"] / f"{index:04d}.png")

# 执行保存操作
process_and_save_files(train_normal_indices, "训练集-正常")
process_and_save_files(test_normal_indices, "测试集-正常")
process_and_save_files(test_abnormal_indices, "测试集-异常")


# --- 8. 最终总结 ---
print("\n" + "=" * 50)
print("数据集准备完成！")
print("文件已保存至以下目录结构：")
print(f"└── {output_root}")
for path in sorted(output_root.glob('**/*')):
    if path.parent != output_root and path.is_dir():
         print(f"    └── {path.relative_to(output_root.parent)}")
print("=" * 50)