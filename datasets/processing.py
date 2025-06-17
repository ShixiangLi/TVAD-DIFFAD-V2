import numpy as np
from PIL import Image
import os
import sys
import random

# --- 配置参数 ---
VIDEO_DATA_PATH = 'datasets/raw_data/video_data.npy'  # 图像数据 .npy 文件路径 (N, C, H, W)
LABEL_DATA_PATH = 'datasets/raw_data/labels.npy'      # 标签数据 .npy 文件路径 (N, H, W)
CURRENT_DATA_PATH = 'datasets/raw_data/current_data.npy' # 新增：电流数据 .npy 文件路径 (N, 24, 3)

# --- 输出目录结构参数 ---
OUTPUT_BASE_DIR = 'datasets/combustion_dataset/chamber' 

# 图像输出目录
TRAIN_GOOD_IMAGE_DIR = os.path.join(OUTPUT_BASE_DIR, 'train', 'good')
TEST_GOOD_IMAGE_DIR = os.path.join(OUTPUT_BASE_DIR, 'test', 'good')
TEST_ANOMALY_IMAGE_DIR_PREFIX = os.path.join(OUTPUT_BASE_DIR, 'test') 

# 真值掩码输出目录
TEST_GOOD_GT_DIR = os.path.join(OUTPUT_BASE_DIR, 'ground_truth', 'good')
TEST_ANOMALY_GT_DIR_PREFIX = os.path.join(OUTPUT_BASE_DIR, 'ground_truth')

# 新增：电流特征输出目录
TRAIN_GOOD_CURRENT_DIR = os.path.join(OUTPUT_BASE_DIR, 'current_features', 'train', 'good')
TEST_GOOD_CURRENT_DIR = os.path.join(OUTPUT_BASE_DIR, 'current_features', 'test', 'good')
TEST_ANOMALY_CURRENT_DIR_PREFIX = os.path.join(OUTPUT_BASE_DIR, 'current_features', 'test')


FILENAME_PREFIX = 'frame'    # 输出文件名的前缀
NUM_TEST_NORMAL = 2000        # 测试集中正常样本数量
NUM_TEST_ABNORMAL = 2000      # 测试集中异常样本数量

DEFAULT_ANOMALY_CLASS_NAME = 'anomaly' # 默认异常类别文件夹名
# ----------------

def process_and_split_data_custom(video_path, label_path, current_data_path,
                                  train_good_img_dir,
                                  test_good_img_dir, test_anomaly_img_dir_base,
                                  test_good_gt_dir, test_anomaly_gt_dir_base,
                                  train_good_current_dir, 
                                  test_good_current_dir, test_anomaly_current_dir_base,
                                  anomaly_class_name,
                                  num_test_normal, num_test_abnormal,
                                  prefix):
    """
    加载图像(N, C, H, W)、标签(N, H, W)和电流数据(N, 24, 3)，
    按指定数量随机分割为训练/测试集，并将图像/掩码保存为 PNG，电流特征保存为 .npy。
    """
    print(f"开始自定义处理与分割...")
    print(f"图像数据路径: {video_path}")
    print(f"标签数据路径: {label_path}")
    print(f"电流数据路径: {current_data_path}")

    # --- 1. 加载数据 ---
    try:
        print("正在加载图像数据...")
        video_data = np.load(video_path)
        print(f"图像数据加载成功，维度: {video_data.shape}")
    except FileNotFoundError:
        print(f"错误: 找不到图像数据文件 {video_path}")
        sys.exit(1)
    except Exception as e:
        print(f"加载图像数据时出错: {e}")
        sys.exit(1)

    try:
        print("正在加载标签数据...")
        label_data = np.load(label_path)
        print(f"标签数据加载成功，维度: {label_data.shape}")
    except FileNotFoundError:
        print(f"错误: 找不到标签数据文件 {label_path}")
        sys.exit(1)
    except Exception as e:
        print(f"加载标签数据时出错: {e}")
        sys.exit(1)

    try: 
        print("正在加载电流数据...")
        current_data = np.load(current_data_path)
        print(f"电流数据加载成功，维度: {current_data.shape}")
    except FileNotFoundError:
        print(f"错误: 找不到电流数据文件 {current_data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"加载电流数据时出错: {e}")
        sys.exit(1)

    # --- 2. 验证数据维度 (MODIFIED FOR N, C, H, W) ---
    if video_data.ndim != 4:
        print(f"错误: 图像数据的维度必须为 4 (N, C, H, W)。实际维度: {video_data.ndim}")
        sys.exit(1)
    if label_data.ndim != 3:
        print(f"错误: 标签数据的维度必须为 3 (N, H, W)。实际维度: {label_data.ndim}")
        sys.exit(1)
    if current_data.ndim != 3 or current_data.shape[1:] != (24, 3): 
        print(f"错误: 电流数据的维度必须为 3 (N, 24, 3)。实际维度: {current_data.shape}")
        sys.exit(1)

    N_img, C_img, H_img, W_img = video_data.shape # MODIFIED: Unpack N, C, H, W
    N_lbl = label_data.shape[0]
    N_curr = current_data.shape[0]

    if not (N_img == N_lbl == N_curr): 
        print(f"错误: 图像数据 ({N_img}), 标签数据 ({N_lbl}), 和电流数据 ({N_curr}) 的样本数量 (N) 不匹配。")
        sys.exit(1)
    N = N_img

    if H_img != label_data.shape[1] or W_img != label_data.shape[2]:
        print("警告: 图像数据和标签数据的空间维度 (H, W) 不匹配。将使用图像维度。")
    H, W = H_img, W_img

    print(f"总样本数 (N): {N}, 图像尺寸: ({C_img}, {H}, {W}), 电流特征维度: ({current_data.shape[1]}, {current_data.shape[2]})")

    # --- 3. 创建输出目录 ---
    actual_test_anomaly_img_dir = os.path.join(test_anomaly_img_dir_base, anomaly_class_name)
    actual_test_anomaly_gt_dir = os.path.join(test_anomaly_gt_dir_base, anomaly_class_name)
    actual_test_anomaly_current_dir = os.path.join(test_anomaly_current_dir_base, anomaly_class_name)

    os.makedirs(train_good_img_dir, exist_ok=True)
    os.makedirs(test_good_img_dir, exist_ok=True)
    os.makedirs(actual_test_anomaly_img_dir, exist_ok=True)
    os.makedirs(test_good_gt_dir, exist_ok=True)
    os.makedirs(actual_test_anomaly_gt_dir, exist_ok=True)
    
    os.makedirs(train_good_current_dir, exist_ok=True) 
    os.makedirs(test_good_current_dir, exist_ok=True)   
    os.makedirs(actual_test_anomaly_current_dir, exist_ok=True)
    print(f"已确保输出目录存在。")

    # --- 4. 预分类样本索引 ---
    normal_indices = []
    abnormal_indices = []
    for i in range(N):
        label_map = label_data[i] 
        if np.any(label_map != 0):
            abnormal_indices.append(i)
        else:
            normal_indices.append(i)
    print(f"找到 {len(normal_indices)} 个正常样本和 {len(abnormal_indices)} 个异常样本。")

    # --- 5. 随机抽样测试集 ---
    random.shuffle(normal_indices)
    random.shuffle(abnormal_indices)

    actual_num_test_normal = min(num_test_normal, len(normal_indices))
    test_normal_indices = normal_indices[:actual_num_test_normal]
    if actual_num_test_normal < num_test_normal:
        print(f"警告: 正常样本不足 {num_test_normal} 个，实际抽取 {actual_num_test_normal} 个作为测试集正常样本。")

    actual_num_test_abnormal = min(num_test_abnormal, len(abnormal_indices))
    test_abnormal_indices = abnormal_indices[:actual_num_test_abnormal]
    if actual_num_test_abnormal < num_test_abnormal:
        print(f"警告: 异常样本不足 {num_test_abnormal} 个，实际抽取 {actual_num_test_abnormal} 个作为测试集异常样本。")
    print(f"将抽取 {len(test_normal_indices)} 个正常样本和 {len(test_abnormal_indices)} 个异常样本作为测试集。")

    # --- 6. 确定训练集索引 ---
    train_normal_indices = normal_indices[actual_num_test_normal:]
    print(f"剩余 {len(train_normal_indices)} 个正常样本作为训练集。")

    # --- 7. 保存文件 (MODIFIED FOR N, C, H, W and Bug Fix) ---
    num_digits = len(str(N - 1)) 

    # 保存训练集正常样本 (图像、电流特征)
    print("\n正在保存训练集正常样本...")
    for i, original_idx in enumerate(train_normal_indices):
        img_slice_chw = video_data[original_idx] # Shape (C, H, W)
        current_feature_vector = current_data[original_idx] 

        base_filename_img = f"{prefix}_{original_idx:0{num_digits}d}.png"
        base_filename_curr = f"{prefix}_{original_idx:0{num_digits}d}.npy" 
        
        image_save_path = os.path.join(train_good_img_dir, base_filename_img)
        current_feature_save_path = os.path.join(train_good_current_dir, base_filename_curr)

        img_array = None
        mode = None
        if C_img == 1:
            img_array = img_slice_chw.squeeze(axis=0); mode = 'L' # (1,H,W) -> (H,W)
        elif C_img == 3:
            img_array = img_slice_chw.transpose(1, 2, 0); mode = 'RGB' # (C,H,W) -> (H,W,C)
        elif C_img == 4:
            img_array = img_slice_chw.transpose(1, 2, 0); mode = 'RGBA' # (C,H,W) -> (H,W,C)
        elif C_img == 0:
            print(f"错误: 样本 {original_idx} 通道数为0"); continue
        else:
            img_array = img_slice_chw[0, :, :]; mode = 'L';
            print(f"警告: 图像通道数 {C_img} 未明确处理 ({original_idx})，默认使用第一通道转灰度图。")

        if img_array is not None:
            if img_array.dtype != np.uint8:
                if np.max(img_array) <= 1.0 and np.min(img_array) >= 0.0: 
                    img_array = (img_array * 255).astype(np.uint8)
                else: 
                    img_array = img_array.astype(np.uint8)
            try:
                pil_image = Image.fromarray(img_array, mode=mode)
                pil_image.save(image_save_path)
            except Exception as e:
                print(f"\n保存训练图像 {image_save_path} (索引 {original_idx}) 时出错: {e}")
        
        try:
            np.save(current_feature_save_path, current_feature_vector)
        except Exception as e:
            print(f"\n保存训练电流特征 {current_feature_save_path} (索引 {original_idx}) 时出错: {e}")

        if (i + 1) % 100 == 0: print(f"\r已保存 {i+1}/{len(train_normal_indices)} 个训练样本", end="")
    print(f"\n训练集正常样本保存完毕。")

    # 保存测试集正常样本 (图像、全零掩码、电流特征)
    print("\n正在保存测试集正常样本及其掩码与电流特征...")
    for i, original_idx in enumerate(test_normal_indices):
        img_slice_chw = video_data[original_idx] # Shape (C, H, W)
        current_feature_vector = current_data[original_idx]

        base_filename_img = f"{prefix}_{original_idx:0{num_digits}d}.png"
        base_filename_curr = f"{prefix}_{original_idx:0{num_digits}d}.npy"
        
        image_save_path = os.path.join(test_good_img_dir, base_filename_img)
        mask_save_path = os.path.join(test_good_gt_dir, base_filename_img) 
        current_feature_save_path = os.path.join(test_good_current_dir, base_filename_curr)

        img_array = None; mode = None
        if C_img == 1: img_array = img_slice_chw.squeeze(axis=0); mode = 'L'
        elif C_img == 3: img_array = img_slice_chw.transpose(1, 2, 0); mode = 'RGB'
        elif C_img == 4: img_array = img_slice_chw.transpose(1, 2, 0); mode = 'RGBA'
        elif C_img == 0: print(f"错误: 样本 {original_idx} 通道数为0"); continue
        else: img_array = img_slice_chw[0, :, :]; mode = 'L';
        
        if img_array is not None:
            if img_array.dtype != np.uint8:
                if np.max(img_array) <= 1.0 and np.min(img_array) >= 0.0: img_array = (img_array * 255).astype(np.uint8)
                else: img_array = img_array.astype(np.uint8)
            try: Image.fromarray(img_array, mode=mode).save(image_save_path)
            except Exception as e: print(f"\n保存测试(正常)图像 {image_save_path} (索引 {original_idx}) 时出错: {e}")

        zero_mask = np.zeros((H, W), dtype=np.uint8)
        try: Image.fromarray(zero_mask, mode='L').save(mask_save_path)
        except Exception as e: print(f"\n保存测试(正常)掩码 {mask_save_path} (索引 {original_idx}) 时出错: {e}")
        
        try: np.save(current_feature_save_path, current_feature_vector)
        except Exception as e: print(f"\n保存测试(正常)电流特征 {current_feature_save_path} (索引 {original_idx}) 时出错: {e}")

        if (i + 1) % 100 == 0: print(f"\r已保存 {i+1}/{len(test_normal_indices)} 个测试集正常样本", end="")
    print(f"\n测试集正常样本及其掩码与电流特征保存完毕。")

    # 保存测试集异常样本 (图像、二值掩码、电流特征)
    print("\n正在保存测试集异常样本及其掩码与电流特征...")
    for i, original_idx in enumerate(test_abnormal_indices):
        img_slice_chw = video_data[original_idx] # Shape (C, H, W)
        label_map = label_data[original_idx]
        current_feature_vector = current_data[original_idx]

        base_filename_img = f"{prefix}_{original_idx:0{num_digits}d}.png"
        base_filename_curr = f"{prefix}_{original_idx:0{num_digits}d}.npy"

        image_save_path = os.path.join(actual_test_anomaly_img_dir, base_filename_img)
        mask_save_path = os.path.join(actual_test_anomaly_gt_dir, base_filename_img) 
        current_feature_save_path = os.path.join(actual_test_anomaly_current_dir, base_filename_curr)

        img_array = None; mode = None
        if C_img == 1: img_array = img_slice_chw.squeeze(axis=0); mode = 'L'
        elif C_img == 3: img_array = img_slice_chw.transpose(1, 2, 0); mode = 'RGB'
        elif C_img == 4: img_array = img_slice_chw.transpose(1, 2, 0); mode = 'RGBA'
        elif C_img == 0: print(f"错误: 样本 {original_idx} 通道数为0"); continue
        else: img_array = img_slice_chw[0, :, :]; mode = 'L';
        
        if img_array is not None:
            if img_array.dtype != np.uint8:
                if np.max(img_array) <= 1.0 and np.min(img_array) >= 0.0: img_array = (img_array * 255).astype(np.uint8)
                else: img_array = img_array.astype(np.uint8)
            try: Image.fromarray(img_array, mode=mode).save(image_save_path)
            except Exception as e: print(f"\n保存测试(异常)图像 {image_save_path} (索引 {original_idx}) 时出错: {e}")

        binary_mask = np.zeros_like(label_map, dtype=np.uint8)
        binary_mask[label_map != 0] = 255
        try: Image.fromarray(binary_mask, mode='L').save(mask_save_path)
        except Exception as e: print(f"\n保存测试(异常)掩码 {mask_save_path} (索引 {original_idx}) 时出错: {e}")

        try: np.save(current_feature_save_path, current_feature_vector)
        except Exception as e: print(f"\n保存测试(异常)电流特征 {current_feature_save_path} (索引 {original_idx}) 时出错: {e}")

        if (i + 1) % 50 == 0: print(f"\r已保存 {i+1}/{len(test_abnormal_indices)} 个测试集异常样本", end="")
    print(f"\n测试集异常样本及其掩码与电流特征保存完毕。")

    print(f"\n--- 自定义处理与分割完成！ ---")
    print(f"总样本数: {N}")
    print(f"训练集 (正常): {len(train_normal_indices)} (图像保存至 '{train_good_img_dir}', 电流特征保存至 '{train_good_current_dir}')")
    print(f"测试集 (正常): {len(test_normal_indices)} (图像保存至 '{test_good_img_dir}', 掩码至 '{test_good_gt_dir}', 电流特征至 '{test_good_current_dir}')")
    print(f"测试集 (异常 - '{anomaly_class_name}'): {len(test_abnormal_indices)} (图像至 '{actual_test_anomaly_img_dir}', 掩码至 '{actual_test_anomaly_gt_dir}', 电流特征至 '{actual_test_anomaly_current_dir}')")


if __name__ == "__main__":
    # 根据您的数据情况，定义异常类别文件夹的名称
    # 例如，如果您的异常类别是 'burn'
    anomaly_folder_name = "burn" 

    process_and_split_data_custom(
        video_path=VIDEO_DATA_PATH,
        label_path=LABEL_DATA_PATH,
        current_data_path=CURRENT_DATA_PATH, 
        train_good_img_dir=TRAIN_GOOD_IMAGE_DIR,
        test_good_img_dir=TEST_GOOD_IMAGE_DIR,
        test_anomaly_img_dir_base=TEST_ANOMALY_IMAGE_DIR_PREFIX, 
        test_good_gt_dir=TEST_GOOD_GT_DIR,
        test_anomaly_gt_dir_base=TEST_ANOMALY_GT_DIR_PREFIX,  
        train_good_current_dir=TRAIN_GOOD_CURRENT_DIR, 
        test_good_current_dir=TEST_GOOD_CURRENT_DIR,   
        test_anomaly_current_dir_base=TEST_ANOMALY_CURRENT_DIR_PREFIX, 
        anomaly_class_name=anomaly_folder_name,
        num_test_normal=NUM_TEST_NORMAL,
        num_test_abnormal=NUM_TEST_ABNORMAL,
        prefix=FILENAME_PREFIX
    )