import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
# from PIL import Image # PIL.Image is not directly used for loading .npy
from torchvision import transforms # Not directly used in these custom classes for .npy
import random
from data.perlin import rand_perlin_2d_np

# --- Helper function to construct current feature path ---
def get_current_feature_path(image_path, dataset_base_path, phase, image_type_folder_name, classname_in_path_struct=True):
    """
    Constructs the path to the .npy file for current features.
    image_path: Full path to the .png image file.
    dataset_base_path: Base path of the dataset, e.g., 'datasets/combustion_dataset' (without 'chamber').
                       Or it could be 'datasets/combustion_dataset/chamber' if classname is already in dataset_base_path.
    phase: 'train' or 'test'.
    image_type_folder_name: 'good' or anomaly type like 'burn'.
    classname_in_path_struct: True if classname (e.g. 'chamber') is part of the path structure for current_features
                              after dataset_base_path.
    """
    base_filename = os.path.basename(image_path)
    npy_filename = base_filename.replace('.png', '.npy').replace('.PNG', '.npy').replace('.jpg', '.npy').replace('.JPG', '.npy')

    if "/train/good/" in image_path:
        cf_path = image_path.replace("/train/good/", f"/current_features/train/good/").replace(base_filename, npy_filename)
    elif f"/test/{image_type_folder_name}/" in image_path:
        cf_path = image_path.replace(f"/test/{image_type_folder_name}/", f"/current_features/test/{image_type_folder_name}/").replace(base_filename, npy_filename)
    else: # Fallback or more robust reconstruction needed

        if phase == 'train':

            path_parts = list(os.path.split(image_path)) # [head, tail=filename]

            class_root_path = os.path.dirname(os.path.dirname(os.path.dirname(image_path))) # Up to 'bottle'
            cf_path = os.path.join(class_root_path, 'current_features', phase, image_type_folder_name, npy_filename)

            try:
                path_segments = image_path.split(os.sep)
                # Find index of phase ('train' or 'test')
                phase_index = -1
                for i, segment in enumerate(path_segments):
                    if segment == phase:
                        phase_index = i
                        break
                
                if phase_index != -1 and phase_index > 0:
                    class_level_path = os.sep.join(path_segments[:phase_index]) # Path up to 'chamber'
                    cf_path = os.path.join(class_level_path, 'current_features', phase, image_type_folder_name, npy_filename)
                else:
                    raise ValueError("Phase not found in path for CF construction")

            except Exception as e:
                 print(f"Critical Error: Failed to construct current_feature_path for {image_path}. Error: {e}")
                 return None # Or raise error

    if not os.path.exists(cf_path):
        print(f"Warning: Current feature file not found at {cf_path} (derived from {image_path})")
        return None # File not found
        
    return cf_path

class CustomTestDataset(Dataset):
    def __init__(self, data_path, classname, img_size, args=None): 
        # data_path is 'datasets/your_custom_dataset_root/classname', e.g., 'datasets/combustion_dataset/chamber'
        self.root_dir_test = os.path.join(data_path, 'test') # .../chamber/test
        self.classname = classname 
        self.img_size = img_size 
        self.data_path_base = data_path # Store for CF path, e.g., .../chamber
        
        self.images = []
        good_images_path = os.path.join(self.root_dir_test, 'good')
        if os.path.isdir(good_images_path):
            self.images.extend(sorted(glob.glob(os.path.join(good_images_path, "*.png")))) # Adjust extensions if needed
        
        potential_anomaly_dirs = [d for d in os.listdir(self.root_dir_test) if os.path.isdir(os.path.join(self.root_dir_test, d)) and d != "good"]
        for anomaly_type_folder in potential_anomaly_dirs:
            anomaly_images_path = os.path.join(self.root_dir_test, anomaly_type_folder)
            self.images.extend(sorted(glob.glob(os.path.join(anomaly_images_path, "*.png"))))
            
        if not self.images:
            print(f"Warning: No images found in {self.root_dir_test} for class {self.classname}.")

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        # ... (original transform_image, ensure it handles cases where image/mask might not load)
        image = cv2.imread(image_path)
        if image is None: raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            if mask_path is not None: print(f"Warning: Mask not found at {mask_path}, using empty mask.")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        image = cv2.resize(image, dsize=(self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, dsize=(self.img_size[1], self.img_size[0]))

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0 
        mask = np.expand_dims(mask, axis=2) # (H, W, 1)

        image = np.transpose(image, (2, 0, 1)) # (C, H, W)
        mask = np.transpose(mask, (2, 0, 1))   # (1, H, W)
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        img_path = self.images[idx] # e.g., .../chamber/test/good/frame_0001.png
        
        # Determine image type (good or anomaly) and construct mask_path
        parts = img_path.split(os.sep)
        image_type_folder = parts[-2] # 'good' or anomaly type e.g. 'burn'
        file_name_only = parts[-1]

        mask_path = None
        has_anomaly_label = np.array([0], dtype=np.float32)

        if image_type_folder != 'good':
            has_anomaly_label = np.array([1], dtype=np.float32)
            mask_basename = file_name_only.split('.')[0] + ".png" # Or _mask.png depending on naming
            mask_path = os.path.join(self.data_path_base, 'ground_truth', image_type_folder, mask_basename)
            # Fallback logic from original if needed for different mask naming conventions
            if not os.path.exists(mask_path):
                 mask_path_alt = os.path.join(os.path.dirname(img_path), '../../ground_truth/', image_type_folder, file_name_only.replace(".png", "_mask.png"))
                 if os.path.exists(mask_path_alt): mask_path = mask_path_alt
                 else: mask_path = None # Will use empty mask

        image_tensor, mask_tensor = self.transform_image(img_path, mask_path)

        # --- Load Current Features ---
        npy_filename_cf = file_name_only.replace('.png', '.npy')
        current_feature_path = os.path.join(self.data_path_base, 'current_features', 'test', image_type_folder, npy_filename_cf)
        
        current_features_data = None
        if os.path.exists(current_feature_path):
            current_features_data = np.load(current_feature_path).astype(np.float32)
        else:
            print(f"Warning: Current feature file not found for CustomTest: {current_feature_path}. Using zeros.")
            # MODIFIED: Update placeholder shape
            current_features_data = np.zeros((24, 3), dtype=np.float32) # Placeholder (24, 3)

        sample_dict = {'image': image_tensor, 'has_anomaly': has_anomaly_label,
                       'mask': mask_tensor, 'idx': idx, 
                       'file_name': os.path.join(image_type_folder, file_name_only),
                       'current_features': current_features_data}
        return sample_dict
    
class CustomTrainDataset(Dataset):
    def __init__(self, data_path, classname, img_size, args):
        # data_path is 'datasets/your_custom_dataset_root/classname', e.g., 'datasets/combustion_dataset/chamber'
        self.classname = classname
        self.root_dir_train_good = os.path.join(data_path, 'train', 'good') # .../chamber/train/good
        self.img_size = img_size
        self.anomaly_source_path = args.get("anomaly_source_path", "") # Path to DTD or similar
        self.data_path_base = data_path # Store for CF path, e.g. .../chamber
        self.args = args # Store args if perlin_synthetic or other methods need them

        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir_train_good, "*.png"))) # Adjust extensions
        if not self.image_paths:
            print(f"Warning: No training images found in {self.root_dir_train_good}.")

        self.anomaly_source_paths = []
        if self.anomaly_source_path and os.path.isdir(self.anomaly_source_path):
             self.anomaly_source_paths = sorted(glob.glob(os.path.join(self.anomaly_source_path, "images", "*", "*.jpg")))
        if not self.anomaly_source_paths:
            print(f"Warning: No anomaly source images (DTD) found in {self.anomaly_source_path}.")

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           # ... (other augmenters from original if needed) ...
                           iaa.Affine(rotate=(-45, 45))]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths) if self.image_paths else 0

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), min(3, len(self.augmenters)), replace=False)
        selected_augs = [self.augmenters[i] for i in aug_ind]
        return iaa.Sequential(selected_augs)


    def perlin_synthetic(self, image_norm_hwc, original_cv2_resized_hwc, anomaly_source_path_sel):
        # Simplified Perlin synthetic anomaly generation from CustomTrainDataset original
        # image_norm_hwc: base image, normalized (0-1), HWC format
        # original_cv2_resized_hwc: original image (0-255), HWC format, for self-augmentation source
        # anomaly_source_path_sel: path to DTD image
        
        if random.random() > 0.1: # 50% chance no anomaly
            return image_norm_hwc.astype(np.float32), \
                   np.zeros((self.img_size[0], self.img_size[1], 1), dtype=np.float32), \
                   np.array([0.0], dtype=np.float32)

        # Generate Perlin mask
        perlin_scale = 6
        min_perlin_scale = 0
        mask_generated = None
        for _ in range(10): # Try generating a valid mask
            ps_x = 2 ** random.randint(min_perlin_scale, perlin_scale -1)
            ps_y = 2 ** random.randint(min_perlin_scale, perlin_scale -1)
            p_noise = rand_perlin_2d_np((self.img_size[0], self.img_size[1]), (ps_x, ps_y))
            p_noise_rot = self.rot(image=p_noise)
            p_mask_binary = np.where(p_noise_rot > 0.5, 1.0, 0.0)
            
            sum_mask = np.sum(p_mask_binary)
            min_area = 0.02 * self.img_size[0] * self.img_size[1]
            max_area = 0.7 * self.img_size[0] * self.img_size[1]
            if sum_mask > min_area and sum_mask < max_area:
                mask_generated = np.expand_dims(p_mask_binary, axis=2).astype(np.float32)
                break
        
        if mask_generated is None: # Failed to make a good mask
            return image_norm_hwc.astype(np.float32), \
                   np.zeros((self.img_size[0], self.img_size[1], 1), dtype=np.float32), \
                   np.array([0.0], dtype=np.float32)

        # Anomaly content: DTD or self-augmentation
        anomaly_content_final_norm_hwc = np.zeros_like(image_norm_hwc, dtype=np.float32)
        use_dtd_source = (self.anomaly_source_paths and anomaly_source_path_sel and random.random() > 0.5)
        
        if use_dtd_source:
            dtd_cv2_img = cv2.imread(anomaly_source_path_sel)
            if dtd_cv2_img is not None:
                dtd_rgb = cv2.cvtColor(dtd_cv2_img, cv2.COLOR_BGR2RGB)
                dtd_resized = cv2.resize(dtd_rgb, dsize=(self.img_size[1], self.img_size[0]))
                dtd_augmented = self.randAugmenter()(image=dtd_resized)
                anomaly_content_final_norm_hwc = dtd_augmented.astype(np.float32) / 255.0
        
        if not use_dtd_source or np.sum(anomaly_content_final_norm_hwc) == 0: # Fallback to self-augmentation
            self_aug_source = self.randAugmenter()(image=original_cv2_resized_hwc.copy())
            anomaly_content_final_norm_hwc = self_aug_source.astype(np.float32) / 255.0
            
        # Blend
        beta = random.uniform(0.2, 0.8)
        img_with_anomaly_texture = anomaly_content_final_norm_hwc * mask_generated
        
        augmented_final_img = image_norm_hwc * (1 - mask_generated) + \
                              (1 - beta) * img_with_anomaly_texture + \
                              beta * image_norm_hwc * mask_generated
        augmented_final_img = np.clip(augmented_final_img, 0.0, 1.0).astype(np.float32)
        
        return augmented_final_img, mask_generated, np.array([1.0], dtype=np.float32)


    def __getitem__(self, idx):
        if not self.image_paths: # Handle empty dataset
            dummy_img = np.zeros((3, self.img_size[0], self.img_size[1]), dtype=np.float32)
            dummy_mask = np.zeros((1, self.img_size[0], self.img_size[1]), dtype=np.float32)
            dummy_cf = np.zeros(3, dtype=np.float32)
            return {'image': dummy_img, "anomaly_mask": dummy_mask, 'augmented_image': dummy_img, 
                    'has_anomaly': np.array([0.0], dtype=np.float32), 'idx': -1, 'current_features': dummy_cf}

        actual_idx = random.randint(0, len(self.image_paths) - 1)
        image_path = self.image_paths[actual_idx]
        
        cv2_img_orig = cv2.imread(image_path)
        if cv2_img_orig is None: raise FileNotFoundError(f"Training image not found: {image_path}")
        
        cv2_rgb_orig = cv2.cvtColor(cv2_img_orig, cv2.COLOR_BGR2RGB)
        cv2_rgb_resized = cv2.resize(cv2_rgb_orig, dsize=(self.img_size[1], self.img_size[0]))
        
        image_normalized_hwc = cv2_rgb_resized.astype(np.float32) / 255.0

        # Original image (resized 0-255 HWC) for self-augmentation source
        original_for_self_aug_hwc = cv2.resize(cv2_img_orig, dsize=(self.img_size[1], self.img_size[0])) # BGR 0-255

        selected_dtd_path = None
        if self.anomaly_source_paths:
            selected_dtd_path = self.anomaly_source_paths[random.randint(0, len(self.anomaly_source_paths) - 1)]
        
        augmented_img_hwc, anomaly_mask_hwc, has_anomaly_val = self.perlin_synthetic(
            image_normalized_hwc.copy(), original_for_self_aug_hwc, selected_dtd_path
        )
        
        # Transpose for PyTorch: (C, H, W)
        image_torch = np.transpose(image_normalized_hwc, (2, 0, 1))
        augmented_image_torch = np.transpose(augmented_img_hwc, (2, 0, 1))
        anomaly_mask_torch = np.transpose(anomaly_mask_hwc, (2, 0, 1))

        # --- Load Current Features ---
        file_name_only_train = os.path.basename(image_path)
        npy_filename_cf_train = file_name_only_train.replace('.png', '.npy')
        current_feature_path_train = os.path.join(self.data_path_base, 'current_features', 'train', 'good', npy_filename_cf_train)
        
        current_features_data_train = None
        if os.path.exists(current_feature_path_train):
            current_features_data_train = np.load(current_feature_path_train).astype(np.float32)
        else:
            print(f"Warning: Current feature file not found for CustomTrain: {current_feature_path_train}. Using zeros.")
            # MODIFIED: Update placeholder shape
            current_features_data_train = np.zeros((24, 3), dtype=np.float32) # Placeholder (24, 3)

        sample_output = {'image': image_torch, 
                         "anomaly_mask": anomaly_mask_torch,
                         'augmented_image': augmented_image_torch, 
                         'has_anomaly': has_anomaly_val, 
                         'idx': actual_idx,
                         'current_features': current_features_data_train}
        return sample_output