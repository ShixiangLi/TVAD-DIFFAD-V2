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

texture_list = ['carpet', 'zipper', 'leather', 'tile', 'wood','grid',
                'Class1', 'Class2', 'Class3', 'Class4', 'Class5',
                 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']

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

    # Path: dataset_base_path / [classname] / current_features / phase / image_type_folder_name / npy_filename
    # Example: datasets/combustion_dataset / chamber / current_features / train / good / frame_0001.npy
    
    # Determine the root for 'current_features'. It should be parallel to 'train', 'test', 'ground_truth'.
    # If image_path is .../combustion_dataset/chamber/train/good/img.png
    # current_path is .../combustion_dataset/chamber/current_features/train/good/img.npy
    
    # Let's assume 'data_path' passed to Dataset __init__ is like 'datasets/combustion_dataset/chamber'
    # then self.root_dir is '.../chamber/train/good' or '.../chamber/test'
    
    # parts = image_path.split(os.sep)
    # For custom dataset: data_path_to_dataset = "datasets/custom_dataset_root_path/classname"
    # Current file structure processing.py saves to: OUTPUT_BASE_DIR / current_features / phase / type / file.npy
    # OUTPUT_BASE_DIR was "datasets/combustion_dataset/chamber"
    
    # Assuming `dataset_base_path_for_cf` refers to the level of "chamber" or "mvtec_root/bottle" etc.
    # Example for CustomDataset:
    # image_path: datasets/combustion_dataset/chamber/train/good/frame_0001.png
    # We need:    datasets/combustion_dataset/chamber/current_features/train/good/frame_0001.npy
    
    # Find root of 'chamber'
    # common_path_part = os.path.commonpath([image_path, os.path.join(dataset_base_path, classname_from_dataset_init)])
    # This logic might be tricky. Let's simplify by replacing parts of the image_path.

    if "/train/good/" in image_path:
        cf_path = image_path.replace("/train/good/", f"/current_features/train/good/").replace(base_filename, npy_filename)
    elif f"/test/{image_type_folder_name}/" in image_path:
        cf_path = image_path.replace(f"/test/{image_type_folder_name}/", f"/current_features/test/{image_type_folder_name}/").replace(base_filename, npy_filename)
    else: # Fallback or more robust reconstruction needed
        # This part needs to be very robust depending on the exact structure of image_path and how self.root_dir is defined
        # print(f"Warning: Could not determine current feature path for {image_path} with type {image_type_folder_name}")
        # Attempt a more general replacement based on the phase
        if phase == 'train':
            # Assuming image_path structure is .../classname/train/good/file.png
            # We want .../classname/current_features/train/good/file.npy
            path_parts = list(os.path.split(image_path)) # [head, tail=filename]
            # path_parts[0] is .../classname/train/good
            # We need to insert 'current_features' before 'train' or 'test' if 'classname' is the level above.
            # This reconstruction is highly dependent on how `data_path` is given to Dataset __init__.
            # Let's assume `image_path` is absolute or relative from a common root.
            # And `current_features` is a sibling to `train`, `test` folders inside the class specific folder.
            # e.g. MyDataset/bottle/train/good/img.png -> MyDataset/bottle/current_features/train/good/img.npy
            class_root_path = os.path.dirname(os.path.dirname(os.path.dirname(image_path))) # Up to 'bottle'
            cf_path = os.path.join(class_root_path, 'current_features', phase, image_type_folder_name, npy_filename)
            # This might be too general. The replacement method is safer if paths are consistent.
            # Fallback: if replacements above fail, this will likely fail too or be incorrect.
            # A simple, but potentially brittle solution:
            try:
                # Attempt to replace the 'train' or 'test' part of the path more generically
                # This assumes 'current_features' is a sibling to the folder containing 'train'/'test'
                # Or, 'current_features' is a sibling to 'train'/'test' *inside* the class folder.
                # Based on processing.py: it's OUTPUT_BASE_DIR / current_features / phase / type
                # OUTPUT_BASE_DIR = 'datasets/combustion_dataset/chamber'
                # So: datasets/combustion_dataset/chamber / current_features / phase / type
                # image_path: datasets/combustion_dataset/chamber / phase / type / img.png
                
                # Find "chamber" (or classname) part
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
        # Attempt an alternative common structure if the classname is part of the base path in args
        # e.g. args["custom_dataset_root_path"] = "datasets/combustion_dataset"
        # and self.classname = "chamber"
        # cf_path_alt = os.path.join(dataset_base_path, classname, 'current_features', phase, image_type_folder_name, npy_filename)
        # This requires passing dataset_base_path and classname, or making assumptions.
        # For now, we rely on the direct replacement or the reconstruction based on processing.py's saving structure.
        print(f"Warning: Current feature file not found at {cf_path} (derived from {image_path})")
        return None # File not found
        
    return cf_path


class MVTecTestDataset(Dataset):
    # ... (original MVTecTestDataset code) ...
    # If MVTec also needs current features, modify __getitem__ similarly to CustomTestDataset:
    # 1. Determine 'image_type_folder' (good or anomaly type).
    # 2. Construct current_feature_path using 'test', image_type_folder, and replacing parts of img_path.
    #    The base for MVTec would be something like:
    #    cf_path = img_path.replace(f'/test/{base_dir}/', f'/current_features/test/{base_dir}/').replace('.png','.npy')
    # 3. Load np.load(cf_path)
    # 4. Add to sample: sample['current_features'] = loaded_current_features

    def __init__(self, data_path,classname,img_size): # data_path is mvtec_root_path/classname
        self.root_dir = os.path.join(data_path,'test') # e.g. .../bottle/test
        self.images = sorted(glob.glob(self.root_dir+"/*/*.png"))
        self.resize_shape = [img_size[0], img_size[1]]
        self.data_path_base = data_path # Store for CF path construction, e.g. .../bottle

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        # ... (original transform_image code) ...
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path) # .../bottle/test/broken_large
        base_dir = os.path.basename(dir_path) # broken_large (image_type_folder)
        
        image, mask = None, None # Initialize
        has_anomaly = None

        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path_orig = os.path.join(dir_path, '../../ground_truth/') # .../bottle/ground_truth/
            mask_path_orig = os.path.join(mask_path_orig, base_dir) # .../bottle/ground_truth/broken_large
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path_final = os.path.join(mask_path_orig, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path_final)
            has_anomaly = np.array([1], dtype=np.float32)

        # --- Load Current Features ---
        # Assumes current_features are stored like: self.data_path_base/current_features/test/base_dir/file.npy
        npy_file_name = file_name.replace('.png', '.npy')
        current_feature_path = os.path.join(self.data_path_base, 'current_features', 'test', base_dir, npy_file_name)
        current_features_tensor = None
        if os.path.exists(current_feature_path):
            current_features_tensor = np.load(current_feature_path).astype(np.float32)
        else:
            print(f"Warning: Current feature file not found for MVTec test: {current_feature_path}. Using zeros.")
            current_features_tensor = np.zeros(3, dtype=np.float32) # Placeholder if not found

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx,'type':img_path[len(self.root_dir):-8],'file_name':base_dir+'_'+file_name,
                  'current_features': current_features_tensor}
        return sample

class MVTecTrainDataset(Dataset):
    # ... (original MVTecTrainDataset code) ...
    # If MVTec also needs current features, modify __getitem__ similarly to CustomTrainDataset:
    # 1. Get image_path.
    # 2. Construct current_feature_path for 'train/good'.
    #    cf_path = image_path.replace('/train/good/', '/current_features/train/good/').replace('.png','.npy')
    # 3. Load np.load(cf_path)
    # 4. Add to sample: sample['current_features'] = loaded_current_features

    def __init__(self, data_path,classname,img_size,args): # data_path is mvtec_root_path/classname
        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good') # .../bottle/train/good
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]
        self.image_paths = sorted(glob.glob(self.root_dir+"/*.png"))
        # ... (rest of __init__)
        self.data_path_base = data_path # Store for CF path construction
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        # ... (any other initializations like anomaly_source_paths, textural_foreground_path)
        self.anomaly_source_paths = sorted(glob.glob(self.anomaly_source_path+"/images/*/*.jpg"))
        if "mvtec_root_path" in args: # For textural classes foreground (seems MVTec specific)
             foreground_path_example = os.path.join(args["mvtec_root_path"],'carpet') 
             self.textural_foreground_path = sorted(glob.glob(foreground_path_example +"/thresh/*.png"))
        else:
             self.textural_foreground_path = []


    def __len__(self):
        return len(self.image_paths)
        
    # ... (helper methods like random_choice_foreground_path, get_foreground_mvtec, randAugmenter, perlin_synthetic remain the same) ...
    def random_choice_foreground_path(self):
        if not self.textural_foreground_path: return None # Handle empty
        foreground_path_id = torch.randint(0, len(self.textural_foreground_path), (1,)).item()
        return self.textural_foreground_path[foreground_path_id]

    def get_foreground_mvtec(self,image_path):
        # This logic depends on self.classname and texture_list, and self.textural_foreground_path
        # which itself depends on args["mvtec_root_path"]
        # Ensure this is robust if not MVTec.
        if self.classname in texture_list and self.textural_foreground_path:
            return self.random_choice_foreground_path()
        else: # Fallback for object classes or if textural_foreground_path is not set
            return image_path.replace('train', 'DISthresh') # Original logic

    def randAugmenter(self): # Simplified from original, can be expanded
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]], self.augmenters[aug_ind[1]], self.augmenters[aug_ind[2]]])
        return aug

    def perlin_synthetic(self, image, thresh_img, anomaly_source_path_selected, original_cv2_image, thresh_path_debug):
        # This is a complex function. The core logic for image augmentation is kept.
        # image: normalized base image (H,W,C)
        # thresh_img: normalized threshold map (H,W)
        # anomaly_source_path_selected: path to an image from DTD for texture
        # original_cv2_image: original image in cv2 format (0-255, BGR or RGB) for self-augmentation
        
        if random.random() > 0.5: # 50% chance of no anomaly
            return image.astype(np.float32), \
                   np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), \
                   np.array([0.0], dtype=np.float32)

        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** random.randint(min_perlin_scale, perlin_scale -1)
        perlin_scaley = 2 ** random.randint(min_perlin_scale, perlin_scale -1)

        has_anomaly_flag = 0
        generated_mask = None
        for _ in range(20): # Try to generate a valid mask
            perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)
            perlin_binary_mask = np.where(perlin_noise > 0.5, 1.0, 0.0)
            
            # Combine with object's threshold map (if provided and valid)
            if thresh_img is not None:
                object_perlin_mask = thresh_img * perlin_binary_mask # Element-wise mult
            else: # If no thresh_img, use perlin_binary_mask directly
                object_perlin_mask = perlin_binary_mask

            current_mask_sum = np.sum(object_perlin_mask)
            # Check if mask is reasonably sized
            if current_mask_sum > (0.01 * self.resize_shape[0] * self.resize_shape[1]) and \
               current_mask_sum < (0.8 * self.resize_shape[0] * self.resize_shape[1]):
                generated_mask = np.expand_dims(object_perlin_mask, axis=2).astype(np.float32)
                has_anomaly_flag = 1
                break
        
        if not has_anomaly_flag or generated_mask is None: # Failed to create a good mask
            return image.astype(np.float32), \
                   np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), \
                   np.array([0.0], dtype=np.float32)

        # Anomaly content generation (DTD or self-augmentation)
        anomaly_content_img = np.zeros_like(image, dtype=np.float32)
        use_dtd = (random.random() > 0.5 and anomaly_source_path_selected and os.path.exists(anomaly_source_path_selected)) \
                   or (self.classname in texture_list and anomaly_source_path_selected and os.path.exists(anomaly_source_path_selected))

        if use_dtd:
            dtd_img = cv2.imread(anomaly_source_path_selected)
            if dtd_img is not None:
                dtd_img = cv2.cvtColor(dtd_img, cv2.COLOR_BGR2RGB)
                dtd_img = cv2.resize(dtd_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
                dtd_img_aug = self.randAugmenter()(image=dtd_img)
                anomaly_content_img = dtd_img_aug.astype(np.float32) / 255.0
        
        if not use_dtd or np.sum(anomaly_content_img) == 0 : # Fallback to self-augmentation
            self_aug_img = self.randAugmenter()(image=original_cv2_image.copy()) # Augment original
            # CutPaste-like mixer (simplified from original)
            # This part can be very dataset/task specific. The original had a complex mixer.
            # For a general version, just using the augmented self_aug_img is safer.
            anomaly_content_img = self_aug_img.astype(np.float32) / 255.0

        # Blend anomaly content with base image using the generated_mask
        img_object_texture = anomaly_content_img * generated_mask # Apply mask to anomaly content
        
        beta_blend = random.uniform(0.2, 0.8) # Blending factor

        augmented_image_final = image * (1 - generated_mask) + \
                                (1 - beta_blend) * img_object_texture + \
                                beta_blend * image * generated_mask
        
        augmented_image_final = np.clip(augmented_image_final, 0.0, 1.0).astype(np.float32)
        return augmented_image_final, generated_mask, np.array([1.0], dtype=np.float32)


    def __getitem__(self, idx):
        # For training, usually pick a random image
        actual_idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[actual_idx]
        
        # Load original image (for base and for self-augmentation source)
        cv2_image_original = cv2.imread(image_path)
        if cv2_image_original is None: raise FileNotFoundError(f"Image not found: {image_path}")
        
        cv2_image_rgb = cv2.cvtColor(cv2_image_original, cv2.COLOR_BGR2RGB)
        cv2_image_resized = cv2.resize(cv2_image_rgb, dsize=(self.resize_shape[1], self.resize_shape[0]))
        
        image_normalized = cv2_image_resized.astype(np.float32) / 255.0 # For DDPM input (H,W,C)

        # Load foreground mask/threshold if applicable (original logic)
        thresh_path = self.get_foreground_mvtec(image_path) # Path to an object mask or texture prior
        thresh_img_normalized = None
        if thresh_path and os.path.exists(thresh_path):
            thresh_cv2 = cv2.imread(thresh_path, 0) # Load as grayscale
            if thresh_cv2 is not None:
                thresh_cv2_resized = cv2.resize(thresh_cv2, dsize=(self.resize_shape[1], self.resize_shape[0]))
                thresh_img_normalized = thresh_cv2_resized.astype(np.float32) / 255.0
        
        # Select DTD anomaly source
        anomaly_source_path_selected = None
        if self.anomaly_source_paths:
            anomaly_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
            anomaly_source_path_selected = self.anomaly_source_paths[anomaly_idx]

        # Perform synthetic anomaly generation
        # Pass original_cv2_image (before normalization, for self-augmentation)
        augmented_image_np, anomaly_mask_np, has_anomaly_np = self.perlin_synthetic(
            image_normalized.copy(), # Base image for augmentation
            thresh_img_normalized,    # Optional foreground mask
            anomaly_source_path_selected,
            cv2.resize(cv2_image_original, dsize=(self.resize_shape[1], self.resize_shape[0])), # Original image for self-aug source
            thresh_path # For debugging if needed
        )
        
        # Transpose for PyTorch (C, H, W)
        image_final_torch = np.transpose(image_normalized, (2, 0, 1)) 
        augmented_image_final_torch = np.transpose(augmented_image_np, (2, 0, 1))
        anomaly_mask_final_torch = np.transpose(anomaly_mask_np, (2, 0, 1))

        # --- Load Current Features ---
        # Assumes current_features stored like: self.data_path_base/current_features/train/good/file.npy
        base_filename = os.path.basename(image_path)
        npy_file_name = base_filename.replace('.png', '.npy') # Adjust for other extensions if needed
        current_feature_path = os.path.join(self.data_path_base, 'current_features', 'train', 'good', npy_file_name)
        
        current_features_tensor = None
        if os.path.exists(current_feature_path):
            current_features_tensor = np.load(current_feature_path).astype(np.float32)
        else:
            print(f"Warning: Current feature file not found for MVTec train: {current_feature_path}. Using zeros.")
            current_features_tensor = np.zeros(3, dtype=np.float32) # Placeholder

        sample = {'image': image_final_torch, 
                  "anomaly_mask": anomaly_mask_final_torch,
                  'augmented_image': augmented_image_final_torch, 
                  'has_anomaly': has_anomaly_np, 
                  'idx': actual_idx,
                  'current_features': current_features_tensor}
        return sample

# ... (VisATestDataset, VisATrainDataset, DAGMTestDataset, DAGMTrainDataset, MPDDTestDataset, MPDDTrainDataset) ...
# TODO: Apply similar current_feature loading logic to these other Dataset classes if they are intended to be used
# with the current_feature modification. The key is to:
#   1. In __init__, store necessary base paths (e.g., self.data_path_base = data_path).
#   2. In __getitem__, after determining `image_path` and its type ('good' or anomaly folder name),
#      construct the `current_feature_path`. For 'train', it's usually in 'train/good'. For 'test',
#      it's in 'test/good' or 'test/anomaly_type'.
#   3. The structure should be `self.data_path_base / current_features / phase / type / filename.npy`.
#   4. Load the .npy file and add it to the `sample` dictionary as `sample['current_features']`.
#   5. Handle cases where the .npy file might be missing (e.g., print a warning and return zeros).


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
            # Construct mask path assuming MVTec-like structure relative to self.data_path_base
            # e.g. self.data_path_base/ground_truth/image_type_folder/mask_name.png
            mask_basename = file_name_only.split('.')[0] + ".png" # Or _mask.png depending on naming
            mask_path = os.path.join(self.data_path_base, 'ground_truth', image_type_folder, mask_basename)
            # Fallback logic from original if needed for different mask naming conventions
            if not os.path.exists(mask_path):
                 mask_path_alt_mvtec = os.path.join(os.path.dirname(img_path), '../../ground_truth/', image_type_folder, file_name_only.replace(".png", "_mask.png"))
                 if os.path.exists(mask_path_alt_mvtec): mask_path = mask_path_alt_mvtec
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
        # ... (same as in MVTecTrainDataset or simplified) ...
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