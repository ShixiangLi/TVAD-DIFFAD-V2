import matplotlib.pyplot as plt
import torch
# from skimage.metrics import structural_similarity as ssim # If using SSIM
from sklearn.metrics import auc, roc_curve, average_precision_score, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix # New imports
# import time # Unused
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
# import torch.nn as nn # nn.Module not directly used here, but functional might be
from models.Recon_subnetwork import UNetModel # update_ema_params not used here
from models.Seg_subnetwork import SegmentationSubNetwork
# import torch.nn as nn # Duplicated
from data.dataset_beta_thresh import (
    MVTecTestDataset, CustomTestDataset
    # Train datasets not needed for eval.py
)
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
from math import exp # For Gaussian window if used, but not in current eval
import torch.nn.functional as F # For BinaryFocalLoss if it were defined/used here
torch.cuda.empty_cache() # Good practice at start
from tqdm import tqdm
import json
import os
from collections import defaultdict
import pandas as pd
import torchvision.utils
# import os # Duplicated
from torch.utils.data import DataLoader
from skimage.measure import label, regionprops # For pixel_pro
# import sys # Unused

# Helper for pixel_pro AUC calculation (from original)
def min_max_norm(image): # Renamed from local min_max_norm in pixel_pro for broader use if needed
    a_min, a_max = image.min(), image.max()
    if a_max - a_min == 0: # Avoid division by zero if image is constant
        return image - a_min # Should result in all zeros
    return (image - a_min) / (a_max - a_min)

def pixel_pro(gt_mask_list, pred_score_map_list): # Takes lists of masks and score maps
    # Convert inputs to numpy arrays if they are not already
    # Expects gt_mask_list to be list of [H, W] binary masks (0 or 255/1)
    # Expects pred_score_map_list to be list of [H, W] float score maps (0-1)
    
    if not isinstance(gt_mask_list, np.ndarray):
        gt_masks_np = np.array(gt_mask_list)
    else:
        gt_masks_np = gt_mask_list
        
    if not isinstance(pred_score_map_list, np.ndarray):
        pred_score_maps_np = np.array(pred_score_map_list)
    else:
        pred_score_maps_np = pred_score_map_list

    # Ensure gt_masks are boolean (True for anomaly)
    gt_masks_bool = gt_masks_np.astype(bool) # If masks are 0/255, this makes 0->False, >0->True

    max_step = 1000
    expect_fpr = 0.3  # default 30%
    
    # Global min/max thresholds from all prediction score maps
    max_th = pred_score_maps_np.max()
    min_th = pred_score_maps_np.min()

    if max_th == min_th: # Handle case where all prediction scores are the same
        # If all preds are same, PRO is either 0 or 1 depending on if that score is above/below a hypothetical threshold
        # This case often means the model isn't differentiating well. AUPRO will be low.
        # Let's return 0 or a very small value as AUPRO would be ill-defined.
        # A simple approach: if any anomaly exists and prediction is low, PRO is 0. If high, could be 1.
        # For robustness, if all scores are identical, treat as unable to segment -> PRO AUC = 0
        print("Warning: All prediction scores are identical in pixel_pro. AUPRO might be 0.")
        # To avoid NaN in AUC, if fprs_selected is empty or has one point, auc is 0 or not well-defined.
        # If max_th == min_th, delta is 0, loop might not run as expected.
        # Consider a scenario: if all preds are 0.5. Thresholds will all be 0.5. binary_score_maps fixed.
        # FPR will be fixed. PRO will be fixed. AUC of a flat line vs constant FPRs...
        return 0.0 # Or handle more gracefully


    delta = (max_th - min_th) / max_step
    
    pros_mean_at_fpr_lt_0_3 = [] # List to store PRO values where FPR <= 0.3
    fprs_for_pro_auc = []      # Corresponding FPR values (normalized to 0-1 range for AUC)

    for step in range(max_step):
        thred = max_th - step * delta
        binary_score_maps = (pred_score_maps_np > thred).astype(bool) # Pixels > thred are anomalies
        
        current_step_pros = [] # PRO values for all regions at this threshold
        
        for i in range(len(binary_score_maps)): # Iterate over each image in the batch/dataset
            gt_mask_single_image = gt_masks_bool[i]
            pred_mask_single_image = binary_score_maps[i]

            if not gt_mask_single_image.any(): # Skip image if no ground truth anomaly
                continue

            labeled_gt, num_gt_regions = label(gt_mask_single_image, connectivity=2, return_num=True)
            if num_gt_regions == 0:
                continue

            props = regionprops(labeled_gt)
            for prop in props: # Iterate over each connected anomalous region in GT
                # Get pixels for this specific GT region
                gt_region_mask = (labeled_gt == prop.label) # Mask for current GT component
                
                # Calculate intersection of this GT region with the overall predicted anomaly map for this image
                intersection = np.logical_and(gt_region_mask, pred_mask_single_image).sum()
                pro = intersection / prop.area # prop.area is the size of the GT region
                current_step_pros.append(pro)
        
        # Calculate FPR for the current threshold across all images
        gt_masks_neg = ~gt_masks_bool # Negative (normal) pixels in GT
        fp = np.logical_and(gt_masks_neg, binary_score_maps).sum()
        
        # Handle division by zero if gt_masks_neg.sum() is 0 (i.e., all pixels are anomalous in GT)
        # This is unlikely for typical anomaly detection datasets but possible.
        if gt_masks_neg.sum() == 0: 
            fpr = 1.0 if fp > 0 else 0.0 # If all GT is anomaly, any FP prediction means FPR is effectively 1
        else:
            fpr = fp / gt_masks_neg.sum()

        if fpr <= expect_fpr:
            if current_step_pros: # Only add if there were actual anomaly regions to calculate PRO for
                pros_mean_at_fpr_lt_0_3.append(np.mean(current_step_pros))
            else: # If no GT anomalies in any image, PRO is undefined. Could append 0 or skip.
                  # For AUPRO, if no anomalies, it's often considered perfect (1) or undefined.
                  # If current_step_pros is empty because all images had no anomalies, it's tricky.
                  # Let's assume if gt_masks_bool.any() is false overall, pixel_pro isn't called or returns 1.
                  # Here, if current_step_pros is empty due to filtering, it implies some images had anomalies but
                  # perhaps were skipped, or no regions found. This shouldn't happen if gt_masks_bool.any() is true.
                  # If a step has valid FPR but no regions (e.g. all GT anomaly regions are tiny and missed by props),
                  # then PRO for that step is effectively 0 if we consider uncalculable PROs as 0.
                pros_mean_at_fpr_lt_0_3.append(0.0) 
            fprs_for_pro_auc.append(fpr)

    if not pros_mean_at_fpr_lt_0_3: # No points found where FPR <= 0.3
        return 0.0

    # Sort by FPR to ensure correct AUC calculation
    sorted_indices = np.argsort(fprs_for_pro_auc)
    fprs_selected_sorted = np.array(fprs_for_pro_auc)[sorted_indices]
    pros_mean_selected_sorted = np.array(pros_mean_at_fpr_lt_0_3)[sorted_indices]

    # Normalize selected FPRs from [0, current_max_fpr_under_0.3] to [0, 1] for AUC
    # This is critical for the interpretation of AUPRO.
    # The original PatchCore AUPRO normalizes the FPR axis from 0 to expect_fpr (0.3).
    if fprs_selected_sorted.max() == fprs_selected_sorted.min(): # All selected FPRs are the same (e.g. all 0)
        # If all selected FPRs are 0, and there's some PRO, AUC is not well-defined by integrating over a point.
        # It could be seen as average PRO at FPR=0.
        # Or, if all FPRs are same non-zero, similar issue.
        # A common robust handling: if the range of x for AUC is zero, AUC is 0.
        return 0.0
    
    # Normalize FPRs to the range [0, 1] based on the actual range of selected FPRs, bounded by expect_fpr.
    # The standard AUPRO integrates from FPR=0 to FPR=expect_fpr.
    # We should ensure our fprs_for_pro_auc contains points up to expect_fpr if possible,
    # or handle the normalization carefully.
    # Let's normalize fprs_selected_sorted to its own range [min_selected_fpr, max_selected_fpr] -> [0,1]
    # This is what `min_max_norm` would do for `fprs_selected_sorted`.
    # However, for AUPRO, the integration is typically over a fixed FPR range (e.g. 0 to 0.3).
    # So, we should scale `fprs_selected_sorted` by `1/expect_fpr`.
    
    fprs_normalized_for_auc = fprs_selected_sorted / expect_fpr
    # Ensure it's clipped to [0,1] after division, in case some fprs were slightly > expect_fpr due to float precision
    fprs_normalized_for_auc = np.clip(fprs_normalized_for_auc, 0, 1)


    # Add (0,0) and (1, value_at_expect_fpr) or (1, last_pro_value) to make AUC well-defined
    # This depends on how the AUC should behave at boundaries.
    # If fprs_normalized_for_auc doesn't start at 0, prepend (0, first_pro_value) or (0,0)
    # If fprs_normalized_for_auc doesn't end at 1, append (1, last_pro_value)
    
    # A simpler way: ifauc requires x to be monotonic.
    # We have sorted (fprs_normalized_for_auc, pros_mean_selected_sorted)
    # Ensure unique FPR points for AUC if multiple PROs exist for same FPR (shouldn't happen with sorted unique FPRs from np.array(fprs_for_pro_auc))
    unique_fprs, unique_indices = np.unique(fprs_normalized_for_auc, return_index=True)
    unique_pros = pros_mean_selected_sorted[unique_indices]

    if len(unique_fprs) < 2: # Need at least two points for AUC
        return 0.0
        
    # Ensure (0, X) and (Y, Z) where Y can be up to 1.
    # If 0 is not in unique_fprs, add it. Assume PRO at FPR=0 is the PRO of the highest threshold.
    # If unique_fprs[0] > 0: 
        # pro_at_fpr_0 = unique_pros[0] # Or a more sophisticated extrapolation/assumption
        # unique_fprs = np.insert(unique_fprs, 0, 0)
        # unique_pros = np.insert(unique_pros, 0, pro_at_fpr_0) # Or 0 if being conservative

    # Similarly for the end point if max unique_fpr < 1
    # if unique_fprs[-1] < 1.0 and expect_fpr was reached by some original fprs:
        # pro_at_fpr_1 = unique_pros[-1]
        # unique_fprs = np.append(unique_fprs, 1.0)
        # unique_pros = np.append(unique_pros, pro_at_fpr_1)


    seg_pro_auc = auc(unique_fprs, unique_pros)
    return seg_pro_auc


def gridify_output(img, row_size=-1): # Seems unused in this eval.py, but harmless
    scale_img = lambda img_tensor: ((img_tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # Assuming img is a batch of images (B, C, H, W)
    # make_grid expects (B, C, H, W)
    grid = torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=128) # pad_value changed from -1
    # Permute grid from (C, H_grid, W_grid) to (H_grid, W_grid, C) for plt.imshow
    return grid.cpu().permute(1, 2, 0)


def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd


def load_checkpoint(args_config, device, sub_class_name, checkpoint_type): # Renamed args to args_config to avoid conflict
    # args_config is the loaded args dictionary
    param_arg_num = args_config['arg_num'] # From loaded args, not filename here
    
    # Path construction using output_path from args_config
    ck_path = os.path.join(
        args_config["output_path"], 
        "model", 
        f"diff-params-ARGS={param_arg_num}", 
        sub_class_name, 
        f"params-{checkpoint_type}.pt"
    )
    print(f"Loading checkpoint from: {ck_path}")
    if not os.path.exists(ck_path):
        print(f"Error: Checkpoint file not found at {ck_path}")
        return None
        
    try:
        # weights_only=True is safer if only loading weights and not arbitrary code
        # However, if args are saved in checkpoint and needed, False is required.
        # Original used False, so keeping it.
        loaded_model_data = torch.load(ck_path, map_location=device, weights_only=False) 
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
          
    return loaded_model_data


def load_parameters_and_checkpoint(device, sub_class_name, checkpoint_type, args_filename="args1.json"):
    # Load args from the JSON file
    args_file_path = os.path.join('./args', args_filename)
    if not os.path.exists(args_file_path):
        args_file_path = os.path.join(os.path.dirname(__file__), 'args', args_filename)
        if not os.path.exists(args_file_path):
            args_file_path_alt = os.path.join(os.getcwd(), 'args', args_filename)
            if os.path.exists(args_file_path_alt):
                args_file_path = args_file_path_alt
            else:
                print(f"Error: Args file {args_filename} not found.")
                return None, None

    with open(args_file_path, 'r') as f:
        args_loaded = json.load(f)
    
    args_config = defaultdict_from_json(args_loaded)
    args_config['arg_num'] = args_filename.split('.')[0].replace('args', '') # e.g., '1' from 'args1.json'

    # Load checkpoint based on these args
    checkpoint_data = load_checkpoint(args_config, device, sub_class_name, checkpoint_type)
 
    return args_config, checkpoint_data


# Gaussian blur for heatmap (from original)
# def gaussian(window_size, sigma): # Unused if using scipy.ndimage.gaussian_filter
#     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()

def image_transform_back_to_255(image_tensor_chw): # Expects (C,H,W) tensor normalized -1 to 1 or 0 to 1
    # If image is already (0,1) from model output (e.g. pred_x_0.clamp(-1,1) then (x+1)/2 for (0,1) range)
    # Or if model outputs (0,1) directly
    # This function assumes input is 0-1 range from the original code's usage context.
    # Original code: `image_transform(pred_x_0_condition.detach().cpu().numpy()[0])`
    # where pred_x_0_condition is clamped to [-1,1].
    # So, the input numpy array to original `image_transform` was [-1,1].
    
    # Let's assume input `image_tensor_chw` is a NumPy array (C,H,W) in range [-1,1] or [0,1]
    # The original `image_transform` was: `np.clip(image* 255, 0, 255).astype(np.uint8)`
    # This implies `image` was already scaled, or it was implicitly handling [-1,1] -> [0,255] by error,
    # or it expected [0,1] input.
    
    # If input is [-1, 1], first map to [0, 1]
    if image_tensor_chw.min() < -0.001: # Heuristic for [-1, 1] range
        img_0_1 = (image_tensor_chw + 1.0) / 2.0
    else: # Assumed to be [0, 1]
        img_0_1 = image_tensor_chw
        
    img_0_255 = np.clip(img_0_1 * 255.0, 0, 255).astype(np.uint8)
    return img_0_255


# BinaryFocalLoss - not typically used in eval.py, but harmless if it stays
# class BinaryFocalLoss(nn.Module): ... (already in train.py)

def cvt2heatmap(gray_img_0_255): # Expects single channel image in 0-255 range
    heatmap_bgr = cv2.applyColorMap(np.uint8(gray_img_0_255), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB for matplotlib

def show_cam_on_image(rgb_img_0_255, anomaly_map_0_1_normalized, weight=0.6): # rgb_img (H,W,C), anomaly_map (H,W)
    # Ensure anomaly_map is (H,W) and scaled 0-255 for cvt2heatmap
    heatmap_rgb = cvt2heatmap(anomaly_map_0_1_normalized * 255.0)
    
    # Blend image with heatmap: dst = src1*alpha + src2*beta + gamma
    # Here, rgb_img_0_255 is uint8, heatmap_rgb is uint8.
    # For blending, they should ideally be float and then converted back.
    blended_img = cv2.addWeighted(rgb_img_0_255.astype(np.uint8), 1-weight, heatmap_rgb.astype(np.uint8), weight, 0)
    return blended_img


# min_max_norm already defined globally

# heatmap plotting function (from original) - simplified for clarity
def save_visualizations(image_path_str, raw_image_orig_chw_01, gt_mask_chw_01, out_mask_chw_01,
                        pred_x_0_condition_chw_01, args_config, sub_class_name, checkpoint_type, image_score_val,
                        x_normal_t_chw_01=None, x_noiser_t_chw_01=None, pred_x_t_noisier_chw_01=None,
                        pred_x_0_normal_chw_01=None, pred_x_0_noisier_chw_01=None):

    viz_path_base = os.path.join(
        args_config["output_path"],
        "metrics",
        f"ARGS={args_config['arg_num']}",
        sub_class_name)
    viz_path_specific = os.path.join(viz_path_base,
        f"visualization_{args_config['eval_normal_t']}_{args_config['eval_noisier_t']}_{args_config['condition_w']}condition_{checkpoint_type}ck"
    )
    os.makedirs(viz_path_specific, exist_ok=True)

    # --- Prepare all display images (convert CHW to HWC, scale to 0-255 uint8) ---
    def prep_display_img(img_data_chw, is_mask=False): # Added is_mask for specific handling if needed
        if img_data_chw is None:
            ph_size = args_config.get("img_size", [64, 64])
            placeholder = np.zeros((ph_size[0], ph_size[1], 3), dtype=np.uint8)
            # Slightly smaller font for "N/A" if it helps
            cv2.putText(placeholder, "N/A", (int(ph_size[1]*0.2), int(ph_size[0]*0.55)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            return placeholder
        
        processed_img_data = image_transform_back_to_255(img_data_chw) # Returns (C,H,W) uint8

        if processed_img_data.shape[0] == 1: # Grayscale C,H,W -> H,W
            img_hw = processed_img_data[0]
            return img_hw
        else: # RGB C,H,W -> H,W,C
            img_hwc = processed_img_data.transpose(1,2,0)
            return img_hwc

    raw_image_disp_hwc = prep_display_img(raw_image_orig_chw_01)
    gt_mask_disp_hw = prep_display_img(gt_mask_chw_01, is_mask=True)

    if out_mask_chw_01 is not None:
        anomaly_map_raw_hw = out_mask_chw_01[0]
        anomaly_map_smoothed_hw = gaussian_filter(anomaly_map_raw_hw, sigma=args_config.get("viz_gaussian_sigma", 4))
        anomaly_map_normalized_hw = min_max_norm(anomaly_map_smoothed_hw)

        raw_for_blend_hwc = raw_image_disp_hwc.copy()
        if raw_for_blend_hwc.ndim == 2:
            raw_for_blend_hwc = cv2.cvtColor(raw_for_blend_hwc, cv2.COLOR_GRAY2RGB)
        heatmap_on_image_disp_hwc = show_cam_on_image(raw_for_blend_hwc, anomaly_map_normalized_hw)
        out_mask_disp_hw = (anomaly_map_normalized_hw * 255.0).astype(np.uint8)
    else:
        heatmap_on_image_disp_hwc = prep_display_img(None)
        ph_size_hw = args_config.get("img_size", [64, 64])
        out_mask_disp_hw = prep_display_img(np.zeros((1, ph_size_hw[0], ph_size_hw[1])))[0]


    x_normal_t_disp_hwc = prep_display_img(x_normal_t_chw_01)
    x_noiser_t_disp_hwc = prep_display_img(x_noiser_t_chw_01)
    pred_x_t_noisier_disp_hwc = prep_display_img(pred_x_t_noisier_chw_01)
    recon_normal_disp_hwc = prep_display_img(pred_x_0_normal_chw_01)
    recon_noisier_disp_hwc = prep_display_img(pred_x_0_noisier_chw_01)
    recon_con_disp_hwc = prep_display_img(pred_x_0_condition_chw_01)

    plot_titles_row1 = ["Input", "GT", "x_normal_t", "x_noiser_t", "pred_x_t_noisier"]
    plot_data_row1 = [
        raw_image_disp_hwc, gt_mask_disp_hw, x_normal_t_disp_hwc,
        x_noiser_t_disp_hwc, pred_x_t_noisier_disp_hwc
    ]
    plot_cmaps_row1 = [None, "gray", None, None, None]

    plot_titles_row2 = ["heatmap", "out_mask", "recon_normal", "recon_noisier", "recon_con"]
    plot_data_row2 = [
        heatmap_on_image_disp_hwc, out_mask_disp_hw, recon_normal_disp_hwc,
        recon_noisier_disp_hwc, recon_con_disp_hwc
    ]
    plot_cmaps_row2 = [None, "gray", None, None, None]

    # --- Modified fig creation and layout ---
    fig, axes = plt.subplots(2, 5, figsize=(15, 7.2), constrained_layout=True) # Increased height, enabled constrained_layout
    fig.suptitle(f'Class: {sub_class_name} - Img: {os.path.basename(image_path_str)} - Score: {image_score_val:.4f}', fontsize=10)

    title_fontsize = 7 # Slightly reduced title font size

    for i in range(5):
        axes[0, i].imshow(plot_data_row1[i], cmap=plot_cmaps_row1[i])
        axes[0, i].set_title(plot_titles_row1[i], fontsize=title_fontsize)
        axes[0, i].axis('off')

    for i in range(5):
        axes[1, i].imshow(plot_data_row2[i], cmap=plot_cmaps_row2[i])
        axes[1, i].set_title(plot_titles_row2[i], fontsize=title_fontsize)
        axes[1, i].axis('off')

    # fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # constrained_layout usually replaces tight_layout
                                             # If you still need to adjust overall rect with constrained_layout,
                                             # it's possible but often not necessary.
                                             # For suptitle space, constrained_layout should handle it.
                                             # If suptitle is still an issue, you might use fig.set_constrained_layout_pads

    base_savename = os.path.basename(image_path_str)
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        base_savename = base_savename.replace(ext, '')
    savename_full = os.path.join(viz_path_specific, f"{sub_class_name}_{base_savename}_viz.png")

    plt.savefig(savename_full)
    plt.close(fig)


def testing(testing_dataset_loader, args_config, unet_model, seg_model, sub_class_name, class_type_str, checkpoint_type, device):
    # args_config is the loaded configuration dictionary
    
    normal_t_eval = args_config["eval_normal_t"]
    noisier_t_eval = args_config["eval_noisier_t"]
    
    # Setup DDPM sampler
    in_channels_ddpm = args_config["channels"]
    betas_ddpm = get_beta_schedule(args_config['T'], args_config['beta_schedule'])
    ddpm_sampler = GaussianDiffusionModel(
        args_config['img_size'], betas_ddpm, 
        loss_weight=args_config.get('loss_weight', 'none'),
        loss_type=args_config['loss-type'], 
        noise=args_config["noise_fn"], 
        img_channels=in_channels_ddpm
    )
    
    # Lists to store predictions and ground truths
    all_image_scores = []
    all_image_labels = []
    all_pixel_gts_list = [] # List of 2D GT masks
    all_pixel_preds_list = [] # List of 2D predicted score maps
    
    unet_model.eval()
    seg_model.eval()
    
    with torch.no_grad():
        tbar_test = tqdm(testing_dataset_loader, desc=f"Testing {sub_class_name}", leave=False)
        for i, sample in enumerate(tbar_test):
            image_tensor = sample["image"].to(device) # (B, C, H, W), assumed normalized
            image_label_gt = sample['has_anomaly'].to(device) # (B, 1) or (B,)
            pixel_mask_gt = sample["mask"].to(device) # (B, 1, H, W)
            image_path_info = sample.get("file_name", f"unknown_image_{i}") # Get filepath for saving viz
            if isinstance(image_path_info, list): image_path_info = image_path_info[0]


            # --- Current Features (for testing) ---
            if 'current_features' in sample and sample['current_features'] is not None:
                 current_features = sample['current_features'].to(device)
            else:
                 # MODIFIED: Update placeholder shape
                 print("Warning: 'current_features' not found in test sample. Using dummy tensor.")
                 current_features = torch.randn(image_tensor.shape[0], 24, 3, device=device) 
            # --- End Current Features ---

            # Prepare timesteps for evaluation
            batch_size_current = image_tensor.shape[0]
            normal_t_batch = torch.tensor([normal_t_eval], device=device).repeat(batch_size_current)
            noisier_t_batch = torch.tensor([noisier_t_eval], device=device).repeat(batch_size_current)
            
            # DDPM inference to get reconstructed image
            eval_loss, pred_x_0_cond, pred_x_0_norm, pred_x_0_nois, \
            x_norm_t, x_nois_t, pred_x_t_nois = ddpm_sampler.norm_guided_one_step_denoising_eval(
                unet_model, image_tensor, normal_t_batch, noisier_t_batch, args_config, current_features # Pass current_features
            )
            
            # Segmentation model inference
            seg_input_tensor = torch.cat((image_tensor, pred_x_0_cond), dim=1)
            pred_anomaly_map = seg_model(seg_input_tensor) # (B, 1, H, W)

            # --- Metric Calculation ---
            # Image-level score (e.g., mean of top-k pixels in anomaly map)
            flat_pred_map = pred_anomaly_map.view(batch_size_current, -1)
            k_top = args_config.get("image_score_top_k", 50)
            if flat_pred_map.shape[1] >= k_top:
                topk_scores, _ = torch.topk(flat_pred_map, k_top, dim=1)
                current_image_score = torch.mean(topk_scores, dim=1)
            else:
                current_image_score = torch.mean(flat_pred_map, dim=1)
            
            all_image_scores.extend(current_image_score.cpu().numpy())
            all_image_labels.extend(image_label_gt.cpu().numpy().flatten().tolist()) # 确保标签是扁平的列表 # Ensure labels are flat

            # Pixel-level: store flattened GT masks and predicted anomaly maps
            # These are added to lists for later processing by pixel_pro
            for b_idx in range(batch_size_current):
                all_pixel_gts_list.append(pixel_mask_gt[b_idx, 0].cpu().numpy()) # (H, W)
                all_pixel_preds_list.append(pred_anomaly_map[b_idx, 0].cpu().numpy()) # (H, W)

            # Save visualizations (optional, can be controlled by an arg)
            if args_config.get("save_visualizations_eval", True) and i < args_config.get("num_viz_to_save", 5): # Save for first few samples
                save_visualizations(
                    image_path_info, 
                    image_tensor[0].cpu().numpy(), # Assuming batch size 1 for detailed viz, or take first of batch
                    pixel_mask_gt[0].cpu().numpy(),
                    pred_anomaly_map[0].cpu().numpy(),
                    pred_x_0_cond[0].cpu().numpy(),
                    args_config, sub_class_name, checkpoint_type, current_image_score[0].item(),
                    x_normal_t_chw_01=x_norm_t[0].cpu().numpy() if x_norm_t is not None else None,
                    x_noiser_t_chw_01=x_nois_t[0].cpu().numpy() if x_nois_t is not None else None,
                    pred_x_t_noisier_chw_01=pred_x_t_nois[0].cpu().numpy() if pred_x_t_nois is not None else None,
                    pred_x_0_normal_chw_01=pred_x_0_norm[0].cpu().numpy() if pred_x_0_norm is not None else None,
                    pred_x_0_noisier_chw_01=pred_x_0_nois[0].cpu().numpy() if pred_x_0_nois is not None else None
                )

    # --- Calculate final metrics ---
    all_image_scores_np = np.array(all_image_scores)
    all_image_labels_np = np.array(all_image_labels)
    
    # Initialize metrics
    auroc_image = 0.0
    accuracy_image = 0.0
    f1_image = 0.0
    fdr_image = 0.0
    mdr_image = 0.0
    auroc_pixel = 0.0
    ap_pixel = 0.0
    aupro_pixel = 0.0
    miou_pixel = 0.0

    if len(np.unique(all_image_labels_np)) > 1:
        auroc_image = roc_auc_score(all_image_labels_np, all_image_scores_np) * 100
        print(f"Image AUROC for {sub_class_name}: {auroc_image:.2f}%")

        # Determine optimal threshold for image-level metrics from ROC curve
        fpr_img, tpr_img, thresholds_img = roc_curve(all_image_labels_np, all_image_scores_np)
        optimal_idx_img = np.argmax(tpr_img - fpr_img)
        optimal_threshold_img = thresholds_img[optimal_idx_img]
        print(f"Optimal Image Threshold for Acc/F1/FDR/MDR: {optimal_threshold_img:.4f}")
        
        binary_predictions_img = (all_image_scores_np >= optimal_threshold_img).astype(int)
        
        accuracy_image = accuracy_score(all_image_labels_np, binary_predictions_img) * 100
        f1_image = f1_score(all_image_labels_np, binary_predictions_img) * 100
        
        # Confusion matrix to calculate FDR and MDR
        cm_img = confusion_matrix(all_image_labels_np, binary_predictions_img)
        if cm_img.size == 1: # Only one class predicted or present
            tn_img, fp_img, fn_img, tp_img = 0,0,0,0
            if all_image_labels_np[0] == 0 and binary_predictions_img[0] == 0: tn_img = len(all_image_labels_np)
            elif all_image_labels_np[0] == 1 and binary_predictions_img[0] == 1: tp_img = len(all_image_labels_np)
            # Add other cases if needed, or rely on sklearn to handle simple cases (though it might still error on ravel())
            # This is a simplified handling for single class result in CM.
        elif cm_img.size == 4 : # Standard 2x2 case
            tn_img, fp_img, fn_img, tp_img = cm_img.ravel()
        else: # Non-standard CM, default to 0 for safety, or log error
            tn_img, fp_img, fn_img, tp_img = 0,0,0,0
            print(f"Warning: Image confusion matrix has unexpected shape: {cm_img.shape}")


        if (fp_img + tp_img) > 0:
            fdr_image = (fp_img / (fp_img + tp_img)) * 100
        else:
            fdr_image = 0.0 # Or np.nan if you prefer to indicate undefined

        if (fn_img + tp_img) > 0:
            mdr_image = (fn_img / (fn_img + tp_img)) * 100
        else:
            mdr_image = 0.0 # Or np.nan

        print(f"Image Accuracy for {sub_class_name}: {accuracy_image:.2f}%")
        print(f"Image F1-score for {sub_class_name}: {f1_image:.2f}%")
        print(f"Image FDR for {sub_class_name}: {fdr_image:.2f}%")
        print(f"Image MDR for {sub_class_name}: {mdr_image:.2f}%")

    else:
        print(f"Image AUROC/Acc/F1/FDR/MDR for {sub_class_name}: Not defined (single class in GT labels)")

    # Pixel AUROC & AUPRO (using the collected lists for pixel_pro)
    if all_pixel_gts_list and all_pixel_preds_list:
        flat_pixel_gts_np = np.concatenate([mask.flatten() for mask in all_pixel_gts_list]).astype(int)
        flat_pixel_preds_np = np.concatenate([smap.flatten() for smap in all_pixel_preds_list])

        if len(np.unique(flat_pixel_gts_np)) > 1:
            auroc_pixel = roc_auc_score(flat_pixel_gts_np, flat_pixel_preds_np) * 100
            ap_pixel = average_precision_score(flat_pixel_gts_np, flat_pixel_preds_np) * 100
            print(f"Pixel AUROC for {sub_class_name}: {auroc_pixel:.2f}%")
            print(f"Pixel AP for {sub_class_name}: {ap_pixel:.2f}%")

            # Pixel mIoU
            fpr_px, tpr_px, thresholds_px = roc_curve(flat_pixel_gts_np, flat_pixel_preds_np)
            optimal_idx_px = np.argmax(tpr_px - fpr_px)
            optimal_threshold_px = thresholds_px[optimal_idx_px]
            print(f"Optimal Pixel Threshold for mIoU: {optimal_threshold_px:.4f}")

            binary_pixel_preds_np = (flat_pixel_preds_np >= optimal_threshold_px).astype(int)
            
            cm_pixel = confusion_matrix(flat_pixel_gts_np, binary_pixel_preds_np)
            if cm_pixel.size == 1:
                tn_px, fp_px, fn_px, tp_px = 0,0,0,0
                if flat_pixel_gts_np[0] == 0 and binary_pixel_preds_np[0] == 0: tn_px = len(flat_pixel_gts_np)
                elif flat_pixel_gts_np[0] == 1 and binary_pixel_preds_np[0] == 1: tp_px = len(flat_pixel_gts_np)
            elif cm_pixel.size == 4:
                tn_px, fp_px, fn_px, tp_px = cm_pixel.ravel()
            else:
                tn_px, fp_px, fn_px, tp_px = 0,0,0,0
                print(f"Warning: Pixel confusion matrix has unexpected shape: {cm_pixel.shape}")


            iou_anomaly_class = 0.0
            if (tp_px + fp_px + fn_px) > 0:
                iou_anomaly_class = tp_px / (tp_px + fp_px + fn_px)
            
            iou_normal_class = 0.0
            if (tn_px + fn_px + fp_px) > 0: # Denominator for normal IoU: TN / (TN + FN + FP)
                iou_normal_class = tn_px / (tn_px + fn_px + fp_px)
            
            miou_pixel = ((iou_anomaly_class + iou_normal_class) / 2) * 100
            print(f"Pixel mIoU for {sub_class_name}: {miou_pixel:.2f}%")

        else:
            print(f"Pixel AUROC/AP/mIoU for {sub_class_name}: Not defined (single class in pixel GT)")

        # AUPRO calculation (original)
        aupro_pixel = pixel_pro(np.array(all_pixel_gts_list), np.array(all_pixel_preds_list)) * 100
        print(f"Pixel AUPRO for {sub_class_name}: {aupro_pixel:.2f}%")
    else:
        print("Pixel metrics skipped: No pixel ground truth or predictions available.")

    # Save metrics to CSV
    metrics_summary_dir = os.path.join(args_config["output_path"], "metrics", f"ARGS={args_config['arg_num']}")
    os.makedirs(metrics_summary_dir, exist_ok=True)
    csv_path = os.path.join(metrics_summary_dir, f"{normal_t_eval}_{noisier_t_eval}t_{class_type_str}_{args_config['condition_w']}condition_{checkpoint_type}ck.csv")
    
    metrics_data = {
        "classname": [sub_class_name], 
        "Image-AUROC": [round(auroc_image, 2)],
        "Image-Accuracy": [round(accuracy_image, 2)],
        "Image-F1": [round(f1_image, 2)],
        "Image-FDR": [round(fdr_image, 2)],
        "Image-MDR": [round(mdr_image, 2)],
        "Pixel-AUROC": [round(auroc_pixel, 2)], 
        "Pixel-mIoU": [round(miou_pixel, 2)],
        "Pixel-AUPRO": [round(aupro_pixel, 2)], # Keeping AUPRO as it was there
        "Pixel-AP": [round(ap_pixel, 2)] # Keeping AP as it was there
    }
    df_metrics = pd.DataFrame(metrics_data)
    
    file_exists = os.path.isfile(csv_path)
    df_metrics.to_csv(csv_path, mode='a', header=not file_exists, index=False)
    print(f"Metrics for {sub_class_name} saved to {csv_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define dataset classes and paths (can be loaded from args or defined here)
    # These are examples; ensure they match your `args1.json` and project structure.
    mvtec_classes_default = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    custom_dataset_classes_default = ['chamber'] # From original eval.py
    
    # Determine which set of classes to evaluate
    # This should be configurable, e.g., via command-line args or a main config
    # For now, using the 'custom_dataset_classes' as per original eval.py main section
    # classes_to_evaluate = custom_dataset_classes_default
    # checkpoint_type_to_load = 'best' # Or 'last', make configurable

    # --- Configuration Loading ---
    # Allow specifying args file and checkpoint type via environment or simple cmd args if needed later.
    # For now, hardcoding them for simplicity.
    args_filename_to_use = os.environ.get("ARGS_FILE", "args1.json")
    checkpoint_type_to_load = os.environ.get("CHECKPOINT_TYPE", "best") # 'best' or 'last'
    
    # Determine which classes to run based on args or a default
    # Example: load args, then see if a specific class list is specified.
    # temp_args_for_class_list, _ = load_parameters_and_checkpoint(device, "dummy", "dummy", args_filename_to_use) # Load args to get class list
    # classes_to_evaluate = temp_args_for_class_list.get('custom_dataset_classes', custom_dataset_classes_default) # Prioritize from args
    
    # Simpler: directly use the default or let args define it later.
    # For now, this loop will iterate through classes decided by the user (e.g. custom_dataset_classes)
    # The actual args file loading happens inside the loop per class to get specific settings if they varied.
    # However, usually, one args file is used for a run.
    
    # Let's load one args file for the entire evaluation session:
    args_config_session, _ = load_parameters_and_checkpoint(device, "chamber", "best", args_filename_to_use)
    if args_config_session is None:
        print("Failed to load base arguments. Exiting.")
        return
        
    classes_to_evaluate = args_config_session.get('custom_dataset_classes', custom_dataset_classes_default)
    # --- End Configuration Loading ---


    for sub_class_item in classes_to_evaluate:
        print(f"\n--- Evaluating Class: {sub_class_item} ---")
        
        # Load parameters and the specific checkpoint for this class
        # Assuming args file remains the same, but checkpoint is per sub_class
        args_config, checkpoint_data = load_parameters_and_checkpoint(device, sub_class_item, checkpoint_type_to_load, args_filename_to_use)

        if args_config is None or checkpoint_data is None:
            print(f"Could not load args or checkpoint for {sub_class_item}. Skipping.")
            continue
        
        print(f"Args successfully loaded: ARGS={args_config['arg_num']}")
        print(f"Checkpoint epoch: {checkpoint_data.get('n_epoch', 'N/A')}")
        
        in_channels_model = args_config["channels"]

        # Initialize UNet model
        unet_model_eval = UNetModel(
            img_size=args_config['img_size'], 
            base_channels=args_config['base_channels'], 
            channel_mults=args_config.get('channel_mults', ""),
            dropout=args_config["dropout"], 
            n_heads=args_config["num_heads"], 
            n_head_channels=args_config["num_head_channels"],
            attention_resolutions=args_config["attention_resolutions"],
            in_channels=in_channels_model
        ).to(device)

        # Initialize Segmentation model
        seg_model_eval = SegmentationSubNetwork(
            in_channels=in_channels_model * 2, # Expects concatenated original and reconstructed
            out_channels=1 # Outputs a single-channel anomaly map
        ).to(device)

        # Load state dicts from checkpoint
        try:
            unet_model_eval.load_state_dict(checkpoint_data["unet_model_state_dict"])
            seg_model_eval.load_state_dict(checkpoint_data["seg_model_state_dict"])
        except KeyError as e:
            print(f"Error: Missing key in checkpoint_data for {sub_class_item}: {e}. Ensure checkpoint is valid.")
            continue
        except Exception as e:
            print(f"Error loading model state_dict for {sub_class_item}: {e}")
            continue
            
        unet_model_eval.eval() # Set to evaluation mode
        seg_model_eval.eval()

        # Determine dataset type and path
        dataset_root_main = "" # Base path for the dataset type (e.g., MVTec root, VisA root)
        TestDS_Class = None
        class_type_name = "Unknown"

        # Update these paths from your args_config
        mvtec_root = args_config.get("mvtec_root_path", "path/to/mvtec")
        custom_root = args_config.get("custom_dataset_root_path", "path/to/custom")


        if sub_class_item in args_config.get('mvtec_classes', mvtec_classes_default):
            dataset_root_main = os.path.join(mvtec_root, sub_class_item)
            TestDS_Class = MVTecTestDataset
            class_type_name = 'MVTec'
        elif sub_class_item in args_config.get('custom_dataset_classes', custom_dataset_classes_default):
            dataset_root_main = os.path.join(custom_root, sub_class_item)
            TestDS_Class = CustomTestDataset
            class_type_name = 'Custom'
        else:
            print(f"Class {sub_class_item} not found in known dataset lists or paths not defined. Skipping.")
            continue
        
        if not os.path.exists(dataset_root_main):
            print(f"Dataset path not found for {sub_class_item}: {dataset_root_main}. Skipping.")
            continue

        testing_dataset_current = TestDS_Class(
            dataset_root_main, sub_class_item, img_size=args_config["img_size"]
        )
        if len(testing_dataset_current) == 0:
            print(f"No test data found for {sub_class_item} at {dataset_root_main}. Skipping.")
            continue
            
        test_loader_current = DataLoader(testing_dataset_current, 
                                         batch_size=args_config.get("eval_batch_size", 1), # Allow configurable batch size for eval
                                         shuffle=True, 
                                         num_workers=args_config.get("num_workers_eval", 4))
        
        # Make directories for this specific evaluation run's metrics/visualizations
        # (testing function also does this, but good to ensure at higher level too)
        eval_output_base = os.path.join(args_config["output_path"], "metrics", f"ARGS={args_config['arg_num']}", sub_class_item)
        os.makedirs(eval_output_base, exist_ok=True)

        testing(test_loader_current, args_config, unet_model_eval, seg_model_eval, 
                sub_class_item, class_type_name, checkpoint_type_to_load, device)

    print("\n--- All evaluations complete. ---")

if __name__ == '__main__':
    main()