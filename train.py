from random import seed
import torch
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from models.Recon_subnetwork import UNetModel
from models.Seg_subnetwork import SegmentationSubNetwork
from tqdm import tqdm
import torch.nn as nn
from data.dataset_beta_thresh import (
    MVTecTrainDataset,MVTecTestDataset,
    CustomTestDataset,CustomTrainDataset
)
from math import exp
import torch.nn.functional as F
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
from sklearn.metrics import roc_auc_score
import pandas as pd
from collections import defaultdict

def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)    

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True): 
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def train(training_dataset_loader, testing_dataset_loader, args, data_len, sub_class, class_type, device, 
          unet_model, seg_model, optimizer_ddpm, optimizer_seg, scheduler_seg, start_epoch=0):
   
    in_channels = args["channels"]
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    
    ddpm_sample =  GaussianDiffusionModel(
            args['img_size'], betas, 
            loss_weight=args.get('loss_weight', 'none'),
            loss_type=args['loss-type'], 
            noise=args["noise_fn"], 
            img_channels=in_channels
            )

    loss_focal = BinaryFocalLoss().to(device)
    loss_smL1= nn.SmoothL1Loss().to(device)
    
    tqdm_epoch = range(start_epoch, args['EPOCHS'])
    
    train_loss_list=[]
    
    best_combined_auroc = 0.0 
    best_image_auroc = 0.0
    best_pixel_auroc = 0.0
    best_epoch = start_epoch
    
    if start_epoch > 0:
        print(f"Evaluating loaded model from epoch {start_epoch} before continuing training...")
        best_image_auroc, best_pixel_auroc = eval(testing_dataset_loader, args, unet_model, seg_model, data_len, sub_class, device)
        best_combined_auroc = best_image_auroc + best_pixel_auroc
        print(f"Loaded model performance - Image AUROC: {best_image_auroc:.2f}, Pixel AUROC: {best_pixel_auroc:.2f}")

    for epoch in tqdm_epoch:
        unet_model.train()
        seg_model.train()
        
        epoch_train_loss = 0.0
        epoch_noise_loss = 0.0
        epoch_focal_loss = 0.0
        epoch_sml1_loss = 0.0

        tbar = tqdm(training_dataset_loader, desc=f"Epoch {epoch+1}/{args['EPOCHS']}", leave=False)
        for i, sample in enumerate(tbar):
            
            aug_image = sample['augmented_image'].to(device)
            anomaly_mask = sample["anomaly_mask"].to(device)
            anomaly_label = sample["has_anomaly"].to(device).squeeze()

            if 'current_features' in sample and sample['current_features'] is not None:
                 current_features = sample['current_features'].to(device)
            else:
                 print("Warning: 'current_features' not found in sample. Using dummy tensor.")
                 current_features = torch.randn(aug_image.shape[0], 24, 3, device=device) 

            noise_loss_val, pred_x0, normal_t, x_normal_t, x_noiser_t = ddpm_sample.norm_guided_one_step_denoising(
                unet_model, aug_image, anomaly_label, args, current_features
            )
            
            seg_input = torch.cat((aug_image, pred_x0), dim=1)
            pred_mask = seg_model(seg_input) 

            focal_loss_val = loss_focal(pred_mask, anomaly_mask)
            smL1_loss_val = loss_smL1(pred_mask, anomaly_mask)
            
            total_loss = noise_loss_val + args.get("focal_loss_weight", 5) * focal_loss_val + args.get("sml1_loss_weight", 1) * smL1_loss_val
            
            optimizer_ddpm.zero_grad()
            optimizer_seg.zero_grad()
            total_loss.backward()
            optimizer_ddpm.step()
            optimizer_seg.step()
            
            epoch_train_loss += total_loss.item()
            epoch_noise_loss += noise_loss_val.item()
            epoch_focal_loss += (args.get("focal_loss_weight", 5) * focal_loss_val.item())
            epoch_sml1_loss += (args.get("sml1_loss_weight", 1) * smL1_loss_val.item())

            tbar.set_postfix(loss=total_loss.item(), noise_loss=noise_loss_val.item(), focal=focal_loss_val.item(), sml1=smL1_loss_val.item())

        avg_epoch_train_loss = epoch_train_loss / len(training_dataset_loader)
        avg_epoch_noise_loss = epoch_noise_loss / len(training_dataset_loader)
        avg_epoch_focal_loss = epoch_focal_loss / len(training_dataset_loader)
        avg_epoch_sml1_loss = epoch_sml1_loss / len(training_dataset_loader)
        
        print(f"Epoch {epoch+1} Summary: Avg Loss: {avg_epoch_train_loss:.4f}, Avg Noise Loss: {avg_epoch_noise_loss:.4f}, Avg Focal: {avg_epoch_focal_loss:.4f}, Avg SmL1: {avg_epoch_sml1_loss:.4f}")

        scheduler_seg.step()

        if (epoch + 1) % args.get("eval_every_epochs", 10) == 0:
            temp_image_auroc, temp_pixel_auroc = eval(testing_dataset_loader, args, unet_model, seg_model, data_len, sub_class, device)

            current_combined_auroc = temp_image_auroc + temp_pixel_auroc
            if current_combined_auroc >= best_combined_auroc:
                if temp_image_auroc >= best_image_auroc :
                    best_combined_auroc = current_combined_auroc
                    best_image_auroc = temp_image_auroc
                    best_pixel_auroc = temp_pixel_auroc
                    best_epoch = epoch + 1
                    save(unet_model, seg_model, optimizer_ddpm, optimizer_seg, scheduler_seg, args=args, final='best', epoch=epoch + 1, sub_class=sub_class)
                    print(f"*** New best model saved at epoch {best_epoch} with Image AUROC: {best_image_auroc:.2f}, Pixel AUROC: {best_pixel_auroc:.2f} ***")

        if (epoch + 1) % args.get("log_loss_every_epochs", 5) == 0:
            train_loss_list.append(round(avg_epoch_train_loss, 4))

    save(unet_model, seg_model, optimizer_ddpm, optimizer_seg, scheduler_seg, args=args, final='last', epoch=args['EPOCHS'], sub_class=sub_class)
    print(f"Last model saved at epoch {args['EPOCHS']}")

    metrics_dir = os.path.join(args["output_path"], "metrics", f"ARGS={args['arg_num']}")
    os.makedirs(metrics_dir, exist_ok=True)
    csv_filename = os.path.join(metrics_dir, f"{args['eval_normal_t']}_{args['eval_noisier_t']}t_{args['condition_w']}_{class_type}_image_pixel_auroc_train.csv")
    
    performance_summary = {"classname": [sub_class], "Image-AUROC": [best_image_auroc], "Pixel-AUROC": [best_pixel_auroc], "epoch": [best_epoch]}
    df_class_summary = pd.DataFrame(performance_summary)
    
    file_exists = os.path.isfile(csv_filename)
    df_class_summary.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
    print(f"Training summary saved to {csv_filename}")


def eval(testing_dataset_loader, args, unet_model, seg_model, data_len, sub_class, device):
    unet_model.eval()
    seg_model.eval()
    
    in_channels = args["channels"]
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample = GaussianDiffusionModel(
            args['img_size'], betas, 
            loss_weight=args.get('loss_weight', 'none'),
            loss_type=args['loss-type'], 
            noise=args["noise_fn"], 
            img_channels=in_channels
            )
    
    total_image_pred = []
    total_image_gt = []
    total_pixel_gt = []
    total_pixel_pred = []
    
    with torch.no_grad():
        tbar_eval = tqdm(testing_dataset_loader, desc=f"Evaluating {sub_class}", leave=False)
        for i, sample in enumerate(tbar_eval):
            image = sample["image"].to(device)
            target_has_anomaly = sample['has_anomaly'].to(device)
            gt_mask = sample["mask"].to(device)

            if 'current_features' in sample and sample['current_features'] is not None:
                 current_features = sample['current_features'].to(device)
            else:
                 current_features = torch.randn(image.shape[0], 24, 3, device=device) 

            normal_t_tensor = torch.tensor([args["eval_normal_t"]], device=image.device).repeat(image.shape[0])
            noisier_t_tensor = torch.tensor([args["eval_noisier_t"]], device=image.device).repeat(image.shape[0])
            
            _, pred_x_0_condition, _, _, _, _, _ = ddpm_sample.norm_guided_one_step_denoising_eval(
                unet_model, image, normal_t_tensor, noisier_t_tensor, args, current_features
            )
            
            seg_input_eval = torch.cat((image, pred_x_0_condition), dim=1)
            pred_mask_seg = seg_model(seg_input_eval)
            out_mask = pred_mask_seg

            flat_out_mask = out_mask.view(out_mask.shape[0], -1) 
            
            if flat_out_mask.shape[1] >= 50:
                 topk_values, _ = torch.topk(flat_out_mask, 50, dim=1, largest=True)
                 image_score = torch.mean(topk_values, dim=1)
            else:
                 image_score = torch.mean(flat_out_mask, dim=1)


            total_image_pred.extend(image_score.cpu().numpy())
            total_image_gt.extend(target_has_anomaly.cpu().numpy())

            total_pixel_pred.extend(out_mask.flatten().cpu().numpy())
            total_pixel_gt.extend(gt_mask.flatten().cpu().numpy().astype(int))
            
    total_image_pred_np = np.array(total_image_pred)
    total_image_gt_np = np.array(total_image_gt)
    total_pixel_pred_np = np.array(total_pixel_pred)
    total_pixel_gt_np = np.array(total_pixel_gt)

    print(f"\nEvaluation for {sub_class}:")
    if len(np.unique(total_image_gt_np)) > 1:
        auroc_image = round(roc_auc_score(total_image_gt_np, total_image_pred_np) * 100, 2)
        print(f"  Image AUROC: {auroc_image:.2f}%")
    else:
        auroc_image = 0.0
        print(f"  Image AUROC: Not defined (only one class in ground truth)")
    
    if len(np.unique(total_pixel_gt_np)) > 1:
        auroc_pixel = round(roc_auc_score(total_pixel_gt_np, total_pixel_pred_np) * 100, 2)
        print(f"  Pixel AUROC: {auroc_pixel:.2f}%")
    else:
        auroc_pixel = 0.0
        print(f"  Pixel AUROC: Not defined (only one class in pixel ground truth)")
   
    return auroc_image, auroc_pixel


def save(unet_model, seg_model, optimizer_ddpm, optimizer_seg, scheduler_seg, args, final, epoch, sub_class):
    model_dir = os.path.join(args["output_path"], "model", f"diff-params-ARGS={args['arg_num']}", sub_class)
    os.makedirs(model_dir, exist_ok=True)
    
    save_path = os.path.join(model_dir, f"params-{final}.pt")
    
    torch.save(
        {
            'n_epoch': epoch,
            'unet_model_state_dict': unet_model.state_dict(),
            'seg_model_state_dict': seg_model.state_dict(),
            'optimizer_ddpm_state_dict': optimizer_ddpm.state_dict(),
            'optimizer_seg_state_dict': optimizer_seg.state_dict(),
            'scheduler_seg_state_dict': scheduler_seg.state_dict(),
            "args_summary": { 
                "arg_num": args["arg_num"],
                "img_size": args["img_size"],
                "base_channels": args["base_channels"],
                "class_type": args.get("class_type", "Unknown")
            }
        }, 
        save_path
    )
    print(f"Saved model checkpoint to {save_path} (Epoch: {epoch})")
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args_file_name = "args1.json"
    args_file_path = os.path.join('./args', args_file_name)
    if not os.path.exists(args_file_path):
        args_file_path = os.path.join(os.path.dirname(__file__), 'args', args_file_name)
        if not os.path.exists(args_file_path):
            args_file_path_alt = os.path.join(os.getcwd(), 'args', args_file_name)
            if os.path.exists(args_file_path_alt):
                args_file_path = args_file_path_alt
            else:
                print(f"Error: Args file not found at {args_file_path} or {args_file_path_alt}")
                return

    with open(args_file_path, 'r') as f:
        args_loaded = json.load(f)
    
    args = defaultdict_from_json(args_loaded)
    args['arg_num'] = args_file_name.split('.')[0].replace('args', '')
    
    manual_seed = args.get("seed", 42)
    seed(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
    print(f"Set seed to {manual_seed}")


    mvtec_classes = args.get('mvtec_classes', ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'])
    visa_classes = args.get('visa_classes', ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'])
    mpdd_classes = args.get('mpdd_classes', ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'])
    dagm_classes = args.get('dagm_classes', [f'Class{i}' for i in range(1, 11)])
    custom_dataset_classes = args.get('custom_dataset_classes', ['chamber'])

    current_classes_to_run = custom_dataset_classes

    for sub_class_name in current_classes_to_run:    
        print(f"\n--- Training for Class: {sub_class_name} ---")
        args["current_class"] = sub_class_name

        class_type_str = "Unknown"
        
        if sub_class_name in mvtec_classes and "mvtec_root_path" in args:
            dataset_root_path = os.path.join(args["mvtec_root_path"], sub_class_name)
            TrainDS = MVTecTrainDataset
            TestDS = MVTecTestDataset
            class_type_str = 'MVTec'
        elif sub_class_name in custom_dataset_classes and "custom_dataset_root_path" in args:
            dataset_root_path = os.path.join(args["custom_dataset_root_path"], sub_class_name)
            TrainDS = CustomTrainDataset
            TestDS = CustomTestDataset
            class_type_str = 'Custom'
        else:
            print(f"Warning: Dataset path or type not defined for class {sub_class_name}. Skipping.")
            continue
        
        args["class_type"] = class_type_str

        print(f"Using dataset path: {dataset_root_path} for class type: {class_type_str}")
        print(f"Training with args: {args['arg_num']}, Image Size: {args['img_size']}")     

        training_dataset = TrainDS(dataset_root_path, sub_class_name, img_size=args["img_size"], args=args)
        testing_dataset = TestDS(dataset_root_path, sub_class_name, img_size=args["img_size"])
        
        if len(training_dataset) == 0:
            print(f"Skipping {sub_class_name} due to empty training dataset.")
            continue

        training_loader = DataLoader(training_dataset, batch_size=args['Batch_Size'], shuffle=True, 
                                     num_workers=args.get("num_workers_train", 4), pin_memory=True, drop_last=True)
        testing_loader = DataLoader(testing_dataset, batch_size=args.get("eval_batch_size", 1), shuffle=False, 
                                    num_workers=args.get("num_workers_test", 4))

        output_dirs_to_create = [
            os.path.join(args["output_path"], "model", f"diff-params-ARGS={args['arg_num']}", sub_class_name),
            os.path.join(args["output_path"], "metrics", f"ARGS={args['arg_num']}", sub_class_name)
        ]
        for dir_path in output_dirs_to_create:
            try:
                os.makedirs(dir_path, exist_ok=True)
            except OSError as e:
                print(f"Error creating directory {dir_path}: {e}")
        
        test_data_len = len(testing_dataset) 
        
        in_channels = args["channels"]
        
        unet_model = UNetModel(
            img_size=args['img_size'], base_channels=args['base_channels'], 
            channel_mults=args.get('channel_mults', ""), dropout=args["dropout"], 
            n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            attention_resolutions=args["attention_resolutions"], in_channels=in_channels,
        ).to(device)

        seg_model = SegmentationSubNetwork(
            in_channels=in_channels * 2, out_channels=1
        ).to(device)
        
        optimizer_ddpm = optim.Adam(unet_model.parameters(), lr=args['diffusion_lr'], weight_decay=args['weight_decay'])
        optimizer_seg = optim.Adam(seg_model.parameters(), lr=args['seg_lr'], weight_decay=args['weight_decay'])

        scheduler_seg = optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=args.get('scheduler_T_max', args['EPOCHS']), eta_min=args.get('scheduler_eta_min', 0))

        start_epoch = 0
        checkpoint_path = os.path.join(args["output_path"], "model", f"diff-params-ARGS={args['arg_num']}", sub_class_name, "params-last.pt")
        
        if os.path.exists(checkpoint_path):
            print(f"发现已保存的最后模型: {checkpoint_path}。正在加载权重以继续训练...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            unet_model.load_state_dict(checkpoint['unet_model_state_dict'])
            seg_model.load_state_dict(checkpoint['seg_model_state_dict'])
            
            if 'optimizer_ddpm_state_dict' in checkpoint:
                optimizer_ddpm.load_state_dict(checkpoint['optimizer_ddpm_state_dict'])
                print("DDPM 优化器状态已加载。")

            if 'optimizer_seg_state_dict' in checkpoint:
                optimizer_seg.load_state_dict(checkpoint['optimizer_seg_state_dict'])
                print("分割模型优化器状态已加载。")

            if 'scheduler_seg_state_dict' in checkpoint:
                scheduler_seg.load_state_dict(checkpoint['scheduler_seg_state_dict'])
                print("分割模型学习率调度器状态已加载。")

            start_epoch = checkpoint['n_epoch']
            print(f"将从 Epoch {start_epoch + 1} 开始继续训练。")
        else:
            print(f"未找到已保存的模型。将从 Epoch 1 开始全新训练。")

        train(training_loader, testing_loader, args, test_data_len, sub_class_name, class_type_str, device,
              unet_model, seg_model, optimizer_ddpm, optimizer_seg, scheduler_seg, start_epoch)

    print("\n--- All training sessions complete. ---")

if __name__ == '__main__':
    main()