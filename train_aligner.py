import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from collections import defaultdict

from models.aligner import ImageEncoder, CurrentEncoder, Aligner
from models.vae import AutoencoderKL
from data.dataset_beta_thresh import CustomTrainDataset

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, top_k=32):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image_features, current_features):
        N = image_features.shape[0]
        device = image_features.device
        
        logits = torch.matmul(image_features, current_features.T) / self.temperature
        identity_mask = torch.eye(N, device=device, dtype=torch.bool)

        negative_logits_i2c = logits.clone()
        negative_logits_i2c[identity_mask] = -float('inf')
        top_k_vals_i2c, _ = torch.topk(negative_logits_i2c, min(self.top_k, N - 1), dim=1)
        positive_logits = logits.diag().unsqueeze(1)
        combined_logits_i2c = torch.cat([positive_logits, top_k_vals_i2c], dim=1)
        labels = torch.zeros(N, device=device, dtype=torch.long)
        loss_i2c = self.loss_fn(combined_logits_i2c, labels)
        
        logits_t = logits.T
        negative_logits_c2i = logits_t.clone()
        negative_logits_c2i[identity_mask] = -float('inf')
        top_k_vals_c2i, _ = torch.topk(negative_logits_c2i, min(self.top_k, N - 1), dim=1)
        combined_logits_c2i = torch.cat([positive_logits, top_k_vals_c2i], dim=1)
        loss_c2i = self.loss_fn(combined_logits_c2i, labels)

        return (loss_i2c + loss_c2i) / 2

def train(args, device):
    # 设置训练数据集和数据加载器
    train_dataset_path = os.path.join(args['custom_dataset_root_path'], args['custom_dataset_classes'][0])
    train_dataset = CustomTrainDataset(train_dataset_path, args['custom_dataset_classes'][0], args['img_size'], args)
    train_loader = DataLoader(train_dataset, batch_size=args['Batch_Size'], shuffle=True, num_workers=8, pin_memory=True)

    # 初始化模型、优化器和损失函数
    aligner_model = Aligner(
        ImageEncoder(latent_dim=args['latent_dim']),
        CurrentEncoder(latent_dim=args['latent_dim'])
    ).to(device)
    optimizer = optim.Adam(aligner_model.parameters(), lr=args['learning_rate'])
    loss_fn = HardNegativeContrastiveLoss(temperature=args['temperature'], top_k=args['hard_negative_top_k']).to(device)

    vae_model = AutoencoderKL(embed_dim=8, ch_mult=[1, 1, 2]).to(device)
    vae_model.load_state_dict(torch.load(args['vae_model_path'], map_location=device))
    vae_model.eval()

    start_epoch = 0
    best_train_loss = float('inf') # 使用训练损失作为评估依据
    model_dir = os.path.join(args['output_path'], "model")
    last_checkpoint_path = os.path.join(model_dir, "aligner_last.pt")
    best_model_path = os.path.join(model_dir, "aligner_best.pt")

    # 如果存在检查点，则恢复训练
    if os.path.exists(last_checkpoint_path):
        print(f"Resuming training from checkpoint: {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        aligner_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_train_loss = checkpoint.get('best_train_loss', float('inf'))
    
    print(f"Starting training with Hard Negative Mining (top_k = {args['hard_negative_top_k']})")

    for epoch in range(start_epoch, args['EPOCHS']):
        aligner_model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['EPOCHS']}")
        
        for batch in progress_bar:
            images = batch['image'].to(device)
            currents = batch['current_features'].to(device)

            posterior = vae_model.encode(images)
            z = posterior.sample()

            optimizer.zero_grad()
            image_embed, current_embed = aligner_model(z, currents)
            loss = loss_fn(image_embed, current_embed)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch+1}, skipping batch.")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aligner_model.parameters(), max_norm=args['gradient_clip_val'])
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_loss:.4f}")

        # 根据训练损失保存检查点和最佳模型
        if (epoch + 1) % args['save_every_epochs'] == 0:
            os.makedirs(model_dir, exist_ok=True)
            
            # 保存最新的检查点
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': aligner_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_train_loss': best_train_loss,
            }
            torch.save(checkpoint_data, last_checkpoint_path)
            
            epoch_save_path = os.path.join(model_dir, f"aligner_epoch_{epoch+1}.pt")
            torch.save(aligner_model.state_dict(), epoch_save_path)
            print(f"Checkpoint saved to {last_checkpoint_path} and {epoch_save_path}")

            # --- 根据训练损失保存最佳模型 ---
            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
                torch.save(aligner_model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path} with training loss: {best_train_loss:.4f}")

def main():
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open('args/args_aligner.json', 'r') as f:
        args = defaultdict_from_json(json.load(f))
    
    train(args, device)

if __name__ == '__main__':
    main()