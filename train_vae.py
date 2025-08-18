import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from models.vae import AutoencoderKL

default_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
class SimpleImageDataset(Dataset):
    def __init__(self, data_dir, transform=default_transform):
        self.paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)

def main():
    data_dir = 'datasets/combustion_dataset/chamber/train/good'
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        raise RuntimeError(f"数据目录 {data_dir} 不存在或为空，请先准备好训练图片！")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SimpleImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = AutoencoderKL(embed_dim=8, ch_mult=[1, 1, 2]).to(device)
    model.load_state_dict(torch.load('outputs/vae/vae_simple.pth', map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def recon_loss(x, xrec):
        return nn.MSELoss()(xrec, x)

    os.makedirs('outputs', exist_ok=True)
    print('开始训练...')
    model.train()
    for epoch in range(50):
        total_loss = 0
        for x in tqdm(dataloader, desc=f'Epoch {epoch+1}/50'):
            x = x.to(device)
            posterior = model.encode(x)
            z = posterior.sample()
            xrec = model.decode(z)
            kl = posterior.kl().mean()
            loss = recon_loss(x, xrec) + 1e-6 * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            torch.save(model.state_dict(), 'outputs/vae_simple.pth')
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')

    
        torch.save(model.state_dict(), 'outputs/vae_simple.pth')
    print('模型已保存到 outputs/vae_simple.pth')

if __name__ == '__main__':
    main() 