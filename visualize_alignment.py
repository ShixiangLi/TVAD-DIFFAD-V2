# visualize_alignment.py

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import argparse
import os
import json
from collections import defaultdict

from models.aligner import ImageEncoder, CurrentEncoder, Aligner
from data.dataset_beta_thresh import CustomTestDataset

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

def plot_similarity_heatmap(image_embeds, current_embeds, save_path):
    similarity_matrix = np.dot(image_embeds, current_embeds.T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=False, cmap="viridis")
    plt.title("Cosine Similarity Matrix (Image vs. Current)")
    plt.xlabel("Current Samples")
    plt.ylabel("Image Samples")
    heatmap_path = os.path.join(save_path, "similarity_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Similarity heatmap saved to {heatmap_path}")

def plot_tsne_visualization(image_embeds, current_embeds, save_path):
    num_samples = image_embeds.shape[0]
    all_embeds = np.vstack([image_embeds, current_embeds])
    
    # Robust perplexity setting for t-SNE
    perplexity = min(30.0, float(num_samples - 1))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(all_embeds)
    
    image_tsne, current_tsne = tsne_results[:num_samples], tsne_results[num_samples:]
    
    plt.figure(figsize=(12, 12))
    plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c='blue', label='Image', marker='o', alpha=0.7)
    plt.scatter(current_tsne[:, 0], current_tsne[:, 1], c='red', label='Current', marker='x', alpha=0.7)
    
    for i in range(num_samples):
        plt.plot([image_tsne[i, 0], current_tsne[i, 0]], [image_tsne[i, 1], current_tsne[i, 1]], 'k-', alpha=0.2)
        
    plt.title("t-SNE Visualization of Aligned Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True)
    tsne_path = os.path.join(save_path, "tsne_visualization.png")
    plt.savefig(tsne_path)
    plt.close()
    print(f"t-SNE visualization saved to {tsne_path}")

def visualize(args, device):
    if not os.path.exists(args['model_path']):
        raise FileNotFoundError(f"Model checkpoint not found at {args['model_path']}")

    test_dataset_path = os.path.join(args['custom_dataset_root_path'], args['custom_dataset_classes'][0])
    test_dataset = CustomTestDataset(test_dataset_path, args['custom_dataset_classes'][0], args['img_size'], args)
    data_loader = DataLoader(test_dataset, batch_size=args['vis_batch_size'], shuffle=True)

    image_encoder = ImageEncoder(latent_dim=args['latent_dim']).to(device)
    current_encoder = CurrentEncoder(latent_dim=args['latent_dim']).to(device)
    aligner_model = Aligner(image_encoder, current_encoder).to(device)
    aligner_model.load_state_dict(torch.load(args['model_path'], map_location=device))
    aligner_model.eval()
    print(f"Model loaded from {args['model_path']}")

    with torch.no_grad():
        batch = next(iter(data_loader))
        images = batch['image'].to(device)
        currents = batch['current_features'].to(device)
        image_embeds, current_embeds = aligner_model(images, currents)
        
        # Check for NaNs before visualization
        if torch.isnan(image_embeds).any() or torch.isnan(current_embeds).any():
            print("Error: NaN values detected in model output embeddings. Cannot visualize.")
            return

        image_embeds_np = image_embeds.cpu().numpy()
        current_embeds_np = current_embeds.cpu().numpy()

    vis_save_path = os.path.join(args['output_path'], "visualizations")
    os.makedirs(vis_save_path, exist_ok=True)
    
    plot_similarity_heatmap(image_embeds_np, current_embeds_np, vis_save_path)
    plot_tsne_visualization(image_embeds_np, current_embeds_np, vis_save_path)

def main():
    parser = argparse.ArgumentParser(description="Visualize modality alignment.")
    parser.add_argument('--model_path', type=str, default='outputs/aligner/model/aligner_epoch_200.pt', help='Path to the trained aligner model checkpoint.')
    parser.add_argument('--config', type=str, default='args/args_aligner.json', help='Path to the config file.')
    parser.add_argument('--vis_batch_size', type=int, default=64, help='Batch size for visualization.')
    cli_args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(cli_args.config, 'r') as f:
        args = defaultdict_from_json(json.load(f))
    
    args['model_path'] = cli_args.model_path
    args['vis_batch_size'] = cli_args.vis_batch_size
    
    visualize(args, device)

if __name__ == '__main__':
    main()