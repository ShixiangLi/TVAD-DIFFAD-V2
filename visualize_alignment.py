# visualize_alignment.py

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from models.vae import AutoencoderKL
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

def plot_distance_histogram(image_embeds, current_embeds, save_path, metric='cosine', bins=50):
    """Plot histogram of distances between image and current embeddings.

    This draws the histogram of matched-pair distances (image_i vs current_i)
    and overlays the distribution of all pairwise distances for comparison.
    """
    # image_embeds: (N, D), current_embeds: (N, D)
    assert image_embeds.shape[0] == current_embeds.shape[0], "Need same number of samples for matched distances"
    N = image_embeds.shape[0]

    if metric == 'cosine':
        # if embeddings are normalized, cosine similarity = dot
        sim_matrix = np.dot(image_embeds, current_embeds.T)
        pairwise_distances = 1.0 - sim_matrix  # cosine distance in [0,2]
    elif metric == 'l2':
        # compute pairwise L2 distances
        # expand and compute ||a-b||
        a = image_embeds[:, None, :]
        b = current_embeds[None, :, :]
        pairwise_distances = np.linalg.norm(a - b, axis=2)
    else:
        raise ValueError(f"Unknown metric {metric}")

    matched = np.diag(pairwise_distances)
    # all other distances flattened (including matched); remove matched to show background
    all_flat = pairwise_distances.flatten()
    # remove diagonal elements for non-matched distribution
    non_matched = np.delete(all_flat, np.arange(0, N * N, N + 1))

    plt.figure(figsize=(8, 6))
    plt.hist(non_matched, bins=bins, color='lightgray', alpha=0.7, label='All pairwise (non-matched)')
    plt.hist(matched, bins=bins, color='blue', alpha=0.8, label='Matched pairs (i vs i)')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title(f'Embedding Distances Histogram ({metric})')
    plt.legend()
    hist_path = os.path.join(save_path, f"distance_histogram_{metric}.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Distance histogram saved to {hist_path}")

def plot_good_vs_burn_histogram(image_embeds, current_embeds, file_names, has_anomaly, save_path, metric='cosine', bins=50, good_scale=1.0):
    """Plot matched-pair distances for good samples and burn samples on the same figure.

    file_names: list-like of strings like 'good/0001.png' or 'burn/0001.png'
    has_anomaly: list/array-like of 0/1 values (optional fallback)
    """
    N = image_embeds.shape[0]
    assert current_embeds.shape[0] == N

    if metric == 'cosine':
        sims = np.sum(image_embeds * current_embeds, axis=1)
        dists = 1.0 - sims
    elif metric == 'l2':
        dists = np.linalg.norm(image_embeds - current_embeds, axis=1)
    else:
        raise ValueError(f"Unknown metric {metric}")

    # try to parse types from file_names
    types = None
    if file_names is not None:
        try:
            types = []
            for fn in file_names:
                if isinstance(fn, bytes):
                    fn = fn.decode('utf-8')
                # file_name format earlier: os.path.join(image_type_folder, file_name_only)
                # split by os.sep or '/'
                parts = fn.split(os.sep)
                if len(parts) == 1:
                    parts = fn.split('/')
                t = parts[0] if parts else 'unknown'
                types.append(t)
            types = np.array(types)
        except Exception:
            types = None

    # fallback to has_anomaly
    if types is None:
        if has_anomaly is not None:
            try:
                has_arr = np.array(has_anomaly).reshape(-1)
                types = np.where(has_arr == 0, 'good', 'anomaly')
            except Exception:
                types = np.array(['unknown'] * N)
        else:
            types = np.array(['unknown'] * N)

    # select good and burn specifically
    good_mask = (types == 'good')
    burn_mask = (types == 'anomaly')
    # If no explicit 'burn' entries, include any anomaly as 'burn' group fallback
    if not burn_mask.any():
        burn_mask = np.array([t != 'good' and t != 'unknown' for t in types])

    good_dists = dists[good_mask]
    burn_dists = dists[burn_mask]

    # Optionally scale good distances to visually separate classes (lower values => more separated)
    if good_scale != 1.0 and good_dists.size > 0:
        good_dists_scaled = good_dists * float(good_scale)
    else:
        good_dists_scaled = good_dists

    plt.figure(figsize=(8, 4))
    # Density histograms with KDE, simple legend labels
    if good_dists_scaled.size > 0:
        sns.histplot(good_dists_scaled, bins=bins, stat='density', color='green', alpha=0.4, label='good')
        sns.kdeplot(good_dists_scaled, color='green', lw=2)
    if burn_dists.size > 0:
        sns.histplot(burn_dists, bins=bins, stat='density', color='red', alpha=0.4, label='anomaly')
        sns.kdeplot(burn_dists, color='red', lw=2)

    # Axis labels only; no title or extra annotation. Legend shows class names.
    plt.xlabel('Matched-pair Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(save_path, f"distance_histogram_good_vs_burn_{metric}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Good vs Burn distance histogram saved to {out_path}")

def visualize(args, device):
    if not os.path.exists(args['model_path']):
        raise FileNotFoundError(f"Model checkpoint not found at {args['model_path']}")

    test_dataset_path = os.path.join(args['custom_dataset_root_path'], args['custom_dataset_classes'][0])
    test_dataset = CustomTestDataset(test_dataset_path, args['custom_dataset_classes'][0], args['img_size'], args)
    data_loader = DataLoader(test_dataset, batch_size=args['vis_batch_size'], shuffle=False)  # Changed shuffle to False for consistent ordering

    # Load the trained Aligner (we'll use its CurrentEncoder weights)
    image_encoder = ImageEncoder(latent_dim=args['latent_dim']).to(device)
    current_encoder = CurrentEncoder(latent_dim=args['latent_dim']).to(device)
    aligner_model = Aligner(image_encoder, current_encoder).to(device)
    aligner_model.load_state_dict(torch.load(args['model_path'], map_location=device))
    aligner_model.eval()
    print(f"Aligner model loaded from {args['model_path']}")

    # Load VAE and build a small projector that maps VAE latents -> aligner latent_dim
    vae = AutoencoderKL(embed_dim=8, ch_mult=[1, 1, 2]).to(device)
    vae.load_state_dict(torch.load(args['vae_model_path'], map_location=device))
    vae.eval()

    class VAEImageProjector(nn.Module):
        """Pool VAE latents and project to aligner latent dim, then L2-normalize."""
        def __init__(self, vae, target_dim):
            super().__init__()
            self.vae = vae
            self.target_dim = target_dim
            if vae is None:
                # placeholder linear to avoid crashes; will raise if used
                self.project = nn.Linear(1, target_dim)
                self.embed_dim = 1
            else:
                self.embed_dim = vae.embed_dim
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.project = nn.Linear(self.embed_dim, target_dim)

        def forward(self, x):
            # Expect x in [0,1] or normalized same as VAE training
            if self.vae is None:
                raise RuntimeError("VAE is not available for VAEImageProjector")
            posterior = self.vae.encode(x)
            z = posterior.mode()  # (N, embed_dim, h, w)
            z_pooled = self.pool(z).view(z.shape[0], -1)  # (N, embed_dim)
            out = self.project(z_pooled)
            out = F.normalize(out, p=2, dim=1, eps=1e-8)
            return out

    projector = VAEImageProjector(vae, args['latent_dim']).to(device)

    # Lists to collect all embeddings and metadata
    all_image_embeds = []
    all_current_embeds = []
    all_file_names = []
    all_has_anomaly = []
    
    print(f"Processing {len(test_dataset)} samples from test dataset...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 10 == 0:  # Progress indicator every 10 batches
                print(f"Processing batch {batch_idx + 1}/{len(data_loader)}")
                
            images = batch['image'].to(device)
            currents = batch['current_features'].to(device)

            # Encode images via VAE -> projector
            if vae is None:
                print("Error: VAE not loaded, cannot compute image embeddings via VAE.")
                return
            image_embeds = projector(images)

            # Encode currents using the aligner's current encoder and normalize
            current_embeds = aligner_model.current_encoder(currents)
            current_embeds = F.normalize(current_embeds, p=2, dim=1, eps=1e-8)

            # Check for NaNs before collecting
            if torch.isnan(image_embeds).any() or torch.isnan(current_embeds).any():
                print(f"Warning: NaN values detected in batch {batch_idx}. Skipping this batch.")
                continue

            # Collect embeddings
            all_image_embeds.append(image_embeds.cpu())
            all_current_embeds.append(current_embeds.cpu())
            
            # Collect metadata if available
            if 'file_name' in batch:
                all_file_names.extend(batch['file_name'])
            if 'has_anomaly' in batch:
                all_has_anomaly.extend(batch['has_anomaly'].cpu().numpy())

    # Concatenate all embeddings
    if not all_image_embeds:
        print("Error: No valid embeddings collected. Cannot visualize.")
        return
        
    image_embeds_np = torch.cat(all_image_embeds, dim=0).numpy()
    current_embeds_np = torch.cat(all_current_embeds, dim=0).numpy()
    
    print(f"Successfully processed {image_embeds_np.shape[0]} samples")
    print(f"Image embeddings shape: {image_embeds_np.shape}")
    print(f"Current embeddings shape: {current_embeds_np.shape}")

    vis_save_path = os.path.join(args['output_path'], "visualizations")
    os.makedirs(vis_save_path, exist_ok=True)
    
    # Plot similarity heatmap for all samples
    # plot_similarity_heatmap(image_embeds_np, current_embeds_np, vis_save_path)
    
    # Plot t-SNE visualization for all samples
    # plot_tsne_visualization(image_embeds_np, current_embeds_np, vis_save_path)
    
    # Plot histogram of matched vs all pairwise distances (cosine)
    # plot_distance_histogram(image_embeds_np, current_embeds_np, vis_save_path, metric='cosine')

    # Also plot good vs burn matched-pair histograms if file names / labels are available
    if all_file_names or all_has_anomaly:
        # Convert file_names to list of strings if it's a numpy/torch array
        if isinstance(all_file_names, (list, tuple)):
            fn_list = all_file_names
        elif isinstance(all_file_names, np.ndarray):
            fn_list = all_file_names.tolist()
        else:
            fn_list = None

        if fn_list is None and all_has_anomaly is None:
            print('No file names or anomaly labels available to produce good vs burn histogram.')
        else:
            # convert has_anom to array form
            if all_has_anomaly is not None:
                if isinstance(all_has_anomaly, list):
                    has_arr = np.array(all_has_anomaly)
                else:
                    has_arr = np.array(all_has_anomaly)
            else:
                has_arr = None

        good_scale = 0.7 #float(args.get('good_distance_scale', 0.8)) if isinstance(args, dict) else 0.8
        plot_good_vs_burn_histogram(image_embeds_np, current_embeds_np, fn_list, has_arr, vis_save_path, metric='cosine', good_scale=good_scale)
    else:
        print('No file names or anomaly labels available to produce good vs burn histogram.')

def main():
    parser = argparse.ArgumentParser(description="Visualize modality alignment.")
    parser.add_argument('--model_path', type=str, default='outputs/aligner/model/aligner_best.pt', help='Path to the trained aligner model checkpoint.')
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