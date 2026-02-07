import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.grouper import CombinatorialGrouper
import tqdm
import argparse

def train_model(mode='erm', epochs=10):
    # --- 1. Setup Data ---
    print(f"Preparing Waterbirds Data for [{mode.upper()}]...")
    dataset = get_dataset(dataset='waterbirds', download=True, root_dir='data')
    
    # Standard ResNet Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    grouper = CombinatorialGrouper(dataset, ['background', 'y'])
    
    train_data = dataset.get_subset('train', transform=transform)
    
    if mode == 'group_dro':
        # Explicitly sample 4 groups per batch for stability
        train_loader = get_train_loader('group', train_data, batch_size=16, grouper=grouper, n_groups_per_batch=4)
    else:
        train_loader = get_train_loader('standard', train_data, batch_size=64)
        
    test_data = dataset.get_subset('test', transform=transform)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    # --- 2. Setup Model (ResNet50) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    d = model.fc.in_features
    model.fc = nn.Linear(d, 2)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none') 

    # --- 3. Training Loop ---
    n_groups = 4
    group_weights = torch.ones(n_groups).to(device)
    group_step_size = 0.01

    for epoch in range(epochs):
        model.train()
        
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y_true, metadata in pbar:
            x, y_true = x.to(device), y_true.to(device)
            
            # Compute groups on CPU, then move to GPU
            groups = grouper.metadata_to_group(metadata).to(device)
            metadata = metadata.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss_per_sample = criterion(outputs, y_true)
            
            if mode == 'erm':
                loss = loss_per_sample.mean()
            
            elif mode == 'group_dro':
                unique_groups, group_indices = torch.unique(groups, return_inverse=True)
                group_losses = []
                for g_idx in range(n_groups):
                    mask = (groups == g_idx)
                    if mask.sum() > 0:
                        group_losses.append(loss_per_sample[mask].mean())
                    else:
                        group_losses.append(torch.tensor(0.0).to(device))
                group_losses = torch.stack(group_losses)
                
                loss = torch.dot(group_losses, group_weights)
                
                # Update group weights
                group_weights = group_weights * torch.exp(group_step_size * group_losses.detach())
                group_weights = group_weights / group_weights.sum()
                
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    # --- 4. Save Outputs (Correctly) ---
    print("Generating Audit Logits...")
    model.eval()
    all_logits = []
    all_labels = []
    all_metadata = []
    
    with torch.no_grad():
        for x, y, meta in tqdm.tqdm(test_loader, desc="Evaluating"):
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y)
            # Save metadata tensor directly
            all_metadata.append(meta)
            
    torch.save({
        'logits': torch.cat(all_logits),
        'labels': torch.cat(all_labels),
        'metadata': torch.cat(all_metadata), # Now included!
        'mode': mode
    }, f'waterbirds_{mode}_results.pt')
    
    print(f"Saved: waterbirds_{mode}_results.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['erm', 'group_dro'], required=True)
    args = parser.parse_args()
    
    train_model(mode=args.mode)