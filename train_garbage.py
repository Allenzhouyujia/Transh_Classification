import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from tqdm import tqdm
from sklearn.utils import class_weight
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
import contextlib

# Custom loss functions from train.py
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(weight=alpha, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Mixup function
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# CutMix function
def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)
    
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        torch.backends.cudnn.benchmark = True  # Enable benchmark mode for faster training

def get_class_distribution(dataset):
    targets = []
    for _, label in dataset.samples:
        targets.append(label)
    return Counter(targets)

def main():
    # Set random seed
    set_seed(42)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a garbage classification model')
    
    # Model and training parameters
    parser.add_argument('--backbone', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 
                                 'efficientnet_b0', 'efficientnet_b3', 'mobilenet_v2'],
                        help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'sgd', 'adamw'], help='Optimizer')
    
    # Loss function
    parser.add_argument('--loss_fn', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'label_smoothing', 'focal_loss'],
                        help='Loss function')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    
    # LR scheduler
    parser.add_argument('--lr_scheduler', type=str, default='reduce_on_plateau',
                        choices=['step', 'cosine', 'reduce_on_plateau', 'none'],
                        help='Learning rate scheduler')
    
    # Data augmentation
    parser.add_argument('--use_mixup', action='store_true', help='Use mixup augmentation')
    parser.add_argument('--use_cutmix', action='store_true', help='Use cutmix augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='Mixup alpha parameter')
    
    # Sample balancing
    parser.add_argument('--balance_samples', action='store_true', help='Balance class samples')
    
    # Performance optimizations
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading')
    parser.add_argument('--early_stopping', type=int, default=0, 
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--gradient_accumulation', type=int, default=1, 
                        help='Number of steps for gradient accumulation')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use (auto, cpu, cuda, mps)')
    
    # New dataset path
    parser.add_argument('--dataset_path', type=str, default='garbage_dataset', 
                        help='Path to the dataset')
    
    args = parser.parse_args()
    
    # 根据参数或自动检测选择设备
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch, 'has_mps') and torch.has_mps and torch.backends.mps.is_available():
            device = torch.device('mps')  # M1/M2 Mac的GPU加速
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Using backbone: {args.backbone}")
    print(f"Using optimizer: {args.optimizer}")
    print(f"Using loss function: {args.loss_fn}")
    print(f"Using LR scheduler: {args.lr_scheduler}")
    print(f"Using mixup: {args.use_mixup}")
    print(f"Using cutmix: {args.use_cutmix}")
    print(f"Using balanced sampling: {args.balance_samples}")
    
    # 根据设备类型设置混合精度训练
    use_amp = args.use_amp and (device.type == 'cuda')  # MPS目前不支持混合精度训练
    if args.use_amp and device.type == 'mps':
        print("Warning: Mixed precision training is not supported on MPS device. Disabling.")
        use_amp = False
    else:
        print(f"Using mixed precision: {use_amp}")
    
    print(f"Gradient accumulation steps: {args.gradient_accumulation}")
    
    # Scale for mixed precision training
    scaler = GradScaler() if use_amp else None
    
    # Data augmentation and normalization - optimize for speed
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        # Apply less expensive augmentations with higher probability
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ], p=0.5),  # Only apply some augmentations sometimes to save time
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets from the new path
    train_dataset = datasets.ImageFolder(root=os.path.join(args.dataset_path, 'TRAIN'), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(args.dataset_path, 'TEST'), transform=test_transform)

    # Class balancing with weighted sampling
    if args.balance_samples:
        class_distribution = get_class_distribution(train_dataset)
        print(f"Original class distribution: {class_distribution}")
        
        targets = torch.tensor([target for _, target in train_dataset.samples])
        class_weights = 1. / torch.tensor([class_distribution[t] for t in range(len(class_distribution))])
        sample_weights = class_weights[targets]
        
        sampler = WeightedRandomSampler(weights=sample_weights, 
                                         num_samples=len(sample_weights), 
                                         replacement=True)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
                                 sampler=sampler, num_workers=args.num_workers,
                                 pin_memory=True if device.type != 'cpu' else False, 
                                 prefetch_factor=2 if args.num_workers > 0 else None)
        print("Using weighted random sampler for class balancing")
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True if device.type != 'cpu' else False, 
                                 prefetch_factor=2 if args.num_workers > 0 else None)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True if device.type != 'cpu' else False, 
                            prefetch_factor=2 if args.num_workers > 0 else None)

    # Map class indices to class names
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # Load model based on specified backbone - for speed optimization
    backbone_args = {'pretrained': True}
    if args.backbone.startswith('efficientnet') or args.backbone == 'mobilenet_v2':
        # Newer torchvision versions use weights parameter instead of pretrained
        try:
            del backbone_args['pretrained']
            backbone_args['weights'] = 'IMAGENET1K_V1'
        except:
            pass  # Fallback to pretrained if this fails
    
    # Fast backbone loading
    if args.backbone == 'resnet18':
        model = models.resnet18(**backbone_args)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.backbone == 'resnet34':
        model = models.resnet34(**backbone_args)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.backbone == 'resnet50':
        model = models.resnet50(**backbone_args)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.backbone == 'resnet101':
        model = models.resnet101(**backbone_args)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(**backbone_args)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif args.backbone == 'efficientnet_b3':
        model = models.efficientnet_b3(**backbone_args)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif args.backbone == 'mobilenet_v2':
        model = models.mobilenet_v2(**backbone_args)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    
    model = model.to(device)

    # Calculate class weights for loss functions
    if args.loss_fn == 'focal_loss':
        class_distribution = get_class_distribution(train_dataset)
        # Calculate inverse frequency weights
        class_weights = torch.tensor([1.0 / class_distribution[i] for i in range(num_classes)], device=device)
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * num_classes
    else:
        class_weights = None

    # Define loss function
    if args.loss_fn == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_fn == 'label_smoothing':
        criterion = LabelSmoothingLoss(classes=num_classes, smoothing=args.smoothing)
    elif args.loss_fn == 'focal_loss':
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_fn}")

    # Define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Define learning rate scheduler
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.lr_scheduler == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    else:  # 'none'
        scheduler = None

    # Training metrics
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0
    early_stop_counter = 0
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        optimizer.zero_grad()  # Zero gradients before accumulation
        
        for batch_idx, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Apply mixup or cutmix augmentation
            use_mixup = args.use_mixup and random.random() < 0.5
            use_cutmix = args.use_cutmix and random.random() < 0.5 and not use_mixup
            
            # 使用autocast进行混合精度训练(仅CUDA支持)
            if use_amp:
                with autocast():
                    if use_mixup:
                        images, labels_a, labels_b, lam = mixup_data(images, labels, args.mixup_alpha)
                        outputs = model(images)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    elif use_cutmix:
                        images, labels_a, labels_b, lam = cutmix_data(images, labels, args.mixup_alpha)
                        outputs = model(images)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        # Standard forward pass
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    # Gradient accumulation
                    loss = loss / args.gradient_accumulation
            else:
                # 不使用混合精度
                if use_mixup:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, args.mixup_alpha)
                    outputs = model(images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                elif use_cutmix:
                    images, labels_a, labels_b, lam = cutmix_data(images, labels, args.mixup_alpha)
                    outputs = model(images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    # Standard forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Gradient accumulation
                loss = loss / args.gradient_accumulation
            
            # Backward pass with mixed precision
            if use_amp:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Calculate accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                
                if use_mixup or use_cutmix:
                    # For mixup/cutmix, calculate accuracy based on highest probability
                    correct += (lam * (predicted == labels_a).sum().item() + 
                                (1 - lam) * (predicted == labels_b).sum().item())
                else:
                    correct += (predicted == labels).sum().item()
                    
                running_loss += loss.item() * args.gradient_accumulation
                
                # Update progress bar
                train_bar.set_postfix(loss=loss.item() * args.gradient_accumulation, 
                                     acc=100.0*correct/total)
        
        # Make sure to step optimizer if batch size doesn't divide evenly
        if use_amp and (len(train_loader) % args.gradient_accumulation != 0):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        elif len(train_loader) % args.gradient_accumulation != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Testing phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Test]")
            for images, labels in test_bar:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                if use_amp:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                
                # Update progress bar
                test_bar.set_postfix(loss=loss.item(), acc=100.0*correct/total)
        
        test_loss = running_loss / len(test_loader)
        test_accuracy = 100.0 * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Update learning rate if using a scheduler
        if args.lr_scheduler == 'reduce_on_plateau':
            scheduler.step(test_loss)
        elif scheduler is not None:
            scheduler.step()
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
        
        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            early_stop_counter = 0  # Reset counter
            print(f"Saving best model with accuracy: {test_accuracy:.2f}%")
            model_name = f"best_model_garbage_{args.backbone}_{args.optimizer}_{args.loss_fn}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_accuracy,
                'backbone': args.backbone,
                'class_names': class_names
            }, model_name)
        else:
            # Early stopping
            if args.early_stopping > 0:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stopping:
                    print(f"Early stopping after {args.early_stopping} epochs without improvement")
                    break

    # Save the final model
    final_model_name = f"final_model_garbage_{args.backbone}_{args.optimizer}_{args.loss_fn}.pth"
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': test_accuracy,
        'backbone': args.backbone,
        'class_names': class_names
    }, final_model_name)

    # Plot training and testing metrics
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    # Plot learning rate if using scheduler
    if args.lr_scheduler != 'none' and args.lr_scheduler != 'reduce_on_plateau':
        plt.subplot(1, 3, 3)
        lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        plt.plot(lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')

    plt.tight_layout()
    plt.savefig(f'training_results_garbage_{args.backbone}_{args.optimizer}_{args.loss_fn}.png')
    
    print(f"Training completed! Best accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main() 