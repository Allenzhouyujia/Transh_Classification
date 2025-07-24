#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import pandas as pd
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import multiprocessing
from torch.cuda.amp import autocast, GradScaler

# Check available devices, prioritize MPS for M2 Mac
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
print(f"Using device: {device}")

# Define model parameters - will be overridden by command line arguments
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Custom loss functions
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

# Define SeparableConv2d module for Xception
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Define Block module for Xception
class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        
        rep = []
        filters = in_filters
        
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        
        for i in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        
        if not start_with_relu:
            rep = rep[1:]
        
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        
        self.rep = nn.Sequential(*rep)
    
    def forward(self, inp):
        x = self.rep(inp)
        
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        
        x += skip
        return x

# Define Xception architecture
class Xception(nn.Module):
    def __init__(self, num_classes=4):
        super(Xception, self).__init__()
        
        # Entry Flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        
        # Middle Flow
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        
        # Exit Flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        
        # Add dropout layers
        self.dropout1 = nn.Dropout(0.5)  # After global pooling
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        # Entry Flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle Flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        
        # Exit Flow
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)  # Apply dropout before final classification
        x = self.fc(x)
        
        return x

# Define SimplicifiedXception architecture
class SimplifiedXception(nn.Module):
    def __init__(self, num_classes=4):
        super(SimplifiedXception, self).__init__()
        
        # Initial convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Middle flow blocks (simplified)
        self.block1 = nn.Sequential(
            SeparableConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Add dropout after activation
            SeparableConv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block2 = nn.Sequential(
            SeparableConv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Add dropout after activation
            SeparableConv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block3 = nn.Sequential(
            SeparableConv2d(256, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  # Add dropout after activation
            SeparableConv2d(728, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Feature extraction completion
        self.conv3 = SeparableConv2d(728, 1024, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.6)  # Increase dropout rate from 0.5 to 0.6
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # Input flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Feature completion
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Create a model with transfer learning
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=4, use_pretrained=True):
        super(TransferLearningModel, self).__init__()
        
        # Choose a pre-trained model
        self.model = models.efficientnet_b0(pretrained=use_pretrained)
        
        # Replace the classifier with a more robust one including multiple dropout layers
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),  # Increase dropout from 0.3 to 0.5
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),  # Add additional dropout layer
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Advanced data augmentation for training
train_transforms = transforms.Compose([
    # Basic preprocessing
    transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
    transforms.RandomCrop(IMG_SIZE),
    
    # Data augmentation transforms
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    
    # Convert to tensor and normalize
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

# Test set preprocessing - no data augmentation
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Mixup augmentation function
def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training function with mixup
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_mixup=True):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_size = inputs.size(0)
        processed_size += batch_size
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Initialize mixup variables
        lam = 1.0
        labels_a = labels
        labels_b = labels
        
        # Use mixup augmentation if enabled
        if use_mixup and np.random.random() < 0.8:  # Apply mixup 80% of the time
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
            labels_a, labels_b = labels_a.to(device), labels_b.to(device)
        
        # Use mixed precision if scaler is provided
        if scaler is not None:
            with autocast():
                outputs = model(inputs)
                if use_mixup and np.random.random() < 0.8:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
            
            # Scale loss and perform backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            outputs = model(inputs)
            if use_mixup and np.random.random() < 0.8:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * batch_size
        
        # For accuracy calculation (approximate with dominant label in mixup)
        _, preds = torch.max(outputs, 1)
        if use_mixup and np.random.random() < 0.8:
            # Use the dominant label for accuracy
            correct_preds = (lam * (preds == labels_a).float() + 
                           (1 - lam) * (preds == labels_b).float())
            running_corrects += correct_preds.sum().item()
        else:
            running_corrects += (preds == labels.data).sum().item()
    
    epoch_loss = running_loss / processed_size
    epoch_acc = running_corrects / processed_size
    
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    
    # Disable gradient calculation
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)
            processed_size += batch_size
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / processed_size
    epoch_acc = running_corrects.float() / processed_size
    
    return epoch_loss, epoch_acc.item()

# Improved training loop with validation set
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs=25, use_mixup=True, use_cutmix=False, mixup_alpha=0.2, early_stopping=8, use_amp=False):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    
    # Record training process
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_acc': []
    }
    
    # Early stopping settings
    patience = early_stopping
    early_stopping_counter = 0
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_mixup)
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        
        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Test phase (periodically check test performance)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
            history['test_acc'].append(test_acc)
        else:
            history['test_acc'].append(None)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduler
        if scheduler:
            scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'test_accuracy': test_acc if epoch % 5 == 0 or epoch == num_epochs - 1 else None,
            }, 'best_model_garbage_xception.pth')
            print(f'Saved new best model, accuracy: {val_acc:.4f}')
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
        
        print()
    
    # Calculate total training time
    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_val_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

# Test and visualize model performance
def test_model(model, test_loader, criterion, device, class_names):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate overall accuracy
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print detailed metrics for each class
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Calculate accuracy for each class
    class_acc = {}
    for i, name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:  # Avoid division by zero
            class_acc[name] = np.sum(np.array(all_preds)[class_mask] == i) / np.sum(class_mask)
        else:
            class_acc[name] = 0.0
    
    # Plot accuracy for each class
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(class_names)), [class_acc[name] for name in class_names], color='skyblue')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Class')
    
    # Add value labels to bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_accuracy.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'loss': test_loss,
        'class_accuracy': class_acc,
        'misclassified_indices': [i for i, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred != label]
    }

# Show memory usage
def print_memory_usage():
    try:
        import psutil
        process = psutil.Process()
        print(f"Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    except ImportError:
        print("Memory usage: psutil not available")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Xception garbage classification model')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', 
                        choices=['efficientnet_b0', 'efficientnet_b3'],
                        help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adamw', 
                        choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    
    # Loss function
    parser.add_argument('--loss_fn', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'label_smoothing', 'focal_loss'],
                        help='Loss function')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    
    # Learning rate scheduler
    parser.add_argument('--lr_scheduler', type=str, default='reduce_on_plateau',
                        choices=['step', 'cosine', 'reduce_on_plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr_patience', type=int, default=3, help='LR scheduler patience')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='LR scheduler factor')
    
    # Data augmentation
    parser.add_argument('--use_mixup', action='store_true', help='Use mixup augmentation')
    parser.add_argument('--use_cutmix', action='store_true', help='Use cutmix augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter')
    
    # Training options
    parser.add_argument('--early_stopping', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--balance_samples', action='store_true', help='Balance class samples')
    
    # Dataset path
    parser.add_argument('--dataset_path', type=str, default='garbage_dataset', 
                        help='Path to the dataset')
    
    args = parser.parse_args()
    
    # Update global variables with command line arguments
    global IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    dataset_path = args.dataset_path
    
    print(f"Training Configuration:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Image Size: {args.img_size}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Loss Function: {args.loss_fn}")
    print(f"  LR Scheduler: {args.lr_scheduler}")
    print(f"  Mixup: {args.use_mixup}")
    print(f"  CutMix: {args.use_cutmix}")
    print(f"  Early Stopping: {args.early_stopping}")
    print(f"  Mixed Precision: {args.use_amp}")
    print(f"  Balance Samples: {args.balance_samples}")
    print(f"  Dataset Path: {dataset_path}")
    
    # Load dataset
    print("Loading dataset...")
    
    # Load the training dataset from TRAIN folder
    train_path = os.path.join(dataset_path, 'TRAIN')
    full_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
    class_names = full_dataset.classes
    print(f"Dataset classes: {class_names}")
    
    # Count samples per class
    class_samples = {}
    for _, label in full_dataset:
        if label not in class_samples:
            class_samples[label] = 0
        class_samples[label] += 1
    
    for label, count in class_samples.items():
        print(f"Class {class_names[label]}: {count} images")
    
    # Calculate weights for weighted sampling
    class_weights = {}
    total_samples = len(full_dataset)
    n_classes = len(class_names)
    
    for label, count in class_samples.items():
        # Inverse frequency weighting
        class_weights[label] = total_samples / (n_classes * count)
    
    # Create sample weights for WeightedRandomSampler
    sample_weights = [class_weights[label] for _, label in full_dataset.samples]
    
    # Create indices for stratified split
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    
    # Create stratified splits
    class_indices = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label, indices in class_indices.items():
        n_samples = len(indices)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        # Shuffle indices for this class
        np.random.shuffle(indices)
        
        # Split indices for this class
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train+n_val])
        test_indices.extend(indices[n_train+n_val:])
    
    # Create datasets with different transformations
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    
    # Create validation and test datasets with test transforms
    val_full_dataset = datasets.ImageFolder(root=train_path, transform=test_transforms)
    test_full_dataset = datasets.ImageFolder(root=train_path, transform=test_transforms)
    
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(test_full_dataset, test_indices)
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Create weighted sampler for the training set
    if args.balance_samples:
        train_sampler = WeightedRandomSampler(
            weights=[sample_weights[i] for i in train_indices],
            num_samples=len(train_indices),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        print("Using weighted random sampler for class balancing")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model based on backbone choice
    print(f"Creating {args.backbone} model with transfer learning...")
    if args.backbone == 'efficientnet_b0':
        model = TransferLearningModel(num_classes=len(class_names), use_pretrained=True)
    elif args.backbone == 'efficientnet_b3':
        # Create EfficientNet-B3 model
        model = models.efficientnet_b3(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(class_names))
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    
    model = model.to(device)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print_memory_usage()
    
    # Define loss function
    if args.loss_fn == 'cross_entropy':
        if args.balance_samples:
            class_counts = [class_samples[i] for i in range(len(class_names))]
            class_weights = torch.FloatTensor([total_samples / (n_classes * count) for count in class_counts]).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.loss_fn == 'label_smoothing':
        criterion = LabelSmoothingLoss(classes=len(class_names), smoothing=args.smoothing)
    elif args.loss_fn == 'focal_loss':
        criterion = FocalLoss(alpha=None, gamma=args.focal_gamma)
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_fn}")
    
    # Define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=args.lr_factor)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.lr_scheduler == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=args.lr_factor, 
            patience=args.lr_patience
        )
    else:  # 'none'
        scheduler = None
    
    # Start training
    print("Starting training...")
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=args.num_epochs,
        use_mixup=args.use_mixup,
        use_cutmix=args.use_cutmix,
        mixup_alpha=args.mixup_alpha,
        early_stopping=args.early_stopping,
        use_amp=args.use_amp
    )
    
    # Plot training results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='validation')
    test_epochs = [i for i, acc in enumerate(history['test_acc']) if acc is not None]
    test_accs = [acc for acc in history['test_acc'] if acc is not None]
    if test_accs:
        plt.scatter(test_epochs, test_accs, label='test', color='green', marker='o')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{args.backbone}_training_results.png')
    
    # Final model evaluation
    print("\nStarting final model evaluation...")
    test_results = test_model(model, test_loader, criterion, device, class_names)
    
    # Save final model with backbone information
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_results['accuracy'],
        'class_accuracy': test_results['class_accuracy'],
        'backbone': args.backbone,
        'class_names': class_names
    }, f'final_model_garbage_{args.backbone}.pth')
    
    # Save best model
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_results['accuracy'],
        'class_accuracy': test_results['class_accuracy'],
        'backbone': args.backbone,
        'class_names': class_names
    }, f'best_model_garbage_{args.backbone}.pth')
    
    print("Model training and evaluation complete!")
    print(f"Overall test accuracy: {test_results['accuracy']:.4f}")
    print("Class accuracies:")
    for class_name, acc in test_results['class_accuracy'].items():
        print(f"  {class_name}: {acc:.4f}")
    print(f"\nResults charts saved: {args.backbone}_training_results.png, confusion_matrix.png, class_accuracy.png")

if __name__ == '__main__':
    # Solve multiprocessing issues
    multiprocessing.freeze_support()
    main() 