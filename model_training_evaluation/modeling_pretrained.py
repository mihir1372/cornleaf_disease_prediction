# %% [code] {"execution":{"iopub.status.busy":"2024-12-02T04:56:41.801170Z","iopub.execute_input":"2024-12-02T04:56:41.801874Z","iopub.status.idle":"2024-12-02T04:56:41.806940Z","shell.execute_reply.started":"2024-12-02T04:56:41.801838Z","shell.execute_reply":"2024-12-02T04:56:41.806051Z"}}
from torch.utils.data import DataLoader, RandomSampler, Dataset, ConcatDataset
from torchvision import transforms, datasets
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import time
from tqdm import tqdm
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from sklearn.metrics import *
import seaborn as sns

import shap

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

N_CLASSES = 4
IMAGE_SIZE = 256
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001

class MinMaxNormalize(object):
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        normalized = (tensor - min_val) / (max_val - min_val)
        return normalized
    
    
class CombineDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.class_to_idx = self._merge_class_to_idx(datasets)

    def _merge_class_to_idx(self, datasets):
        merged = {}
        for dataset in datasets:
            if hasattr(dataset, 'class_to_idx'):
                merged.update(dataset.class_to_idx)
        return merged

train_transform = transforms.Compose(
    [
        transforms.Resize(size=(256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        MinMaxNormalize()
    ]
)

augment_transform = transforms.Compose(
    [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        MinMaxNormalize()
    ]    
)

val_transform = transforms.Compose(
    [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        MinMaxNormalize()
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        MinMaxNormalize()
    ]
)

train = datasets.ImageFolder(
    root="/kaggle/input/corn-leaf-dataset/data/train_test_split_data/train", transform=train_transform
)

augmented = datasets.ImageFolder(
    root="/kaggle/input/corn-leaf-dataset/data/train_test_split_data/augmented", transform=augment_transform
)

val = datasets.ImageFolder(
    root="/kaggle/input/corn-leaf-dataset/data/train_test_split_data/val", transform=val_transform
)

test = datasets.ImageFolder(
    root="/kaggle/input/corn-leaf-dataset/data/train_test_split_data/test", transform=test_transform
)

train_new = CombineDataset([train, augmented])

print("Dataset Labels:\n", train_new.class_to_idx, "\n")

combined_targets = train.targets + augmented.targets

for name, dataset in zip(["TRAIN", "VALIDATION", "TEST"], [train_new, val, test]):
    if name == "TRAIN":
        images_per_class = pd.Series(combined_targets).value_counts()
    else:
        images_per_class = pd.Series(dataset.targets).value_counts()
    print(f"Images per Class in {name}:")
    print(images_per_class, "\n")

labels_for_viz = {v: k for k, v in train_new.class_to_idx.items()}

fig, ax = plt.subplots(3, 5, figsize=(15, 10))
ax = ax.flatten()
for i in range(15):
    sample = random.randint(0, len(train_new))
    ax[i].imshow(train_new[sample][0].permute(1, 2, 0), cmap='gray')
    ax[i].title.set_text(labels_for_viz[train_new[sample][1]])

train_dataloader = DataLoader(dataset=train_new, batch_size=32, shuffle=True)

val_dataloader = DataLoader(dataset=val, batch_size=32, shuffle=True)

test_dataloader = DataLoader(dataset=test, batch_size=32, shuffle=False)

img, label = next(iter(train_dataloader))
print("Batch and Image Shape:", img.shape, "--> [batch_size, color_channels, height, width]")
print("\nLabels:", label)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.empty_cache()

criterion = nn.CrossEntropyLoss()

def train(model, train_loader, val_loader, num_epochs=10, patience=5, min_delta=0):
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    training_losses, training_accuracies = [], []
    validation_losses, validation_accuracies = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        training_accuracies.append(train_accuracy)
        print(f"Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        val_loss, val_accuracy = validate(model, val_loader)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered.")
            break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.show()

def validate(model, val_loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, val_accuracy

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    return test_accuracy, all_labels, all_predictions


# ### ResNet50

model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 4),           
)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"The model has {params} trainable parameters.")

# train(model, train_dataloader, val_dataloader, num_epochs=15, patience=5, min_delta=0.005)

model = torch.load("/kaggle/working/models/resnet50_best.pt", weights_only=False)

acc, actual, predicted = test(model, test_dataloader)

class_names = ["Blight", "Rust", "Spot", "Healthy"]

plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(actual, predicted), annot=True, fmt='d', cbar=False, xticklabels=["Blight", "Rust", "Spot", "Healthy"], yticklabels=["Blight", "Rust", "Spot", "Healthy"])
plt.title("ResNet-50")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("/kaggle/working/conf_mat_resnet.pdf", bbox_inches="tight", dpi=350)
plt.show()

print(classification_report(actual, predicted, target_names=class_names))

from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

n_classes = len(class_names)
all_labels_bin = label_binarize(actual, classes=range(n_classes))
all_predictions_bin = label_binarize(predicted, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_predictions_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(6, 5))
colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve for {class_names[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for ResNet-50')
plt.legend(loc="lower right")
plt.savefig("/kaggle/working/roc_resnet.pdf", bbox_inches="tight", dpi=350)
plt.show()

# torch.save(model, "/kaggle/working/models/resnet50_best.pt")
# torch.save(model.state_dict(), "/kaggle/working/models/resnet50_best_weights.pt")


# ### VGG16

model1 = models.vgg16(pretrained=True)
num_features = model1.classifier[6].in_features
model1.fc = nn.Sequential(
    nn.Linear(num_features, 4),          
)
model1 = model1.to(device)

optimizer = optim.Adam(model1.parameters(), lr=LEARNING_RATE)

model_parameters = filter(lambda p: p.requires_grad, model1.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"The model has {params} trainable parameters.")

# train(model1, train_dataloader, val_dataloader, num_epochs=EPOCHS, patience=10, min_delta=0.01)

model1 = torch.load("/kaggle/working/models/vgg16_best.pt", weights_only=False)

acc1, actual1, predicted1 = test(model1, test_dataloader)

plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(actual1, predicted1), annot=True, fmt='d', cbar=False, xticklabels=["Blight", "Rust", "Spot", "Healthy"], yticklabels=["Blight", "Rust", "Spot", "Healthy"])
plt.title("VGG-16")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("/kaggle/working/conf_mat_vgg.pdf", bbox_inches="tight", dpi=350)
plt.show()

print(classification_report(actual1, predicted1, target_names=class_names))

n_classes = len(class_names)
all_labels_bin = label_binarize(actual1, classes=range(n_classes))
all_predictions_bin = label_binarize(predicted1, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_predictions_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(6, 5))
colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve for {class_names[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for VGG-16')
plt.legend(loc="lower right")
plt.savefig("/kaggle/working/roc_vgg.pdf", bbox_inches="tight", dpi=350)
plt.show()

# torch.save(model1, "/kaggle/working/models/vgg16_best.pt")
# torch.save(model1.state_dict(), "/kaggle/working/models/vgg16_best_weights.pt")

# ### EfficientNet V2 M

model2 = models.efficientnet_v2_m(pretrained=True)
num_features = model2.classifier[1].in_features
model2.fc = nn.Sequential(
    nn.Linear(num_features, 4),    
)
model2 = model2.to(device)

optimizer = optim.Adam(model2.parameters(), lr=LEARNING_RATE)

model_parameters = filter(lambda p: p.requires_grad, model2.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"The model has {params} trainable parameters.")

# train(model2, train_dataloader, val_dataloader, num_epochs=EPOCHS, patience=5, min_delta=0.01)

model2 = torch.load("/kaggle/working/models/efficientnetv2_best.pt", weights_only=False)

acc2, actual2, predicted2 = test(model2, test_dataloader)

plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(actual2, predicted2), annot=True, fmt='d', cbar=False, xticklabels=["Blight", "Rust", "Spot", "Healthy"], yticklabels=["Blight", "Rust", "Spot", "Healthy"])
plt.title("EfficientNet-V2-Medium")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("/kaggle/working/conf_mat_eff.pdf", bbox_inches="tight", dpi=350)
plt.show()

print(classification_report(actual2, predicted2, target_names=class_names))

n_classes = len(class_names)
all_labels_bin = label_binarize(actual2, classes=range(n_classes))
all_predictions_bin = label_binarize(predicted2, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_predictions_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(6, 5))
colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve for {class_names[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for EfficientNet-V2 M')
plt.legend(loc="lower right")
plt.savefig("/kaggle/working/roc_eff.pdf", bbox_inches="tight", dpi=350)
plt.show()

# torch.save(model2, "/kaggle/working/models/efficientnetv2_best.pt")
# torch.save(model2.state_dict(), "/kaggle/working/models/efficientnetv2_best_weights.pt")
