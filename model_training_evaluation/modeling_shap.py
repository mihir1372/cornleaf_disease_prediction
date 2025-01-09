from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms, datasets
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from itertools import cycle
from sklearn.preprocessing import label_binarize
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

os.makedirs("../artifacts/models/")

class MinMaxNormalize(object):
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        normalized = (tensor - min_val) / (max_val - min_val)
        return normalized

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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

N_CLASSES = 4
IMAGE_SIZE = 256
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 70
LEARNING_RATE = 0.001

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model.to(device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"The model has {params} trainable parameters.")

# total_train=0
# for i, data in enumerate(train_dataloader, 0):
#     inputs, labels = data
#     total_train += labels.size(0)

# total_train

torch.cuda.empty_cache()

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# training loop
training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []

early_stopper = EarlyStopper(patience=5, min_delta=0.01)

for epoch in range(EPOCHS):
    start_time = time.time()
    
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) 
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    if epoch > 50:
        scheduler.step()
    
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for data in val_dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    end_time = time.time()
    epoch_time = end_time - start_time
    
    train_loss = running_loss / len(train_dataloader)
    train_accuracy = correct_train / total_train
    val_loss /= len(val_dataloader)
    val_accuracy = correct_val / total_val
    
    training_loss.append(train_loss)
    training_accuracy.append(train_accuracy)
    validation_loss.append(val_loss)
    validation_accuracy.append(val_accuracy)
    
    print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Time Taken: {epoch_time:.2f} seconds, LR: {optimizer.param_groups[0]["lr"]}')
    
    if early_stopper.early_stop(val_loss):
        print("Early stopping triggered.")
        break

plt.figure()
plt.plot(training_accuracy, label="training")
plt.plot(validation_accuracy, label="validation")
plt.legend()
plt.title("Training vs Valiation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.figure()
plt.plot(training_loss, label="training")
plt.plot(validation_loss, label="validation")
plt.legend()
plt.title("Training vs Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# model = torch.load("/kaggle/working/models/full_model_09381.pt", weights_only=False, map_location=torch.device('cpu'))

iterator = iter(DataLoader(dataset=train, batch_size=32, shuffle=False))

batch = next(iterator)
images, targets = batch
images = images.to(device)
targets

start_time = time.time()

background = images[0:20]
train_images = images[25:26]

e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(train_images)

end_time = time.time()

print(f"Time taken : {end_time - start_time} seconds")

with torch.no_grad():
    train_images = train_images.to(device)
    outputs = model(train_images)
    predicted_classes = outputs.argmax(dim=1)

predicted_classes

shap_values_predicted = [shap_values[0]]

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values_predicted]
train_numpy = np.swapaxes(np.swapaxes(train_images.cpu().numpy(), 1, -1), 1, 2)

plt.figure()
shap.image_plot(shap_numpy, train_numpy, show=False)
plt.savefig("/kaggle/working/shap_images/cnn_shap_4.jpg")

# testing loop
model.eval()
correct_test = 0
total_test = 0
test_loss = 0.0
all_labels = []
all_predictions = []

with torch.no_grad():
    for data in test_dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

test_loss /= len(test_dataloader)
test_accuracy = correct_test / total_test

print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')


class_names = ["Blight", "Rust", "Spot", "Healthy"]
plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(all_labels, all_predictions), annot=True, fmt='d', cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Custom CNN")
plt.show()

class_names = ["Blight", "Rust", "Spot", "Healthy"]
plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(all_labels, all_predictions), annot=True, fmt='d', cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Custom CNN")
plt.savefig("/kaggle/working/conf_mat.pdf", bbox_inches='tight', dpi=350)

report_dict = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict)
report_df

print(classification_report(all_labels, all_predictions, target_names=class_names))


# accuracy = f"{test_accuracy:.4f}".replace(".", "")
# filename = f"model_{accuracy}.pt"
# torch.save(model.state_dict(), f"/kaggle/working/models/{filename}")
# torch.save(model, f"/kaggle/working/models/full_{filename}")

n_classes = len(class_names)
all_labels_bin = label_binarize(all_labels, classes=range(n_classes))
all_predictions_bin = label_binarize(all_predictions, classes=range(n_classes))

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
plt.title('ROC Curves for Custom CNN')
plt.legend(loc="lower right")
plt.savefig("/kaggle/working/roc_custom.pdf", bbox_inches='tight', dpi=350)