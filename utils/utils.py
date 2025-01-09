import os
import numpy as np
import cv2
import shutil
import random
import torchvision.transforms.functional as TF

from skimage.restoration import denoise_nl_means, estimate_sigma
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


class Utils:
    def rename_images(main_folder):
        sub_folders = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
        for folder in sub_folders:
            folder_path = os.path.join(main_folder, folder)
            if os.path.exists(folder_path):
                files = os.listdir(folder_path)
                for index, filename in enumerate(sorted(files), start=1):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        new_name = f"{index:04d}_{folder.lower()}{os.path.splitext(filename)[1]}"
                        os.rename(
                            os.path.join(folder_path, filename),
                            os.path.join(folder_path, new_name),
                        )
        print("done renaming")

    def denoise_images(main_folder, output_folder):
        counter = 1
        sub_folders = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
        for folder in sub_folders:
            folder_path = os.path.join(main_folder, folder)
            output_path = os.path.join(output_folder, folder)

            os.makedirs(output_path, exist_ok=True)

            if os.path.exists(folder_path):
                files = os.listdir(folder_path)
                for filename in files:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(folder_path, filename)
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        sigma_est = np.mean(estimate_sigma(image, channel_axis=-1))
                        denoised_image = denoise_nl_means(
                            image,
                            h=3 * sigma_est,
                            patch_size=5,
                            patch_distance=1,
                            channel_axis=-1,
                            fast_mode=False,
                        )

                        output_filename = os.path.join(output_path, filename)
                        denoised_image = (denoised_image * 255).astype(np.uint8)
                        cv2.imwrite(
                            output_filename,
                            cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR),
                        )
                        print("done ", counter)
                        counter += 1

    def train_test_split(main_folder, output_folder):
        sub_folders = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

        train_folder = os.path.join(output_folder, "train")
        val_folder = os.path.join(output_folder, "val")
        test_folder = os.path.join(output_folder, "test")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        for folder in sub_folders:
            folder_path = os.path.join(main_folder, folder)
            if os.path.exists(folder_path):
                files = [
                    f
                    for f in os.listdir(folder_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]

                random.shuffle(files)

                total_files = len(files)
                train_end = int(0.6 * total_files)
                val_end = int(0.8 * total_files)

                train_files = files[:train_end]
                val_files = files[train_end:val_end]
                test_files = files[val_end:]

                # Create sub-folders for each class in the output folders
                class_train_folder = os.path.join(train_folder, folder)
                class_val_folder = os.path.join(val_folder, folder)
                class_test_folder = os.path.join(test_folder, folder)
                os.makedirs(class_train_folder, exist_ok=True)
                os.makedirs(class_val_folder, exist_ok=True)
                os.makedirs(class_test_folder, exist_ok=True)

                counter = 1
                for f in train_files:
                    shutil.copy(os.path.join(folder_path, f), class_train_folder)
                    print("Into training ", counter)
                    counter += 1
                counter = 1

                for f in val_files:
                    shutil.copy(os.path.join(folder_path, f), class_val_folder)
                    print("Into validation ", counter)
                    counter += 1
                counter = 1

                for f in test_files:
                    shutil.copy(os.path.join(folder_path, f), class_test_folder)
                    print("Into testing ", counter)
                    counter += 1
                counter = 1

    def calculate_mean_std(folder_path):
        channel_mean = []
        channel_std = []
        total_img = 0

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(("png", "jpg", "jpeg")):
                    img_path = os.path.join(root, file)
                    image = Image.open(img_path)
                    image = image.convert("RGB")
                    img_array = np.array(image).astype(np.float32) / 255.0

                    channel_mean.append(np.mean(img_array, axis=(0, 1)))
                    channel_std.append(np.std(img_array, axis=(0, 1)))
                    total_img += 1

        channel_mean = np.array(channel_mean)
        channel_std = np.array(channel_std)

        mean = np.mean(channel_mean, axis=0)
        std = np.mean(channel_std, axis=0)

        return mean, std


class MinMaxScaler(object):
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        normalized = (tensor - min_val) / (max_val - min_val)
        return normalized


class AugmentImages(Dataset):
    def __init__(self, original_dataset, output_folder, train_transform):
        self.original_dataset = original_dataset
        self.output_folder = output_folder
        self.transform = train_transform

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.augment_and_export()

    def augment_and_export(self):
        class_names = {0: "Blight", 1: "Common_Rust", 2: "Gray_Leaf_Spot", 3: "Healthy"}

        for idx, (image, label) in enumerate(self.original_dataset):
            class_folder = os.path.join(self.output_folder, f"{class_names[label]}")
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            for i in range(2):
                augmented_image = self.transform(TF.to_pil_image(image))
                augmented_image = TF.to_pil_image(augmented_image)

                filename = f"augmented_{idx}_{i}.jpg"
                save_path = os.path.join(class_folder, filename)
                augmented_image.save(save_path)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        return self.original_dataset[idx]


class CombineDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.class_to_idx = self._merge_class_to_idx(datasets)

    def _merge_class_to_idx(self, datasets):
        merged = {}
        for dataset in datasets:
            if hasattr(dataset, "class_to_idx"):
                merged.update(dataset.class_to_idx)
        return merged
