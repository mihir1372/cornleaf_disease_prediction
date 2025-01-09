from torchvision import transforms, datasets

from utils.utils import Utils, AugmentImages, MinMaxScaler, CombineDataset

def main():
    # Utils.rename_images("./data/all_data")
    # Utils.denoise_images("./data/all_data", "./data/denoised_all_data")
    # Utils.train_test_split("./data/denoised_all_data", "./data/train_test_split_data")

    train_transform = transforms.Compose(
        [
            transforms.Resize(size=(256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            MinMaxScaler()
        ]
    )

    train = datasets.ImageFolder(
        root="./data/train_test_split_data/train", transform=train_transform
    )
        
    output_folder = "./data/train_test_split_data/augmented"

    augmented_dataset = AugmentImages(train, output_folder, train_transform)



if __name__ == "__main__":
    main()