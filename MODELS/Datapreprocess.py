from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image as i
import torchvision.transforms.functional as F
import torch
from torchvision import datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


val_test_transform_resnet = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform_inception = transforms.Compose([
    transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Augmentation Transforms for ResNet
Aug_val_test_transform_resnet = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)]),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Augmentation Transforms for Inception
Aug_val_test_transform_inception = transforms.Compose([
    transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)]),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# class AugmentedImageFolder_resnet(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.transform = val_test_transform_resnet 

#     def __len__(self):
#         return 2 * len(self.dataset)

#     def __getitem__(self, index):
#         original_index = index // 2
#         image, label = self.dataset[original_index]

#         if index % 2 == 1:
#             image = transforms.functional.hflip(image)

#         image = self.transform(image)
#         return image, label
# class AugmentedImageFolder_inception(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.transform = val_test_transform_inception 

#     def __len__(self):
#         return 2 * len(self.dataset)

#     def __getitem__(self, index):
#         original_index = index // 2
#         image, label = self.dataset[original_index]

#         if index % 2 == 1:
#             image = transforms.functional.hflip(image)

#         image = self.transform(image)
#         return image, label


# class AugmentedImageFolder_resnet(Dataset):
#     def __init__(self, dataset, device=device):
#         self.dataset = dataset
#         self.device = device
#         self.transform = val_test_transform_resnet  # Ensure this works with GPU tensors

#     def __len__(self):
#         return 2 * len(self.dataset)

#     def __getitem__(self, index):
#         original_index = index // 2
#         image, label = self.dataset[original_index]

#         # # Move image to GPU
#         # image = image.to(self.device)

#         # Apply GPU-based horizontal flip
#         if index % 2 == 1:
#             image = F.hflip(image)  # GPU-based flip

#         # Apply transformations (ensure these support GPU tensors)
#         image = self.transform(image)  

#         return image, label
    
# class AugmentedImageFolder_inception(Dataset):
#     def __init__(self, dataset, device=device):
#         self.dataset = dataset
#         self.device = device
#         self.transform = val_test_transform_inception  # Ensure this works with GPU tensors

#     def __len__(self):
#         return 2 * len(self.dataset)

#     def __getitem__(self, index):
#         original_index = index // 2
#         image, label = self.dataset[original_index]

#         # Move image to GPU
#         # image = image.to(self.device)

#         # Apply GPU-based horizontal flip
#         if index % 2 == 1:
#             image = F.hflip(image)  # GPU-based augmentation

#         # Apply transformations (ensure they support GPU tensors)
#         image = self.transform(image)

#         return image, label


class AugmentedImageFolder_resnet(Dataset):
    def __init__(self, dataset, device=device):
        self.dataset = dataset
        self.device = device
        self.transform = val_test_transform_resnet  # Final transform (ensure it supports GPU tensors)

    def __len__(self):
        return 3 * len(self.dataset)  # 1 original + 4 augmented copies

    def __getitem__(self, index):
        original_index = index // 3  # Get the original image index
        image, label = self.dataset[original_index]

        # Apply different augmentations based on index
        aug_type = index % 3

        if aug_type == 1:
            image = F.hflip(image)  # Horizontal flip
        elif aug_type == 2:
            image = F.rgb_to_grayscale(image, num_output_channels=3)  # Grayscale (keep 3 channels)
        # elif aug_type == 3:
        #     image = F.adjust_brightness(image, 0.8)  # Darken (reduce brightness)
        # elif aug_type == 4:
        #     image = F.adjust_brightness(image, 1.5)  # Whiten (increase brightness)

        # Apply final transformation (ensure it works with GPU tensors)
        image = self.transform(image)  

        return image, label

class AugmentedImageFolder_inception(Dataset):
    def __init__(self, dataset, device=device):
        self.dataset = dataset
        self.device = device
        self.transform = val_test_transform_inception  # Final transform (ensure it supports GPU tensors)

    def __len__(self):
        return 3 * len(self.dataset)  # 1 original + 4 augmented copies

    def __getitem__(self, index):
        original_index = index // 3  # Get the original image index
        image, label = self.dataset[original_index]

        # Apply different augmentations based on index
        aug_type = index % 3

        if aug_type == 1:
            image = F.hflip(image)  # Horizontal flip
        elif aug_type == 2:
            image = F.rgb_to_grayscale(image, num_output_channels=3)  # Grayscale (keep 3 channels)
        # elif aug_type == 3:
        #     image = F.adjust_brightness(image, 0.5)  # Darken (reduce brightness)
        # elif aug_type == 4:
        #     image = F.adjust_brightness(image, 1.5)  # Whiten (increase brightness)

        # Apply final transformation (ensure it works with GPU tensors)
        image = self.transform(image)  

        return image, label


class DualTransformDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset  # Your ImageFolder dataset
        self.transform_resnet = val_test_transform_resnet
        self.transform_inception = val_test_transform_inception

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]  # Load image and label
        image_resnet = self.transform_resnet(image)
        image_inception = self.transform_inception(image)
        return image_resnet, image_inception, label




class DualTransformDataset_2(Dataset):
    def __init__(self, root):
        # Load dataset without transform (we apply transforms manually)
        self.dataset = datasets.ImageFolder(root=root)
        self.transform_resnet = val_test_transform_resnet
        self.transform_inception = val_test_transform_inception

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load image and label
        img_path, label = self.dataset.imgs[idx]
        img = self.dataset.loader(img_path)

        # Apply separate transformations
        img_resnet = self.transform_resnet(img)
        img_inception = self.transform_inception(img)

        return img_resnet, img_inception, label




















# class AugmentedImageFolder_resnet(Dataset):
#     def __init__(self, dataset, device=device):
#         self.dataset = dataset
#         self.device = device

#     def __len__(self):
#         return 2 * len(self.dataset)  # Double dataset size

#     def __getitem__(self, index):
#         original_index = index // 2  # Get base image
#         image, label = self.dataset[original_index]

#         # APPLY FRESH AUGMENTATION for every sample
#         augmented_image = Aug_val_test_transform_resnet(image)

#         # # Flip ONLY for odd indices
#         # if index % 2 == 1:
#         #     augmented_image = F.hflip(augmented_image)

#         return augmented_image, label
    
# class AugmentedImageFolder_inception(Dataset):
#     def __init__(self, dataset, device=device):
#         self.dataset = dataset
#         self.device = device
#         self.transform = Aug_val_test_transform_inception

#     def __len__(self):
#         return 2 * len(self.dataset)

#     def __getitem__(self, index):
#         original_index = index // 2
#         image, label = self.dataset[original_index]

#         # Apply transformations
#         image = self.transform(image)

#         return image, label

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
class Applytransform_resnet(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.transform = val_test_transform_resnet

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image_path, label = self.subset.dataset.samples[self.subset.indices[index]]  # Get correct file path

        image = i.open(image_path).convert("RGB")  # Load im
        
        if self.transform:
            image = self.transform(image)  # Apply transformations
        
        return image, label

class Applytransform_inception(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.transform = val_test_transform_inception

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image_path, label = self.subset.dataset.samples[self.subset.indices[index]]  # Get correct file path

        image = i.open(image_path).convert("RGB")  # Load im
        
        if self.transform:
            image = self.transform(image)  # Apply transformations
        
        return image, label
