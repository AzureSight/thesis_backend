from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

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

class AugmentedImageFolder_resnet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = val_test_transform_resnet 

    def __len__(self):
        return 2 * len(self.dataset)

    def __getitem__(self, index):
        original_index = index // 2
        image, label = self.dataset[original_index]

        if index % 2 == 1:
            image = transforms.functional.hflip(image)

        image = self.transform(image)
        return image, label

class AugmentedImageFolder_inception(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = val_test_transform_inception 

    def __len__(self):
        return 2 * len(self.dataset)

    def __getitem__(self, index):
        original_index = index // 2
        image, label = self.dataset[original_index]

        if index % 2 == 1:
            image = transforms.functional.hflip(image)

        image = self.transform(image)
        return image, label

class Applytransform_resnet(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.transform = val_test_transform_resnet

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image_path, label = self.subset.dataset.samples[self.subset.indices[index]]  # Get correct file path

        image = Image.open(image_path).convert("RGB")  # Load im
        
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

        image = Image.open(image_path).convert("RGB")  # Load im
        
        if self.transform:
            image = self.transform(image)  # Apply transformations
        
        return image, label
