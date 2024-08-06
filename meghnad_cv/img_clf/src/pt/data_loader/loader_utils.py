import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils.common_defs import class_header, method_header


@class_header(
    description='''Mean and standard deviation calculator for cutom dataset''')
class CustomDataset(Dataset):
    def __init__(self, images_directory: str, transform=None, resize: tuple = (224, 224)):
        self.images_directory = images_directory
        self.transform = transform
        self.resize = resize
        self.image_files = self.get_image_files()


    @method_header(description='''Number of images in folder''',
                   returns='''returns number of images in int''')
    def __len__(self) -> int:
        return len(self.image_files)


    @method_header(description='''Load the image and apply transformations''',
                   arguments=''' Index for running through multiple file/ image index''',
                   returns='''return image tensor''')
    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path)
        image = image.resize(self.resize)
        if self.transform is not None:
            image = self.transform(image)
        return image


    @method_header(description='''Retrieve the image file names from the directory''',
                   returns=''''images files names''')
    def get_image_files(self):
        image_files = []
        for root, dirs, files in os.walk(self.images_directory):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_files.append(os.path.join(root, file))
        return image_files

@method_header(
    description='''Loop through images to get mean and standard deviation''',
    arguments='''Images_directory: i folder for where images are located,'
               'image_size:input size of the images,batch_size: Number of images per batch''',
    returns='''mean and standard devitation for train images''')
def mean_std(images_directory:str,image_size:tuple=(28,28),batch_size: int=32):
    transform = transforms.ToTensor()
    images_directory = images_directory
    dataset = CustomDataset(images_directory, transform=transform, resize=(image_size[0], image_size[1]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mean = 0.0
    std = 0.0
    total_images = 0
    for images in dataloader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_size

    mean /= total_images
    std /= total_images

    statistics = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    mean_list = mean.tolist()
    std_list = std.tolist()
    return mean_list,std_list