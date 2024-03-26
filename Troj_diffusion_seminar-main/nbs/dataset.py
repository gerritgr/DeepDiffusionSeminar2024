import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from tqdm.auto import tqdm

''' Inspired from: https://github.com/explainingai-code/DDPM-Pytorch '''

class Image_Dataset(Dataset):
    def __init__(self, path2data, transform=None, im_ext = '*.png'):
        self.path2data = path2data
        self.transform = transform
        self.im_ext = im_ext
        self.data, self.labels = self.load_images(path2data)

    def load_images(self, path2data):
        import os
        import glob
        im = []
        labels = []

        for i in os.listdir(path2data):
            files = glob.glob(os.path.join(path2data, i, self.im_ext))
            for j in files:
                im.append(j)
                labels.append(i)
        return im, labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        sample = Image.open(sample)

        if self.transform:
            sample = self.transform(sample)
        else:
            sample = transforms.ToTensor()(sample)
        
        sample = (2*sample) - 1    #normalizing the image between -1 and 1
        return sample, torch.tensor(int(label))