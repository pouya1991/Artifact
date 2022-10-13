import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
from skimage import io, transform
from submodule_cv.transformers.CutOut import CutOut
from submodule_cv.transformers.SizeJitter import SizeJitter
import numpy
import torchvision

class PatchDataset(Dataset):
    def __init__(self, x_set, y_set, model_config=None, training_set=False, size=-1):

        """
        Args:
            x_set (string): List of paths to images
            y_set (int): Labels of each image in x_set
            model_config (dict): Dict of model config.
        """
        self.x_set = x_set
        self.y_set = y_set
        self.model_config = model_config
        self.training_set = training_set
        self.size = size
        self.transform = self.get_transform()
        self.length = len(x_set)

        if len(x_set) != len(y_set):
            raise ValueError('x set length does not match y set length')

    def get_transform(self):

        transforms_array = []

        if self.model_config :
            self.normalize = True if 'normalize' in self.model_config and self.model_config['normalize']['use_normalize'] else False
            self.augmentation = True if 'augmentation' in self.model_config and self.model_config['augmentation']['use_augmentation'] else False
        else:
            self.normalize= False
            self.augmentation= False

        if self.augmentation:
            resize_ = True if 'resize' in self.model_config['augmentation'] else False
            crop_ = True if 'crop' in self.model_config['augmentation'] else False
            flip_ = True if 'flip' in self.model_config['augmentation'] else False
            rotation_ = True if 'rotation' in self.model_config['augmentation'] else False
            colo_jitter_ = True if 'color_jitter' in self.model_config['augmentation'] else False
            cut_out_ = True if 'cut_out' in self.model_config['augmentation'] else False
            size_jitter_ = True if 'size_jitter' in self.model_config['augmentation'] else False

        if self.size!=-1:
            if not self.augmentation:
                transforms_array.append(transforms.Resize(self.size))
            else:
                orig_size = self.original_size()
                if resize_: temp_size = self.model_config['augmentation']['resize']
                self.model_config['augmentation']['resize'] = self.size
                resize__ = True if resize_ else False
                if not resize_: resize_ = True
                if crop_:
                    temp_crop = self.model_config['augmentation']['crop']
                    self.model_config['augmentation']['crop'] = int((self.size/orig_size) * temp_crop)
                if self.training_set and cut_out_:
                    temp_cut_out = self.model_config['augmentation']['cut_out']['size_cut']
                    self.model_config['augmentation']['cut_out']['size_cut'] = int((self.size/orig_size) * temp_cut_out)
        if self.augmentation:
            if self.training_set:
                if flip_ and self.model_config['augmentation']['flip']:
                    transforms_array.append(transforms.RandomHorizontalFlip())
                    transforms_array.append(transforms.RandomVerticalFlip())
                if colo_jitter_ and self.model_config['augmentation']['color_jitter']:
                    transforms_array.append(transforms.ColorJitter(hue=.05, saturation=.05))
                if resize_:
                    transforms_array.append(transforms.Resize(self.model_config['augmentation']['resize']))
                if crop_:
                    transforms_array.append(transforms.RandomCrop(self.model_config['augmentation']['crop']))
                if size_jitter_ and self.model_config['augmentation']['size_jitter']['use_size_jitter']:
                    transforms_array.append(SizeJitter(self.model_config['augmentation']['size_jitter']['ratio'],
                                                       self.model_config['augmentation']['size_jitter']['probability'],
                                                       self.model_config['augmentation']['size_jitter']['color'],
                                                       self.model_config['augmentation']['size_jitter']['dynamic_bool'])
                    )
                if rotation_ and self.model_config['augmentation']['rotation']:
                    transforms_array.append(transforms.RandomRotation(20, interpolation=2))
            else:
                if resize_ and crop_:
                    transforms_array.append(transforms.Resize(self.model_config['augmentation']['crop']))
                elif crop_:
                    transforms_array.append(transforms.CenterCrop(self.model_config['augmentation']['crop']))
                elif resize_:
                    transforms_array.append(transforms.Resize(self.model_config['augmentation']['resize']))

        transforms_array.append(transforms.ToTensor())

        if (self.normalize):
            transforms_array.append(transforms.Normalize(mean=self.model_config['normalize']['mean'], std=self.model_config['normalize']['std']))
        else:
            transforms_array.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))

        if self.augmentation and self.training_set and cut_out_ and self.model_config['augmentation']['cut_out']['use_cut_out']:
            transforms_array.append(CutOut(self.model_config['augmentation']['cut_out']['num_cut'],
                                           self.model_config['augmentation']['cut_out']['size_cut'],
                                           self.model_config['augmentation']['cut_out']['color_cut']))

        if self.size!=-1 and self.augmentation:
            if resize__: self.model_config['augmentation']['resize'] = temp_size
            if not resize__: del self.model_config['augmentation']['resize']
            if crop_: self.model_config['augmentation']['crop'] = temp_crop
            if self.training_set and cut_out_: self.model_config['augmentation']['cut_out']['size_cut'] = temp_cut_out
        transforms_ = transforms.Compose(transforms_array)
        print(transforms_)
        return transforms_

    def original_size(self):
        x = Image.open(self.x_set[0][0]).convert('RGB')
        return transforms.ToTensor()(x).shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = Image.open(self.x_set[idx][0]).convert('RGB')
        y = self.y_set[idx]
        x = self.transform(x)
        return x, torch.tensor(y), self.x_set[idx][0], self.x_set[idx][1]
