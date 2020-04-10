#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision import datasets , transforms
from torch.utils.data import DataLoader
import torch
import PIL

class Cifar10Data:
    def __init__(self,):
        self.train_transforms = transforms.Compose([transforms.RandomCrop(size=32 , padding=4 , padding_mode="symmetric",pad_if_needed=True),
                                      transforms.RandomRotation((0,5),resample = PIL.Image.NEAREST),
                                      transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      ])
        self.val_transforms = transforms.Compose([transforms.ToTensor()])
        self.trainset = datasets.CIFAR10(train=True,root = "data/",download=True,transform=self.train_transforms)
        self.valset  = datasets.CIFAR10(train=False , root="data/",download=True,transform=self.val_transforms)

        return None

    def dataloader(self, batch_size = 128, num_workers = 4):
        loader_param = {"batch_size":batch_size,
                        "pin_memory":True,
                        "num_workers":num_workers,
                        "shuffle":True}

        trainLoader = DataLoader(self.trainset,**loader_param)
        valLoader = DataLoader(self.valset  ,**loader_param)
        return {"train":trainLoader , "val":valLoader}
    
    @property
    def num_classes(self,): return len(self.trainset.classes)

