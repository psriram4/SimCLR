import cv2
import numpy as np
import torch 
import torchvision.transforms as transforms

class GaussianBlur:
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        
        return sample

class SimCLRTrainTransform:
    def __init__(self, input_height=224, gaussian_blur=True, jitter_strength=1.0, normalize=None):
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1
            data_transforms.append(GaussianBlur(kernel_size=kernel_size), p=0.5)

        data_transforms.append(transforms.ToTensor())

        if self.normalize:
            data_transforms.append(normalize)
    
        self.train_transform = transforms.Compose(data_transforms)

        # self.online_transform = transforms.Compose(
        #     [transforms.RandomResizedCrop(self.input_height), transforms.RandomHorizontalFlip(), self.final_transform]
        # )

    def __call__(self, sample):
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)

        return xi, xj


class SimCLRTestTransform(object):
    def __init__(self, input_height=224, normalize=None):
        self.input_height = input_height
        self.normalize = normalize

        data_transforms = [
            transforms.Resize(size=self.input_height),
            transforms.ToTensor()
        ]

        if self.normalize:
            data_transforms.append(normalize)
            
        self.test_transform = transforms.Compose(data_transforms)
        

    def __call__(self, sample):
        transform = self.test_transform
        xi = transform(sample)
        xj = transform(sample)

        return xi, xj
