import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

BATCH_SIZE = 500

def set_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class DeterministicCIFAR(torch.utils.data.Dataset):
    def __init__(self, order_seed, aug_seed, epochs, transform):
        self.dset = torchvision.datasets.CIFAR10(root='/tmp', train=True,
                                                 download=True, transform=None)
        self.transform = transform
        # generate order of data during training using order_seed
        set_seed(order_seed)
        self.data_order = []
        for _ in range(epochs):
            epoch_order = list(range(len(self.dset)))
            random.shuffle(epoch_order)
            self.data_order += epoch_order
        # generate seeds for each image augmentation using aug_seed
        set_seed(aug_seed)
        self.aug_seeds = np.random.randint(2**32, size=len(self.data_order))

    def __len__(self):
        return len(self.data_order)

    def __getitem__(self, iter_idx):
        epoch = iter_idx // len(self.dset)
        example_idx = self.data_order[iter_idx]
        aug_idx = len(self.dset) * epoch + example_idx

        img, lab = self.dset[example_idx]
        aug_seed = self.aug_seeds[aug_idx]
        set_seed(aug_seed)
        return self.transform(img), lab

def get_loaders(order_seed, aug_seed, epochs):
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    normalize = T.Normalize(np.array(CIFAR_MEAN)/255, np.array(CIFAR_STD)/255)

    train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    test_transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    train_dset = DeterministicCIFAR(order_seed, aug_seed, epochs, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=BATCH_SIZE,
                                               shuffle=False, num_workers=8)
    test_dset = torchvision.datasets.CIFAR10(root='/tmp', train=False,
                                             download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=8)
    return train_loader, test_loader

