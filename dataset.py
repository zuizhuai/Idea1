import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class train_dataset(Dataset):
    def __init__(self, img_root, gt_root, gt_edge_root, trainsize):
        super(train_dataset, self).__init__()
        self.train_size = trainsize
        self.img = [os.path.join(img_root, f) for f in os.listdir(img_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gt = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gt_edge = [os.path.join(gt_edge_root, f) for f in os.listdir(gt_edge_root) if
                        f.endswith('.jpg') or f.endswith('.png')]
        self.imgs = sorted(self.img)
        self.gts = sorted(self.gt)
        self.gt_edge = sorted(self.gt_edge)
        self.filter_file()
        self.img_trans = transforms.Compose([
            transforms.Resize([self.train_size, self.train_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_trans = transforms.Compose([
            transforms.Resize([self.train_size, self.train_size]),
            transforms.ToTensor()
        ])

    def filter_file(self):
        assert len(self.imgs) == len(self.gts)
        images = []
        gts = []
        edge = []
        for img_path, gt_path, edge_path in zip(self.imgs, self.gts, self.gt_edge):
            Img = Image.open(img_path)
            gt = Image.open(gt_path)
            if Img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                edge.append(edge_path)

        self.imgs = images
        self.gts = gts
        self.gt_edge=edge

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')

        gt_path = self.gts[idx]
        gt = Image.open(gt_path).convert('L')

        edge_path=self.gt_edge[idx]
        gt_edge=Image.open(edge_path)

        img = self.img_trans(img)
        gt = self.gt_trans(gt)
        edge=self.gt_trans(gt_edge)

        return img, gt,edge

    def __len__(self):
        return len(self.imgs)


class test_dataset(Dataset):
    def __init__(self, image_root, gt_root, testsize):
        super(test_dataset, self).__init__()
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]
        name = name.replace('.jpg', '.png') if name.endswith('.jpg') else name

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
