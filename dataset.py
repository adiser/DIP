from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class MedicalImages(Dataset):
    def __init__(self, arch, split = 0, train = True, preprocessing_filter = None):
        """
        Args:
            split (int): cross validation split
            train (bool): train flag
            arch (str): image recognition architecture -- dictates input sizes    
        """
        if train:
            self.file_name = 'data/train_split_{}.txt'.format(split)
        else:
            self.file_name = 'data/val_split_{}.txt'.format(split)
        
        if 'resnet' in arch or 'vgg' in arch: 
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        
        elif arch == 'bninception':
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]
            
        elif 'inception' in arch:
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        
        else:
            raise ValueError("{} architecture is not recognized".format(arch))
        
        self.preprocessing_filter = preprocessing_filter

        self.transforms = transforms.Compose([
                                transforms.Resize(self.input_size),
                                transforms.CenterCrop(self.input_size),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                transforms.Normalize(self.input_mean, self.input_std)
                                ])
        
        with open(self.file_name) as fh:
            self.path_label_pairs = fh.readlines()
            
    def __getitem__(self, i):
        img_path, label = self.path_label_pairs[i].strip().split(' ')   

        if self.preprocessing_filter:
            img_path = img_path.split('/')
            img_path[1] = "{}_{}".format(img_path[1], self.preprocessing_filter)
            img_path = '/'.join(img_path)
        
        img = Image.open(img_path)
        
        img = self.transforms(img)
        
        label = int(label)
        
        return (img, label)
    
    def __len__(self):
        return (len(self.path_label_pairs))