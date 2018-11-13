from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class MedicalImages(Dataset):
    def __init__(self, arch, split = 0, train = True, preprocessing_filter = None, dataset = "normal"):
        """
        Args:
            split (int): cross validation split
            train (bool): train flag
            arch (str): image recognition architecture -- dictates input sizes    
        """

        if dataset == 'normal':
            self.folder_name = 'data/{}'
        else:
            self.folder_name = 'data_aug/{}'

        if train:
            self.text_file = 'train_split_{}.txt'.format(split)
        else:
            self.text_file = 'test_split_{}.txt'.format(split)

        self.file_name = self.folder_name.format(self.text_file)
        
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

class TumorImages(Dataset):
    def __init__(self, arch, split = 0, train = True):
        self.split = split
        self.train = train 
        self.arch = arch
        
        if self.train:
            self.text_file = 'tumor_data/train_split_{}.txt'.format(split)
        else:
            self.text_file = 'tumor_data/test_split_{}.txt'.format(split)
            
        self.alter_input_to_model()
        self.transforms = transforms.Compose([transforms.Resize(self.input_size),
                                              transforms.Grayscale(3),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.input_mean, self.input_std)])
                                             
        with open(self.text_file) as fh:
            self.all_paths = fh.readlines()
            
    def alter_input_to_model(self):
        if 'resnet' in self.arch or 'vgg' in self.arch:
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

        elif self.arch == 'bninception':
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

        elif 'inception' in self.arch:
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
    
        
    def __getitem__(self, i):
        file_path = self.all_paths[i].strip()
        
        img = Image.open(file_path)        
        img = self.transforms(img)
        
        label = int(file_path[-6])
        
        return img, label
        
    
    def __len__(self):
        return len(self.all_paths)