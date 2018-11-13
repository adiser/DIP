import pretrainedmodels
import torch
from dataset import MedicalImages
import pandas as pd
import argparse

def eval_to_csv(arch, k, input_type, pth_file, out_file):

    val_dataset = MedicalImages(arch = arch, split = k, train = False, preprocessing_filter = input_type, dataset = 'data_aug')
    val_loader = torch.utils.data.DataLoader(val_dataset, 1, shuffle = False)

    model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
    num_ftrs = model.last_linear.in_features
    model.last_linear = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(pth_file)['state_dict'])
    print("loaded checkpoint")
    model.eval()
    model.cuda()

    df = pd.DataFrame(columns = ['p_sah', 'p_sdh', 'targets'])
    sahs = []
    sdhs = []
    targets = []

    print("Predicting ...")
    for i, (image, label) in enumerate(val_loader):
        
        image = torch.autograd.Variable(image.cuda())
        
        output = model(image)
        
        sah_pred = output.data[0][0]
        sdh_pred = output.data[0][1]
        ground_truth = label[0]
        
        sahs.append(sah_pred)
        sdhs.append(sdh_pred)
        targets.append(ground_truth)
        if i % 100 == 0:
            print("Predicted {} samples".format(i+1))

    df['p_sah'] = sahs
    df['p_sdh'] = sdhs
    df['targets'] = targets

    print("Outputing csv file...")
    df.to_csv(out_file)
    print("Finished")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, default="resnet101")
    parser.add_argument('--k', type=int, default=0, choices = [0,1,2,3,4])
    parser.add_argument('--input_type', type=str, default=None)
    parser.add_argument('--pth_file', type=str)
    parser.add_argument('--out_file', type=str)

    args = parser.parse_args()
    eval_to_csv(args.arch, args.k, args.input_type, args.pth_file, args.out_file)


