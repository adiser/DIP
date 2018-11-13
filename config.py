import argparse
parser = argparse.ArgumentParser()
parser.add_argument('split_num', type=int, choices=[0,1,2,3,4])
parser.add_argument('log_file_name', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--pretraining', type=bool, default = False)
parser.add_argument('--dataset', type=str, choices = ['tumor_data', 'data', 'data_aug'])
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('-p', '--preprocessing_filter', type=str, default=None, choices = ['blur','sharp','contrast'])
parser.add_argument('--prefix', type=str, default="")
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])

# ========================= Learning Configs ==========================
parser.add_argument('--num_epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--decay_factor', default=3, type=int, metavar='M',
                    help='decay factor')

# ========================= Monitor Configs ==========================
parser.add_argument('--eval_freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
