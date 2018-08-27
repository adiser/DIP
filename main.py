from dataset import MedicalImages
import torch
from torch.utils.data import Dataset, DataLoader
import pretrainedmodels
import time
import shutil
from config import parser

def main():
    #Hyper parameters
    arch = args.arch
    split_num = args.split_num
    lr = args.lr
    momentum = args.momentum
    start_epoch = args.start_epoch
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    eval_freq = args.eval_freq
    decay_factor = args.decay_factor
    preprocessing_filter = args.preprocessing_filter
    prefix = args.prefix
    
    print("Training {} with a momentum of {} and a decay factor of {}".format(arch, momentum, decay_factor))

    train_dataset = MedicalImages(arch = args.arch, split = split_num, train = True, preprocessing_filter = preprocessing_filter)
    val_dataset = MedicalImages(arch = args.arch, split = split_num, train = False, preprocessing_filter = preprocessing_filter)

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False)

    model_name = arch
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    num_features = model.last_linear.in_features 
    model.last_linear = torch.nn.Linear(num_features, 2)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = momentum)

    total_step = len(train_loader)

    best_prec = 0
    
    for epoch in range(start_epoch, num_epochs):

        train(model, train_loader, criterion, optimizer, epoch)

        if (epoch + 1) % eval_freq == 0 or epoch == num_epochs - 1:
            prec = validate(val_loader, model, criterion)

            is_best = prec > best_prec
            best_prec = max(prec, best_prec)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
            }, is_best)

        if (epoch+1) % 5 == 0:
            lr /= decay_factor
            update_lr(optimizer, lr)

    # model.save_state_dict()

def train(model, train_loader, criterion, optimizer, epoch):
    
    model.train()
    start = time.time()

    for i, (images, labels) in enumerate(train_loader):
        
        images = torch.autograd.Variable(images)
        labels = torch.autograd.Variable(labels)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        
        elapsed = end-start
        print("Epoch [{}], Iteration [{}/{}], Loss: {:.4f}, Elapsed Time {:.4f}"
            .format(epoch+1, i+1, len(train_loader), loss.data[0], elapsed))
        

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.prefix, args.arch, str(args.split_num), str(args.decay_factor), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.prefix, args.arch, str(args.split_num), str(args.decay_factor), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)

def validate(val_loader, model, criterion):

    model.eval()
    correct = 0
    total = 0
    start = time.time()
    for i, (images, labels) in enumerate(val_loader):
 
        images = torch.autograd.Variable(images, volatile=True)
        targets = torch.autograd.Variable(labels, volatile=True)

        # compute output
        outputs = model(images)
        loss = criterion(outputs, targets)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    prec = correct/total * 100

    end = time.time()
    elapsed = end-start

    print(('Testing Results: Prec {:.3f}%% Loss {:.5f} Elapsed {}'
          .format(prec, loss.data[0], elapsed)))
    
    return prec

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    args = parser.parse_args()
    main()