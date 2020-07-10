'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pdb

from models import *
from utils import count_parameters
from data import CIFAR10, CIFAR100, DataParams, DATA_ROOTS, get_normalization_constants
from utils import save_model, load_model


MODELS_DIR = "/scratch/artemis/azouaoui/models/CIFAR10/"


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument("-m", "--model", type=int, default=0, help="Model type")
parser.add_argument("-e", "--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--mini", type=int, default=None, help="Number of data samples to train/test on")
parser.add_argument("-d", "--dataset", type=str, default="cifar10", help="Dataset", choices=["cifar10", "cifar100"])
parser.add_argument("-b", "--batch-size", type=int, default=128, help="Batch size")
parser.add_argument("--name", type=str, default="test", help="Experiment name")
parser.add_argument("-c", "--checkpoint", action="store_true", help="Checkpointing")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_epoch = -1
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

########
# Data #
########
print('==> Preparing data..')


dset = args.dataset

means, stds = get_normalization_constants(dset)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # randomly translate by up to 4 pixels in each direction
    # transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
])

root = DATA_ROOTS[dset]

train_params = DataParams(root=root,
                          train=True,
                          transform=transform_train,
                          target_transform=None,
                          download=False,
                          mini=args.mini)

test_params = DataParams(root=root,
                         train=False,
                         transform=transform_test,
                         target_transform=None,
                         download=False,
                         mini=args.mini)

if dset == "cifar10":
    trainset = CIFAR10(train_params)
    testset = CIFAR10(test_params)
    nclasses = 10
# TODO: Modify models to account for the different number of classes
elif dset == "cifar100":
    trainset = CIFAR100(train_params)
    testset = CIFAR100(test_params)
    nclasses = 100
else:
    raise NotImplementedError


# trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
           # 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')


# Build model dictionary
MODELS = {
    0: VGG('VGG19', nclasses),
    1: ResNet18(nclasses),
    2: PreActResNet18(nclasses),
    3: GoogLeNet(nclasses),
    4: DenseNet121(nclasses),
    5: ResNeXt29_2x64d(nclasses),
    6: MobileNet(nclasses),
    7: MobileNetV2(nclasses),
    8: DPN92(nclasses),
    9: SENet18(nclasses),
    10: ShuffleNetV2(1, nclasses),
    11: EfficientNetB0(nclasses),
    12: RegNetX_200MF(nclasses),
}
net = MODELS[args.model]


# Model information
print(f"Model architecture \n{net}")
print(f"Model has {count_parameters(net)} parameters")


net = net.to(device)
if device == 'cuda':
    # Do not use model parallelization
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)


#################
# Checkpointing #
#################
model_dir = os.path.join(MODELS_DIR, args.name)
model_path = os.path.join(model_dir, "checkpoint.pth")
best_path = os.path.join(model_dir, "checkpoint_best.pth")

if args.checkpoint:
    if not os.path.isfile(best_path):
        save_model(best_path, net, optimizer, scheduler, start_epoch, best_acc, best_epoch)
    # Load the model
    checkpoint = load_model(best_path)
    net.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"]
    best_acc = checkpoint["best_acc"]
    best_epoch = checkpoint["best_epoch"]


# Handle tensorboard
comment = '' if args.name == "test" else args.name
writer = SummaryWriter(comment=comment)

# Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch_idx % 50 ==  0:
            # print('Train Epoch: {} [{}/{} ({:})]')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(inputs), len(trainloader.dataset),
                    100. * (batch_idx + 1) / len(trainloader), loss.item()))


    train_acc = 100. * correct / len(trainloader.dataset)

    print("Train set accuracy: {}/{} ({:.2f}%)".format(correct, len(trainloader.dataset), train_acc))

    writer.add_scalar("loss/train", loss.item(), epoch)
    writer.add_scalar("accuracy/train", train_acc, epoch)


def test(epoch):
    global best_acc
    global best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_loss /= total
    test_acc = 100. * correct / total
    writer.add_scalar("loss/test", test_loss, epoch)
    writer.add_scalar("accuracy/test", test_acc, epoch)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, total,
            test_acc))





    # Save checkpoint.
    # acc = 100.*correct/total
    is_best = test_acc > best_acc
    if is_best:
        # print('Saving..')
        # state = {
        #     'net': net.state_dict(),
        #     'acc': acc,
        #     'epoch': epoch,
        # }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/ckpt.pth')
        best_acc = test_acc
        best_epoch = epoch
    writer.add_scalar("accuracy/best_test", best_acc, epoch)

    if args.checkpoint:
        save_model(model_path, net, optimizer, scheduler, epoch, best_acc, best_epoch, is_best)


for epoch in range(start_epoch, args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step(epoch)

print(f"Best score: {best_acc} (epoch {best_epoch})")

writer.close()

save_model(model_path, net, optimizer, scheduler, epoch, best_acc, best_epoch, is_final=True)
