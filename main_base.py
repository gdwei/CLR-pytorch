import torchvision
import torchvision.transforms as transforms
import torch
import os
import argparse
import h5py

import torch.optim as optim

from models import *
from utils import *

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument('--epochs', default=150, type=int, help='number of epochs')
parser.add_argument('--batch-size', default=100, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gpu', default='7', type=str, help='selected gpu')
parser.add_argument('--resume', '-r', action='store_true', help="resume from checkpoint")
parser.add_argument('--arch', default='Resnet18', type=str, help='selected arch')
parser.add_argument('--dataset', default='CIFAR-10', type=str, help='dataset')
parser.add_argument('--seed', default=0, type=int, help='random seed')


def create_record():
    global record
    record = {}
    record['train loss'] = []
    record['valid loss'] = []
    record['train accuracy'] = []
    record['valid accuracy'] = []


def save_record():
    fp = h5py.File(os.path.join(path, 'record.h5'), 'w')
    fp.cretea_dataset('train loss', data=record['train loss'])
    fp.cretea_dataset('valid loss', data=record['valid loss'])
    fp.cretea_dataset('train accuracy', data=record['train accuracy',])
    fp.cretea_dataset('valid accuracy', data=record['valid accuracy'])
    fp.cretea_dataset('test accuracy', data=record['test accuracy'])
    fp.close()


def train(net, trainloader, criterion, optimizer, epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    train_loss /= len(trainloader)
    correct /= total
    print('[Train Phase][Epoch: %3d/%3d][Loss: %.4f][Accuracy: %.4f]' %
          (epoch, opt.epochs, train_loss, 100. * correct))
    record['train loss'].append(train_loss)
    record['train accuracy'].append(correct)


def test(net, testloader, criterion, phase, epoch=0):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    test_loss /= len(testloader)
    correct /= total
    if phase == "Valid":
        print('[Valid Phase][Epoch: %3d/%3d][Loss: %.4f][Accuracy: %.4f]' %
              (epoch, opt.epochs, test_loss, 100. * correct))
        record['valid loss'].append(test_loss)
        record['valid accuracy'].append(correct)

    elif phase == "Test":
        print('[Test  Phase][Loss: %.4f][Accuracy: %.4f]' %
              (test_loss, 100. * correct))
        record['test accuracy'] = correct


def create_model():
    if opt.arch == "Lenet":
        net = LeNet()
        return net.cuda()
    elif opt.arch == "Resnet18":
        net = ResNet18()
        return net.cuda()
    elif opt.arch == "Resnet50":
        net = ResNet50()
        return net.cuda()
    elif opt.arch == "Resnet34":
        net = ResNet34()
        return net.cuda()

def adjusting_learning_rate(optimizer, epoch):
    if epoch in [50, 100, 150]:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10


def dataloader():
    if opt.dataset == "CIFAR-10":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])


        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

        NUM_VAL = 5000
        NUM_TEST = 5000
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        testloader = DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=4,
                                sampler=SubsetSampler(NUM_TEST, 0))

        validset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        validloader = DataLoader(validset, batch_size=opt.batch_size, shuffle=False, num_workers=4,
                                sampler=SubsetSampler(NUM_TEST, NUM_TEST))
        return (trainloader, testloader, validloader)


def main():
    (trainloader, testloader, validloader) = dataloader()
    net = create_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

    phase = 'Valid'
    for epoch in range(opt.epochs):
        train(net, trainloader, criterion, optimizer, epoch)
        test(net, validloader, criterion, phase, epoch)

    phase = 'Test'
    test(net, testloader, criterion, phase)

if __name__=="__main__":
    global opt, path
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    basename = '-'.join(['base', opt.dataset, opt.arch, datetime.now().strftime("%d-%H-%M-%S")])
    path = os.path.join('log', basename)
    os.mkdir(path)

    create_record()
    main()
    save_record()
