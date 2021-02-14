from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import csv
from tensorboardX import SummaryWriter
from datetime import datetime
import math
import numpy as np

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
writer = SummaryWriter(logdir=f'ae_log/ae_log_{current_time}')

activation = nn.LeakyReLU
# authors use this initializer, but it doesn't seem essential
def Initializer(layers, slope=0.2):
    for layer in layers:
        if hasattr(layer, 'weight'):
            w = layer.weight.data
            std = 1/np.sqrt((1 + slope**2) * np.prod(w.shape[:-1]))
            w.normal_(std=std)  
        if hasattr(layer, 'bias'):
            layer.bias.data.zero_()

def Encoder(scales, depth, latent, colors):
    layers = []
    layers.append(nn.Conv2d(colors, depth, 1, padding=1))
    kp = depth
    for scale in range(scales):
        k = depth << scale
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
        layers.append(nn.AvgPool2d(2))
        kp = k
    k = depth << scales
    layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
    layers.append(nn.Conv2d(k, latent, 3, padding=1))
    Initializer(layers)
    return nn.Sequential(*layers)

def Decoder(scales, depth, latent, colors):
    layers = []
    kp = latent
    for scale in range(scales - 1, -1, -1):
        k = depth << scale
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
        layers.append(nn.Upsample(scale_factor=2))
        kp = k
    layers.extend([nn.Conv2d(kp, depth, 3, padding=1), activation()])
    layers.append(nn.Conv2d(depth, colors, 3, padding=1))
    Initializer(layers)
    return nn.Sequential(*layers)

def train(args, encoder, decoder, device, train_loader, optimizer, epoch):
    encoder.train()
    decoder.train()
    train_ae_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        h = encoder(data)
        output = decoder(h)
        ae_loss = F.mse_loss(output, data, reduction='mean')
        loss = ae_loss
        loss.backward()
        optimizer.step()
        train_ae_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #	100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    
    writer.add_scalar('data/train_ae_loss', train_ae_loss, epoch)
    print('Train set: Average ae loss: {:.8f})'.format(train_ae_loss))
    

def test(encoder, decoder, device, test_loader, epoch):
    encoder.eval()
    decoder.eval()
    test_ae_loss = 0
    correct = 0
    #ce_loss = nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            h = encoder(data)
            output = decoder(h)	
            test_ae_loss += F.mse_loss(output, data, reduction='mean').item()  # sum up batch loss 
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    #test_loss /= len(test_loader.dataset)
    writer.add_scalar('data/test_ae_loss', test_ae_loss, epoch)
    print('Test set: Average ae loss: {:.8f})'.format(test_ae_loss))
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #	test_loss, correct, len(test_loader.dataset),
    #	100. * correct / len(test_loader.dataset)))

def output_rep(encoder, decoder, device, train_loader, test_loader):
    encoder.eval()
    decoder.eval()
    fout = open('rep_ae_test.csv', 'w')
    writer = csv.writer(fout, delimiter=',')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = encoder(data)
            output = output.view((output.size(0),-1))
            output = list(output.data.cpu().numpy())
            labels = list(target.data.cpu().numpy())
            for i in range(len(labels)):
                line = list(output[i]) + [labels[i]]
                writer.writerow(line)
    fout.close()
    
    fout = open('rep_ae_train.csv', 'w')
    writer = csv.writer(fout, delimiter=',')
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = encoder(data)
            output = output.view((output.size(0),-1))
            output = list(output.data.cpu().numpy())
            labels = list(target.data.cpu().numpy())
            for i in range(len(labels)):
                line = list(output[i]) + [labels[i]]
                writer.writerow(line)
    fout.close()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
            help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
            help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
            help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
            help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
            help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
            help='For Saving the current Model')
    parser.add_argument('--width', type=int, default=32,
            help='width (default: 32)')
    parser.add_argument('--latent_width', type=int, default=4,
            help='latent_width (default: 4)')
    parser.add_argument('--depth', type=int, default=16,
            help='depth (default: 16)')
    parser.add_argument('--advdepth', type=int, default=16,
            help='advdepth (default: 16)')
    parser.add_argument('--advweight', type=float, default=0.5,
            help='advweight (default: 0.5)')
    parser.add_argument('--reg', type=float, default=0.2,
            help='reg (default: 0.2)')
    parser.add_argument('--latent', type=int, default=2,
            help='latent (default: 2)')
    parser.add_argument('--colors', type=int, default=1,
            help='colors (default: 1)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,'pin_memory': True,'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
            transforms.Resize((24,24)),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
            ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
            transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
            transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    scales = int(round(math.log(args.width // args.latent_width, 2)))
    encoder = Encoder(scales, args.depth, args.latent, args.colors).to(device)
    decoder = Decoder(scales, args.depth, args.latent, args.colors).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=1e-5)

    for epoch in range(1, args.epochs + 1):
        print(f'epoch: {epoch}')
        train(args, encoder, decoder, device, train_loader, optimizer, epoch)
        #test(encoder, decoder, device, test_loader, epoch)

        if args.save_model:
            torch.save(encoder.state_dict(), "mnist_cnn.pt")

    output_rep(encoder, decoder, device, train_loader, test_loader)

if __name__ == '__main__':
        main()

