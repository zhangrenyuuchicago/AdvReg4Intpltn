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

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
writer = SummaryWriter(logdir='ae_log/ae_log_{current_time}')

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(# like the Composition layer you built
                nn.Conv2d(1, 4, 3, stride=2, padding=1),
                nn.ReLU(),
	        nn.Conv2d(4, 8, 3, stride=2, padding=1),
	        nn.ReLU(),
	        nn.Conv2d(8, 16, 7)
	    )
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(16, 8, 7),
	        nn.ReLU(),
	        nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
	        nn.ReLU(),
	        nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1),
	        nn.Sigmoid()
	    )

    def forward(self, x):
        h = self.encoder(x)
        x = self.decoder(h)
        return x, h


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, h = model(data)
        ae_loss = F.mse_loss(output, data, reduction='mean')
        loss = ae_loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #	100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, epoch):
    model.eval()
    test_ae_loss = 0
    correct = 0
    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, h = model(data)
            test_ae_loss += F.mse_loss(output, data, reduction='mean').item()  # sum up batch loss 
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    #test_loss /= len(test_loader.dataset)
    writer.add_scalar('data/test_ae_loss', test_ae_loss, epoch)
    print('Test set: Average ae loss: {:.8f})'.format(test_ae_loss))
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #	test_loss, correct, len(test_loader.dataset),
    #	100. * correct / len(test_loader.dataset)))

def output_rep(model, device, train_loader, test_loader):
    model.eval()
    fout = open('rep_ae_test.csv', 'w')
    writer = csv.writer(fout, delimiter=',')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.encoder(data)
            output = output.view((output.size()[0],-1))
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
            output = model.encoder(data)
            output = output.view((output.size()[0],-1))
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
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
            help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
            help='learning rate (default: 1.0)')
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
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
            ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
            transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
            transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Autoencoder().to(device)
    
    optimizer = optim.Adadelta(list(model.parameters()), lr=args.lr)

    #scheduler = StepLR(optimizer, step_size=3, gamma=args.gamma, verbose=True)
    for epoch in range(1, args.epochs + 1):
        print(f'epoch: {epoch}')
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
        #scheduler.step()

        #output_rep(model, device, train_loader, test_loader)
        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")

    output_rep(model, device, train_loader, test_loader)

if __name__ == '__main__':
        main()

