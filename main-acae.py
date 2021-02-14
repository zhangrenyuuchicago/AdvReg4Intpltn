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

GAMMA=0.2
LAMBDA=0.5

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
writer = SummaryWriter(logdir=f'acae_log/acae_log_{current_time}')

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(# like the Composition layer you built
                nn.Conv2d(1, 4, 3, stride=2, padding=1),
                nn.LeakyReLU(),
	        nn.Conv2d(4, 8, 3, stride=2, padding=1),
	        nn.LeakyReLU(),
	        nn.Conv2d(8, 16, 7)
	    )
    
    def forward(self, x):
        h = self.encode(x)
        return h

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
                nn.ConvTranspose2d(16, 8, 7),
	        nn.LeakyReLU(),
	        nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
	        nn.LeakyReLU(),
	        nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1),
	        nn.Sigmoid()
	    )

    def forward(self, h):
        x = self.decode(h)
        return x

def loss4ae(x, rec_x, critic_x_alpha):
    sq_loss = F.mse_loss(rec_x, x, reduction='mean')
    critic_loss = critic_x_alpha.pow(2).mean()
    return sq_loss + LAMBDA*critic_loss

def loss4critic(critic_x_alpha, alpha, critic_inter_x_rec_x):
    sq_loss = F.mse_loss(critic_x_alpha, alpha, reduction='mean')
    critic_loss = critic_x_alpha.pow(2).mean()
    return sq_loss + critic_loss

def train(args, encoder, decoder, critic, device, train_loader, ae_opt, critic_opt, epoch):
    sum_critic_loss = 0
    sum_ae_loss = 0
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)    
        critic.train()
        encoder.eval()
        decoder.eval()
    
        critic_opt.zero_grad()
        z1 = encoder(data)
        z_shape = z1.size()
        z1 = z1.view((z1.size(0), -1))
        inv_idx = torch.arange(z1.size(0)-1, -1, -1).long().to(device)
        inv_tensor = z1.index_select(0, inv_idx)
        z2 = z1[inv_idx]
        
        alpha = torch.rand((data.size(0),1)).to(device)/2.0
        z_alpha = alpha*z1 + (1.0-alpha)*z2
        z_alpha= z_alpha.view(z_shape)
        x_alpha = decoder(z_alpha)
        critic_x_alpha = critic(x_alpha)
        z1_org = z1.view(z_shape)
        rec_x1 = decoder(z1_org)
        inter_x1_rec_x1 = GAMMA*data + (1.0-GAMMA)*rec_x1
        critic_inter_x1_rec_x1 = critic(inter_x1_rec_x1)
        
        critic_loss = loss4critic(critic_x_alpha, alpha, critic_inter_x1_rec_x1)
        sum_critic_loss += critic_loss.cpu().data.numpy()
        
        critic_loss.backward()
        critic_opt.step()
        
        critic.eval()
        encoder.train()
        decoder.train()

        ae_opt.zero_grad()
        z1 = encoder(data)
        z_shape = z1.size()
        z1 = z1.view((z1.size(0), -1))
        inv_idx = torch.arange(z1.size(0)-1, -1, -1).long().to(device)
        inv_tensor = z1.index_select(0, inv_idx)
        z2 = z1[inv_idx]

        alpha = torch.rand((data.size(0),1)).to(device)/2.0
        z_alpha = alpha*z1 + (1.0 - alpha)*z2
        z_alpha= z_alpha.view(z_shape)
        x_alpha = decoder(z_alpha)

        critic_x_alpha = critic(x_alpha)
        z1_org = z1.view(z_shape)
        rec_x1 = decoder(z1_org)

        ae_loss = loss4ae(data, rec_x1, critic_x_alpha)
        sum_ae_loss += ae_loss.cpu().data.numpy()

        ae_loss.backward()
        ae_opt.step()
        count += 1

    sum_critic_loss /= count
    sum_ae_loss /= count
    writer.add_scalar('data/critic_loss', sum_critic_loss, epoch)
    writer.add_scalar('data/ae_loss', sum_ae_loss, epoch)
    print(f'epoch: {epoch}, critic loss: {sum_critic_loss}, ae loss: {sum_ae_loss}')

'''
def test(encoder, decoder, critic, device, test_loader):
    #model.eval()
    encoder.eval()
    decoder.eval()
    critic.eval()

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

    print('Test set: Average ae loss: {:.8f})'.format(test_ae_loss))
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #	test_loss, correct, len(test_loader.dataset),
    #	100. * correct / len(test_loader.dataset)))
'''

def output_rep(encoder, decoder, critic, device, train_loader, test_loader):
    encoder.eval()
    decoder.eval()
    critic.eval()
    
    fout = open('rep_acae_test.csv', 'w')
    writer = csv.writer(fout, delimiter=',')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            z1 = encoder(data)
            output = z1.view((z1.size(0), -1))
            output = list(output.data.cpu().numpy())
            labels = list(target.data.cpu().numpy())
            for i in range(len(labels)):
                line = list(output[i]) + [labels[i]]
                writer.writerow(line)
    fout.close()

    fout = open('rep_acae_train.csv', 'w')
    writer = csv.writer(fout, delimiter=',')
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            z1 = encoder(data)
            output = z1.view((z1.size(0), -1))
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

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    critic = Critic().to(device)

    ae_opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) , lr=args.lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr)

    #ae_scheduler = StepLR(ae_opt, step_size=2, gamma=args.gamma, verbose=True)
    #critic_scheduler = StepLR(critic_opt, step_size=2, gamma=args.gamma, verbose=True)
    for epoch in range(1, args.epochs + 1):
        print(f'epoch: {epoch}')
        train(args, encoder, decoder, critic, device, train_loader, ae_opt, critic_opt, epoch)
        #test(model, device, test_loader)
        #ae_scheduler.step()
        #critic_scheduler.step()

        #output_rep(encoder, decoder, critic, device, train_loader, test_loader)

    output_rep(encoder, decoder, critic, device, train_loader, test_loader)

if __name__ == '__main__':
        main()

