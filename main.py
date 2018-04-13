import argparse

from model import Model
from dataset import CUBDataset
import torch
from torchvision import datasets
from configs import settings
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import nn
import torch.nn.functional as F
datasets.CUB = CUBDataset
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import os
from tqdm import tqdm


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()


def loss_fn(x, labels):
    main_loss = F.cross_entropy(x, labels)

    return main_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CapsNet')

    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-num_epochs', type=int, default=500)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-r', type=int, default=3)
    parser.add_argument('-pretrained', type=str, default="")
    parser.add_argument('--dataset', type=str, default='CUB', metavar='N',
                        help='name of dataset: SmallNORB or MNIST or NORB or CIFAR10')
    parser.add_argument('-gpu', type=int, default=1, help="which gpu to use")
    parser.add_argument('--loss', type=str, default='margin_loss', metavar='N',
                        help='loss to use: cross_entropy_loss, margin_loss, spread_loss')
    parser.add_argument('--routing', type=str, default='angle_routing', metavar='N',
                        help='routing to use: angle_routing, EM_routing, quickshift_routing, '
                             'reduce_noise_angle_routing')
    parser.add_argument('--use-recon', type=bool, default=True, metavar='N',
                        help='use reconstruction loss or not')
    parser.add_argument('--use-additional-loss', type=int, default=0, metavar='B',
                        help='use additional loss: 0: none, 1: contrastive, 2: lifted loss')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='num of workers to fetch data')
    parser.add_argument('-clip', type=float, default=5)
    parser.add_argument('--growthRate', type=int, default=12, metavar='N',
                        help='Growth rate for DenseNet.')
    parser.add_argument('--depth', type=int, default=110, help='Model depth.')
    parser.add_argument('--norm_template', type=int, default=1, help='Norm of the template')
    parser.add_argument('--multi-abstract', type=bool, default=False, metavar='N',
                        help='use multi level of abstraction or not')

    args = parser.parse_args()
    setting = settings[args.dataset]
    args.num_classes = setting['num_classes']
    args.env_name = '{}'.format(args.dataset)

    train_dataset = getattr(datasets, args.dataset)(root=setting['root'],
                                                    download=True,
                                                    train=True,
                                                    transform=setting['train_transform'])
    test_dataset = getattr(datasets, args.dataset)(root=setting['root'],
                                                   download=True,
                                                   train=False,
                                                   transform=setting['test_transform'])

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=True)
    use_cuda = torch.cuda.is_available()
    model = Model(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(setting['num_classes'], normalized=True)

    setting_logger = VisdomLogger('text', opts={'title': 'Settings'}, env=args.env_name)
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'}, env=args.env_name)
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'}, env=args.env_name)
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'}, env=args.env_name)
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'}, env=args.env_name)
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(args.num_classes)),
                                                     'rownames': list(range(args.num_classes))}, env=args.env_name)

    weight_folder = 'weights/{}'.format(args.env_name.replace(' ', '_'))
    if not os.path.isdir(weight_folder):
        os.mkdir(weight_folder)

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    steps, lambda_ = len(train_dataset) // args.batch_size, 5e-6,

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
        m = 0.8
        lambda_ = 0.9

    with torch.cuda.device(args.gpu):
        if use_cuda:
            print("activating cuda")
            model.cuda()

        for epoch in range(args.num_epochs):
            reset_meters()

            # Train
            print("Epoch {}".format(epoch))

            with tqdm(total=steps) as pbar:
                for data in train_loader:

                    imgs, labels = data  # b,1,28,28; #b
                    imgs, labels = Variable(imgs), Variable(labels)
                    if use_cuda:
                        imgs = imgs.cuda()
                        labels = labels.cuda()

                    labels = labels.squeeze()

                    out, x1, x2, loss_1, loss_2 = model(imgs)

                    loss = loss_fn(out, labels)

                    loss = loss + lambda_ * (loss_1.sum() + loss_2.sum())

                    optimizer.zero_grad()

                    # # compute and add gradient
                    # x1.retain_grad()
                    # x2.retain_grad()
                    #
                    # loss_1.backward(Variable(torch.ones(*loss_1.size()).cuda(args.gpu)), retain_graph=True)
                    # x1_grad = x1.grad.clone()
                    # optimizer.zero_grad()
                    #
                    # loss_2.backward(Variable(torch.ones(*loss_2.size()).cuda(args.gpu)), retain_graph=True)
                    # x2_grad = x2.grad.clone()
                    # optimizer.zero_grad()

                    loss.backward()
                    # x1.grad += args.lambda_ * x1_grad
                    # x2.grad += args.lambda_ * x2_grad

                    optimizer.step()

                    meter_accuracy.add(out.data, labels.data)
                    meter_loss.add(loss.data[0])
                    pbar.set_postfix(loss=meter_loss.value()[0], acc=meter_accuracy.value()[0])
                    pbar.update()

                loss = meter_loss.value()[0]
                acc = meter_accuracy.value()[0]

                if epoch == 0:
                    setting_logger.log(str(args))

                train_loss_logger.log(epoch, loss)
                train_error_logger.log(epoch, acc)

                print("\nEpoch{} Train acc:{:4}, loss:{:4}".format(epoch, acc, loss))
                scheduler.step(loss)
                torch.save(model.state_dict(), weight_folder + "/model_{}.pth".format(epoch))

                reset_meters()
                # Test
                print('Testing...')
                model.eval()
                for i, data in enumerate(test_loader):
                    imgs, labels = data  # b,1,28,28; #b
                    imgs, labels = Variable(imgs, volatile=True), Variable(labels, volatile=True)
                    if use_cuda:
                        imgs = imgs.cuda()
                        labels = labels.cuda()

                    labels = labels.squeeze()

                    out, x1, x2, loss_1, loss_2 = model(imgs)

                    loss = loss_fn(out, labels)

                    meter_accuracy.add(out.data, labels.data)
                    confusion_meter.add(out.data, labels.data)
                    meter_loss.add(loss.data[0])

                loss = meter_loss.value()[0]
                acc = meter_accuracy.value()[0]

                test_loss_logger.log(epoch, loss)
                test_accuracy_logger.log(epoch, acc)
                confusion_logger.log(confusion_meter.value())

                print("Epoch{} Test acc:{:4}, loss:{:4}".format(epoch, acc, loss))
                model.train()




