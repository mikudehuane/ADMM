from copy import deepcopy

import torch
import torch as t
import torchvision
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage, transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import os.path as osp

from common import dataset_dir, cifar10, log_dir, mnist
from nn_models import NetACifar10, NetAMnist, NetBCifar10, NetBMnist, NetCMnist, NetAllConvMnist, NetAllFcMnist, \
    NetAllConvCifar10, NetMostFcCifar10, VGG16Cifar10, VGG16BNCifar10, VGG16Mnist, VGG19BNCifar10, VGG16BNMnist


def predict(net, images):
    """
    Please call torch.no_grad() outside this function
    """
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    return predicted


def accuracy(net, loader, max_checked=1000000, eval_before=True, train_after=True):
    if eval_before:
        net.eval()
    num_right = 0
    num_total = 0
    with torch.no_grad():
        for img_batch, label_batch in loader:
            img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
            num_total += len(label_batch)
            preds = predict(net, img_batch)
            num_right += (preds == label_batch).sum().item()
            if num_total >= max_checked:
                break
    if train_after:
        net.train()
    return num_right / num_total


def norm(net_x, net_z, net_u):
    norm = 0
    for x_par, z_par, u_par in zip(net_x.parameters(), net_z.parameters(), net_u.parameters()):
        norm += ((x_par - z_par + u_par) ** 2).sum()
    return norm


def detach(net):
    for par in net.parameters():
        par.requires_grad = False


def num_zero(net):
    with torch.no_grad():
        summ = 0
        total = 0
        for par in net.parameters():
            summ += (par == 0).sum()
            prod = 1
            for s in par.data.shape:
                prod *= s
            total += prod
    return summ, total


def update_z(net_z, net_x, net_u, l1_reg, rho):
    l1_thres = l1_reg / rho
    with torch.no_grad():
        for x_par, z_par, u_par in zip(net_x.parameters(), net_z.parameters(), net_u.parameters()):
            x_add_u = x_par.data + u_par.data
            x_add_u[x_add_u > l1_thres] -= l1_thres
            x_add_u[x_add_u < -l1_thres] += l1_thres
            x_add_u[(-l1_thres <= x_add_u) & (x_add_u <= l1_thres)] = 0
            z_par.data[...] = x_add_u


def main():
    dataset_indicator = "mnist"
    net_indicator = "netA"
    resize = None
    resize = None
    net_indicator = net_indicator + dataset_indicator
    batch_size = 200
    l1_reg = 0.001
    lr = 0.01
    num_its = 10000
    rho = 0.1
    num_inner_epoch = 1

    print(f"dataset: {dataset_indicator}, net: {net_indicator}")

    dataset_dict = dict(
        cifar10=cifar10,
        mnist=mnist,
    )
    train_set, test_set = dataset_dict[dataset_indicator](resize)
    train_loader = t.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = t.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    net_dict = dict(
        netAcifar10=NetACifar10, netAmnist=NetAMnist,
        netBcifar10=NetBCifar10, netBmnist=NetBMnist,
        netCmnist=NetCMnist,
        netAllConvcifar10=NetAllConvCifar10, netAllConvmnist=NetAllConvMnist,
        netAllFcmnist=NetAllFcMnist,
        netMostFccifar10=NetMostFcCifar10,
        VGG16cifar10=VGG16Cifar10, VGG16mnist=VGG16Mnist,
        VGG16BNcifar10=VGG16BNCifar10, VGG16BNmnist=VGG16BNMnist,
        VGG19BNcifar10=VGG19BNCifar10,
    )
    net = net_dict[net_indicator]()
    try:
        label = net.label
    except AttributeError as e:
        label = 'debug'

    print(net)

    loss_func = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(net.parameters(), lr=lr)

    start = time.time()

    net = net.cuda()
    net.train()

    net_u = deepcopy(net)
    for u_par in net_u.parameters():
        u_par.data.zero_()
    net_z = deepcopy(net)
    detach(net_u)
    detach(net_z)
    for current_it in range(num_its):
        # x minimization
        inner_it = 0
        for inner_epoch in range(num_inner_epoch):
            for it_idx, data in enumerate(train_loader):  # i start from 0
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                outputs = net(inputs)
                l_value = loss_func(outputs, labels)
                aug_norm = norm(net, net_z, net_u)
                loss =l_value + aug_norm * rho / 2

                print(f"Iteration: {current_it}-{inner_it}\t loss: {loss}", end='\r')

                loss.backward()
                optimizer.step()

                inner_it += 1
        test_acc_x = accuracy(net, test_loader)
        num_zero_x, total_par = num_zero(net)

        update_z(net_z, net, net_u, l1_reg=l1_reg, rho=rho)
        with torch.no_grad():
            for x_par, z_par, u_par in zip(net.parameters(), net_z.parameters(), net_u.parameters()):
                u_par.data[...] += (x_par.data - z_par.data)
        test_acc_z = accuracy(net_z, test_loader)
        num_zero_z, _ = num_zero(net_z)
        print(f"Iteration: {current_it}\t test_acc: x-{test_acc_x} z-{test_acc_z}\t"
              f"zero: x-{num_zero_x} z-{num_zero_z} / {total_par}")

    end = time.time()
    time_using = end - start
    print('finish training')
    print(f'time: {time_using}s')

    net_z.eval()
    train_acc = accuracy(net_z, train_loader)
    test_acc = accuracy(net_z, test_loader)
    print(f"Final train accuracy: {train_acc}, test accuracy: {test_acc}")


if __name__ == '__main__':
    main()