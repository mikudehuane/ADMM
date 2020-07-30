import torchvision.transforms as transforms
import torchvision as tv
import os.path as osp


# dataset_dir = '/root/data'
dataset_dir = 'D:\\Projects\\data'
pretrained_models_dir = 'pretrained_models'
log_dir = 'log'
fig_fd = 'figs'


def cifar10(resize=None):
    _transform_cifar10 = []
    if resize is not None:
        _transform_cifar10.append(transforms.Resize(resize))
    _transform_cifar10.extend([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_cifar10 = transforms.Compose(_transform_cifar10)
    cifar10_trainset = tv.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        download=False,
        transform=transform_cifar10
    )
    cifar10_testset = tv.datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        download=False,
        transform=transform_cifar10
    )
    return cifar10_trainset, cifar10_testset


cifar10_classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def mnist(resize=None):
    _transform_mnist = []
    if resize is not None:
        _transform_mnist.append(transforms.Resize(resize))
    _transform_mnist.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform_mnist = transforms.Compose(_transform_mnist)

    mnist_trainset = tv.datasets.MNIST(
        root=dataset_dir,
        train=True,
        download=False,
        transform=transform_mnist
    )
    mnist_testset = tv.datasets.MNIST(
        root=dataset_dir,
        train=False,
        download=False,
        transform=transform_mnist
    )
    return mnist_trainset, mnist_testset
