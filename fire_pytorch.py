import os
import numpy as np
from matplotlib import pyplot as plt
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader#, Dataset
import torchvision.transforms as transforms

from datasets import FireNetDataset
from models.firenet_pt import FireNet

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Seed
seed_val = 666
torch.manual_seed(seed_val)
np.random.seed(seed_val)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

must_train = True
must_test = True

def show_samples(data):
    fig = plt.figure()

    for i in range(len(data)):
        sample = train_data[i]

        print(i, sample[0].shape, sample[1].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        label = 'Fire' if sample[1] == 1 else 'Nofire'
        ax.set_title('Sample {}'.format(label))
        ax.axis('off')
        img = sample[0].transpose(2, 0)
        plt.imshow(img.transpose(0, 1))

        if i == 3:
            plt.show()
            break

img_dims = (64, 64)
model_name = 'model_firenet_pt_.pth'

#### Training
if must_train:
    dt = FireNetDataset(size=img_dims)
    x_train, y_train, x_val, y_val = dt.load_train_val()

    # Normalize data.
    x_train = dt.preprocess(x_train)
    x_val = dt.preprocess(x_val)

    # summary
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(y_train[y_train==1].shape[0], 'fire')
    print(y_train[y_train==0].shape[0], 'no_fire')

    print('x_val shape:', x_val.shape)
    print(x_val.shape[0], 'test samples')
    print(y_val[y_val==1].shape[0], 'fire')
    print(y_val[y_val==0].shape[0], 'no_fire')

    train_data = TensorDataset(torch.from_numpy(x_train.transpose((0, 3, 1, 2))), torch.from_numpy(y_train))
    val_data = TensorDataset(torch.from_numpy(x_val.transpose((0, 3, 1, 2))), torch.from_numpy(y_val))

    num_classes = len(dt.classes)
    input_shape = x_train.shape[1:]
    print('num_classes', num_classes, 'input_shape', input_shape)

    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            # download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                              shuffle=True, num_workers=4)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=32,
                                             shuffle=False, num_workers=2)

    def init_weights(m):
        print(m, isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear))
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)

    net = FireNet(num_classes)
    net.apply(init_weights)
    # exit()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, eps=1e-7)#, eps=None)
    # optimizer = optim.SparseAdam(net.parameters(), lr=0.001)#, eps=None)

    epochs = 100

    data_loaders = {"train": train_loader, "val": validation_loader}
    data_lengths = {"train": len(x_train), "val": len(x_val)}

    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0


    for epoch in range(epochs):
        print('Epoch {:03d}/{:03d}'.format(epoch +1, epochs), end=": ")

        # Each epoch has a training and validation phase
        epoch_time = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                # optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0.0

            # Iterate over data.
            for i, data in enumerate(data_loaders[phase]):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # statistics
                running_loss += loss.item()#data[0]
                # print('loss.item()', loss.item())
                # Accuracy
                # outputs = (outputs > 0.5).float()
                correct = (outputs.argmax(dim=1) == labels).float().sum()
                # correct = (outputs == labels).float().sum()
                running_acc += correct

            epoch_loss = running_loss / data_lengths[phase]
            epoch_acc = running_acc / data_lengths[phase]
            print('{} Loss: {:.4f}'.format(phase.capitalize(), epoch_loss), 'Acc: {:.4f}'.format(epoch_acc), end=" | ")

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())

        epoch_elapsed = time.time() - epoch_time
        print('time elapsed: {:.0f}m {:.0f}s'.format(
        epoch_elapsed // 60, epoch_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    torch.save(best_model_wts, 'models/saved/' + model_name)


### Test
if must_test:
    dt = FireNetDataset(size=img_dims)
    x_test, y_test = dt.load_test()

    # Normalize data.
    x_test = dt.preprocess(x_test)

    # summary
    print('x_test shape:', x_test.shape)
    print(x_test.shape[0], 'test samples')
    print(y_test[y_test==1].shape[0], 'fire')
    print(y_test[y_test==0].shape[0], 'no_fire')

    test_data = TensorDataset(torch.from_numpy(x_test.transpose((0, 3, 1, 2))), torch.from_numpy(y_test))

    num_classes = len(dt.classes)
    input_shape = x_test.shape[1:]
    print('num_classes', num_classes, 'input_shape', input_shape)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32,
                                              shuffle=False, num_workers=4)

    print('Evaluating model')
    net = FireNet(num_classes)
    net.load_state_dict(torch.load('models/saved/' + model_name))

    net.eval()  # Set model to evaluate mode

    total = 0
    correct = 0
    Y_test = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            Y_test.extend(labels)
            y_pred.extend(predicted)

    target_names = ['No Fire', 'Fire']
    # confusion = confusion_matrix(Y_test, y_pred)#, labels=target_names)
    # accuracy = (confusion[0][0] + confusion[1][1]) / np.sum(confusion)

    class_report = classification_report(Y_test, y_pred,
                            target_names=target_names)#, output_dict=True)

    print('Accuracy of the network on the test images: {:.2f}%'.format(
        100 * correct / total))

    # print('Confusion Matrix', confusion)
    print('Classification Report')
    print(class_report)
