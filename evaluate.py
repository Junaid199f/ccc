import os.path

from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import Dataset
import random
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
import utils
import json
import time



class Evaluate:
    def __init__(self, batch_size):
        self.dataset = Dataset()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.get_dataset(self.batch_size)
        self.lr = 0.001

    def train(self, model, epochs,hash_indv, warmup=False):
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.optimizer = optim.SGD(model.parameters(), lr=.01,
                      momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        model = model.cuda()
        #scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        # train_losslist = []
        # train_on_gpu = torch.cuda.is_available()
        # model = model.cuda()
        # # number of epochs to train the model
        # n_epochs = [*range(epochs)]  # you may increase this number to train a final model
        #
        # valid_loss_min = np.Inf  # track change in validation loss
        #
        # for epoch in n_epochs:
        #     # keep track of training and validation loss
        #     train_loss = 0.0
        #     valid_loss = 0.0
        #
        #     ###################
        #     # train the model #
        #     ###################
        #     model.train()
        #     for data, target in self.train_loader:
        #         # move tensors to GPU if CUDA is available
        #         if train_on_gpu:
        #             data, target = data.cuda(), target.cuda()
        #         # clear the gradients of all optimized variables
        #         self.optimizer.zero_grad()
        #         # forward pass: compute predicted outputs by passing inputs to the model
        #         output, x = model(data)
        #         # calculate the batch loss
        #         loss = self.criterion(output, target)
        #         # backward pass: compute gradient of the loss with respect to model parameters
        #         loss.backward()
        #         # perform a single optimization step (parameter update)
        #         self.optimizer.step()
        #         # update training loss
        #         train_loss += loss.item() * data.size(0)
        #
        #     ######################
        #     # validate the model #
        #     ######################
        #     model.eval()
        #     with torch.no_grad():
        #         for data, target in self.valid_loader:
        #             # move tensors to GPU if CUDA is available
        #             if train_on_gpu:
        #                 data, target = data.cuda(), target.cuda()
        #             # forward pass: compute predicted outputs by passing inputs to the model
        #             output, x = model(data)
        #             # calculate the batch loss
        #             loss = self.criterion(output, target)
        #             # update average validation loss
        #             valid_loss += loss.item() * data.size(0)
        #         # calculate average losses
        #         train_loss = train_loss / len(self.train_loader.dataset)
        #         valid_loss = valid_loss / len(self.valid_loader.dataset)
        #
        #         train_losslist.append(train_loss)
        #
        #         # print training/validation statistics
        #         print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        #             epoch, train_loss, valid_loss))
        #         # save model if validation loss has decreased
        #         if valid_loss <= valid_loss_min:
        #             print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        #                 valid_loss_min,
        #                 valid_loss))
        #             #torch.save(model.state_dict(), os.path.join(os.path.join(os.path.join(os.getcwd(),'checkpoints'),str(hash_indv)),'model_cifar.pt'))
        #             valid_loss_min = valid_loss
        #         #scheduler.step(valid_loss)
        # # track test loss
        # test_loss = 0.0
        # class_correct = list(0. for i in range(10))
        # class_total = list(0. for i in range(10))
        # model.eval()
        # # iterate over test data
        # with torch.no_grad():
        #     for data, target in self.test_loader:
        #         # move tensors to GPU if CUDA is available
        #         if train_on_gpu:
        #             data, target = data.cuda(), target.cuda()
        #         # forward pass: compute predicted outputs by passing inputs to the model
        #         output, x = model(data)
        #         # calculate the batch loss
        #         loss = self.criterion(output, target)
        #         # update test loss
        #         test_loss += loss.item() * data.size(0)
        #         # convert output probabilities to predicted class
        #         _, pred = torch.max(output, 1)
        #         # compare predictions to true label
        #         correct_tensor = pred.eq(target.data.view_as(pred))
        #         correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
        #             correct_tensor.cpu().numpy())
        #         # calculate test accuracy for each object class
        #         for i in range(16):
        #             label = target.data[i]
        #             class_correct[label] += correct[i].item()
        #             class_total[label] += 1
        #
        #     # average test loss
        # test_loss = test_loss / len(self.test_loader.dataset)
        # print('Test Loss: {:.6f}\n'.format(test_loss))
        #
        # for i in range(10):
        #     if class_total[i] > 0:
        #         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
        #             self.classes[i], 100 * class_correct[i] / class_total[i],
        #             np.sum(class_correct[i]), np.sum(class_total[i])))
        #     else:
        #         print('Test Accuracy of %5s: N/A (no training examples)' % (self.classes[i]))
        #
        # print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        #     100. * np.sum(class_correct) / np.sum(class_total),
        #     np.sum(class_correct), np.sum(class_total)))
        # loss = 100 - (100 * np.sum(class_correct) / np.sum(class_total))
        # #Saving data in Files
        #
        # results = {
        #     'train_loss': train_loss,
        #     'valid_loss': valid_loss,
        #     'test_loss': test_loss
        # }

        for epoch in range(epochs):
            print('\nEpoch: %d' % epoch)
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(),targets.cuda()
                self.optimizer.zero_grad()
                outputs,x = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                        inputs, targets = inputs.cuda(),targets.cuda()
                        outputs,x = model(inputs)
                        loss = self.criterion(outputs, targets)

                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                        #utils.progress_bar(batch_idx, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            # Save checkpoint.
        acc = 100. * correct / total
        # state = {
        #                 'net': model.state_dict(),
        #                 'acc': acc,
        #                 'epoch': epoch
        #     }
        loss = 100- acc
        # with open(os.path.join(os.path.join(os.path.join(os.getcwd(),'checkpoints'),str(hash_indv)),'output.json'), 'w') as json_file:
        #     json.dump(state, json_file)
        return loss