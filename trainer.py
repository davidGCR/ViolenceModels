import torch
import torchvision
from tensorboardcolab import TensorBoardColab
import time
import copy

class Trainer:
    def __init__(self,model,dataloaders,criterion,optimizer):
        self.model = model
        # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
        self.model_name = "alexnet"
        # Number of classes in the dataset
        self.num_classes = 2
        # Batch size for training (change depending on how much memory you have)
        self.batch_size = 64
        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        self.feature_extract = True

        self.input_size = 224
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.tb = TensorBoardColab()
    
    def train_epoch(self,epoch):
        epoch_acc_train = 0.0
        epoch_loss_train = 0.0
        epoch_acc_test = 0.0
        epoch_loss_test = 0.0
        
        self.model.train()  # Set model to training mode
           
        running_loss = 0.0
        running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders['train']:
                inputs = inputs.permute(1, 0, 2, 3, 4)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception:
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = self.model(inputs)
                        loss1 = self.criterion(outputs, labels)
                        loss2 = self.criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
        
                    loss.backward()
                    self.optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders['train'].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            self.tb.save_value('trainLoss', 'train_loss', epoch, epoch_loss)
            self.tb.save_value('trainAcc', 'train_acc', epoch, epoch_acc)
        return epoch_acc, epoch_loss


    def test_epoch(self,epoch):
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for inputs, labels in dataloaders['test']:
            inputs = inputs.permute(1, 0, 2, 3, 4)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders['test'].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders['test'].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))
            self.tb.save_value('testLoss', 'test_loss', epoch, epoch_loss)
            self.tb.save_value('testAcc', 'test_acc', epoch, epoch_acc)
            

        return epoch_loss, epoch_acc
