from torchvision import datasets, models, transforms
import torch
from collections import OrderedDict
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

def create_dataloaders(train_dir, valid_dir, test_dir):
    # Define your transforms for the training, validation, and testing sets
    data_transforms_training = transforms.Compose([transforms.RandomResizedCrop(224), 
                                                   transforms.RandomRotation(30),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])
                                                  ])
    data_transforms_validation = transforms.Compose([transforms.Resize(256),
                                                     transforms.CenterCrop(224), 
                                                     transforms.ToTensor(), 
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])
                                                    ])
    data_transforms_testing = transforms.Compose([transforms.Resize(256),
                                                  transforms.RandomResizedCrop(224), 
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225]),
                                                 ])


    # Datasets with ImageFolder
    image_training_datasets = datasets.ImageFolder(train_dir,transform=data_transforms_training)
    image_validation_datasets = datasets.ImageFolder(valid_dir,transform=data_transforms_validation)
    image_testing_datasets = datasets.ImageFolder(test_dir,transform=data_transforms_testing)

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders_training = torch.utils.data.DataLoader(image_training_datasets, batch_size=32, shuffle=True)
    dataloaders_validation = torch.utils.data.DataLoader(image_validation_datasets, batch_size=32, shuffle = True)
    dataloaders_testing = torch.utils.data.DataLoader(image_testing_datasets, batch_size=32, shuffle=True)
    
    return dataloaders_training, dataloaders_validation, dataloaders_testing, image_training_datasets

def train_network(model, epochs, optimizer, criterion, train_loader, valid_loader, learning_rate, gpu='False'):
    """
    Function takes model as input and train it with the given criterion , optimizor and learning_rate
    
    Args:
        model: Model to be trained
        epochs: No of times to iterate
        optimizer: Optimizer object based on adam algorithm
        criterion: Criterion
        train_loader: DataLoader for Training
        valid_loader: Data Loader for validation
        learning_rate: Learning Rate to be used
    
    Returns:
        None
    """
    
    valid_and_print = 30
    step_till_now = 0
    
    epoch = 0
    if gpu=='True' and torch.cuda.is_available(): 
        model.to('cuda')
    else:
        model.to('cpu')
        
    model.train()
    
    # Iterate through each epoch
    while epoch<= epochs:
        total_loss_run = 0
        for input_data, label_data in iter(train_loader):
            step_till_now += 1
            
            # Changing it to Variable and cuda if available
            input_data,label_data = get_variable(input_data, label_data, gpu)

            output_data = model.forward(input_data)
            loss = criterion(output_data, label_data)

            optimizer.zero_grad()

            # Backward in train phase
            loss.backward()
            optimizer.step()

            total_loss_run= loss.item() + total_loss_run

            if step_till_now % valid_and_print == 0:
                total_accuracy, loss_in_validation = validate_and_check(model, criterion, valid_loader, gpu)

                print("\nEpoch :{}/{} :- ".format(epoch+1, epochs))
                print("Loss In Training:- " + str(total_loss_run/valid_and_print), 
                      "Loss in Validation:- {}".format(loss_in_validation/valid_and_print),
                      "Accuracy in Validation:- {}".format(total_accuracy))
                total_loss_run = 0
        epoch+=1

def get_variable(input_data, label_data, gpu):
    """
    Function Wraps input data and label data in a Variable
    
    Args:
        input_data: Input Data
        label_data: Label Data
        gpu: Variable whether to use gpu or not
    
    Returns:
        input_data , variable_data wrapped in a Variable
    """
    input_data = Variable(input_data)
    label_data = Variable(label_data)

    if gpu and torch.cuda.is_available():
        input_data = Variable(input_data.float().cuda())
        label_data = Variable(label_data.long().cuda())
    return input_data, label_data

def validate_and_check(model, criterion, valid_loader, gpu):
    """
    Function Validate the model by using a validation data loafer
    
    Args:
        model : Model to validate
        criterion: Criterion
        valid_loader: Validation Data Loader
        gpu: Gpu use 
        
    Returns:
        Validation Accuracy, Validation Loss
    """
    model.eval()
    accuracy = 0
    loss = 0
    total = 0
    
    with torch.no_grad():
        for data in valid_loader:
            input_data , label_data = data
            input_data , label_data = get_variable(input_data, label_data, gpu)
                
            output_data = model(input_data)

            dummy,output_prediction = torch.max(output_data.data, 1)
    
            total+= label_data.size(0)
            status = label_data==output_prediction
            accuracy+= status.sum().item()
            loss += criterion(output_data, label_data).item()
        accuracy/=total
    return accuracy*100, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--gpu', help='Use GPU true/false', default="True")
    parser.add_argument('--hidden_layers', help='No. of Hidden Layers', default=4096)
    parser.add_argument('--arch', help="models to use (vgg19, densenet121)", default='vgg19')
    parser.add_argument('--epochs', help="Numer of epochs", default=10)
    parser.add_argument('--learning_rate', help="Learning rate to b used", default=0.001)
    args = parser.parse_args()
    gpu = args.gpu
    learning_rate = args.learning_rate
    hidden_layers = args.hidden_layers
    epochs = args.epochs
    model_use = args.arch
    
    data_dir = 'flowers/'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    dataloaders_training, dataloaders_validation, dataloaders_testing, image_training_datasets = create_dataloaders(train_dir,         valid_dir, test_dir)
    training_class_to_idx = image_training_datasets.class_to_idx
    
    # Using VGG 19 Model with pretrained
    if model_use == "vgg19":
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    elif model_use == "densenet121":
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Invalid Model")
        return
    
    # Defining classifier   
    model.classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(25088,int(hidden_layers)) ),
                                    ('fc2', nn.Linear(int(hidden_layers), 102)),
                                    ('relu', nn.ReLU()),
                                    ('output', nn.LogSoftmax(dim=1))
                                        ]))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=int(learning_rate))
    model.class_to_idx = training_class_to_idx
    epochs = epochs
    train_network(model, epochs, optimizer, criterion, dataloaders_training, dataloaders_validation,learning_rate, gpu)
    
    model.class_to_idx = image_training_datasets.class_to_idx

    model_checkpoint = {
        'batch_size': dataloaders_training.batch_size,
        'output_size': 102,
        'class_to_idx': model.class_to_idx,
        'model_epochs': epochs,
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': int(learning_rate),
        'model_state_dict': model.state_dict()
    }

    # Saving the checkpoint
    torch.save(model_checkpoint, 'model_checkpoint.pth')
    
if __name__ == "__main__":
    main()