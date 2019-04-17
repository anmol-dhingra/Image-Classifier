import argparse
import json
from torchvision import datasets, models, transforms
import torch
from collections import OrderedDict
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def classifier(model):
    """
    Defining Classifier for the model_checkpoint.pth
    """
    
    model.classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(model.classifier[0].in_features, 4096)),
                                    ('fc2', nn.Linear(4096, 102)),
                                    ('relu', nn.ReLU()),
                                    ('output', nn.LogSoftmax(dim=1))
                                        ]))
    return model

def checkpoint_load(checkpoint_path, gpu):
    """
    Loading checkppoint based on the path given
    
    Args:
        checkpoint_path: Checkpoint Path 
        
    Returns:
        model:Model that is loaded
        class_to_idx = Class to Idx of the model
        
    """
    model_info = torch.load(checkpoint_path)
    model = models.vgg19(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idx = model_info['class_to_idx']

    model = classifier(model)
    model.load_state_dict(model_info["model_state_dict"])
    return model, model.class_to_idx

def process_image(image):
    """ 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    
    Args:
        image: Image
    
    Returns: 
        Numpy Image
    """
    
    my_image = Image.open(image)
    my_image.thumbnail((256,256))
    
    # Resizing Image to 224, 224
    my_image = my_image.resize((224,224))
    
    # Changing Imgae to Numpy array
    numpy_image = np.array(my_image)
    numpy_image = (numpy_image - numpy_image.mean() )/ numpy_image.std()
    numpy_image = numpy_image.transpose(2,0,1)
    return numpy_image

def predict(image_path, model_path, topk=5, gpu=False):
    """
    Predicting the image as per the image_path and model_path

    Args:
        image_path: Image Path
        model_path: Path of the model_checpoint
    
    Returns:
        Probability and Classes
    """
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        model, _ = checkpoint_load(model_path, gpu)
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topk)
        probs_list = probs[0].tolist()
        classes_list = classes[0].add(1).tolist()
    return probs_list, classes_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagepath', dest='imagepath', default=None)
    parser.add_argument('--json_path', help ="Category Name", default="cat_to_name.json")
    parser.add_argument('--gpu', help="GPU Usage True/False", default="True")
    parser.add_argument('--topk', help="Top K Classes", default=5)
    args = parser.parse_args()
    
    gpu= args.gpu
    with open(args.json_path, 'r') as f:
        cat_to_name = json.load(f)
    
    probs, classes = predict(args.imagepath, 'model_checkpoint.pth',args.topk, gpu)
    print("Probs: {}, Classes {}".format(probs, classes))
    
if __name__ == "__main__":
    main()


