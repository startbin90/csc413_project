
import numpy as np

from torch.autograd import Variable
import torch
import torchvision

from PIL import Image
from misc_functions import (get_example_params,
                            customPreProcessing,
                            convert_to_grayscale,
                            save_gradient_images)
from shutil import copyfile
                    
from vanilla_backprop import VanillaBackprop
from smooth_grad import generate_smooth_grad
from torchvision import models, transforms
TEST_DATASET_FILEPATH = '/Users/maxyou/Desktop/CSC413/project/pytorch-cnn-visualizations/test/'
LABELS_FILEPATH = '/Users/maxyou/Desktop/CSC413/project/pytorch-cnn-visualizations/ILSVRC2010_test_ground_truth.txt'

def generate_one_image_smooth_gradient(model_name, image_name, target_class, full_image_path, pretrain=True):
    if model_name == 'denseNet':
        pretrained_model = models.densenet121(pretrained=pretrain)
    elif model_name == "resNet": # res net
        pretrained_model = models.resnet18(pretrained=pretrain)

    original_image = Image.open(full_image_path).convert('RGB')
    prep_img = customPreProcessing(original_image)
    save_image_name = image_name + model_name + '_SM_pretrained'
    # save_image_name = image_name + model_name + '_SM_randomWeights'

    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model, _type=model_name)
    param_n = 50
    param_sigma_multiplier = 4
    smooth_grad = generate_smooth_grad(VBP,  # ^This parameter
                                       prep_img,
                                       target_class,
                                       param_n,
                                       param_sigma_multiplier)
    # Save colored gradients
    save_gradient_images(smooth_grad, save_image_name + '_colored')
    # Convert to grayscale
    grayscale_smooth_grad = convert_to_grayscale(smooth_grad)
    # Save grayscale gradients
    save_gradient_images(grayscale_smooth_grad, save_image_name)
    print('Smooth grad completed')

def generate_multiple_images_smooth_gradients(labels, images):
    for target_class, image_name in zip(labels, images):
        full_image_path = os.getcwd() + '\ILSVRC2012_DATA' + '\\val\\' + image_name
        generate_one_image_smooth_gradient('denseNet', image_name, target_class, full_image_path, pretrain=False)
        generate_one_image_smooth_gradient('resNet', image_name, target_class, full_image_path, pretrain=False)

def evaluate(model_name='denseNet', pretrain=True):

    images = os.listdir(os.getcwd()+'\ILSVRC2012_DATA'+'\\val')

    imageNet = torchvision.datasets.ImageFolder(os.getcwd()+'\ILSVRC2012_DATA', transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    # get the labels
    f = open(os.getcwd()+'\ILSVRC2012_validation_ground_truth.txt', "r")
    labels = torch.tensor(np.array([int(label[:-1]) for label in f.readlines()]))
    f.close()
    imageNet_loader = torch.utils.data.DataLoader(imageNet, batch_size=100)
    if model_name == 'denseNet':
        pretrained_model = models.densenet121(pretrained=pretrain)
    elif model_name == 'resNet': # res net
        pretrained_model = models.resnet18(pretrained=pretrain)

    if torch.cuda.is_available():
        pretrained_model.to('cuda')
        labels = labels.to('cuda')

    subset_start = 0
    subset_end = 100
    total_corrects = 0
    correct_prediction_images = []
    ith_batch = 1
    for input_batch in imageNet_loader:
        batch_data = input_batch[0].to('cuda')
        with torch.no_grad(): 
            outputs = pretrained_model(batch_data)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        elementwise_comparison = predictions == labels[subset_start:subset_end]
        batch_corrects = torch.sum(elementwise_comparison).item()
        total_corrects += batch_corrects
        print('batch {} corrects: {}'.format(ith_batch, total_corrects))
        
        if batch_corrects > 0:
            nplist = elementwise_comparison.cpu().numpy()
            correct_prediction_indices = np.where(nplist > 0)[0]
            # get the image name and save it in the folder
            correct_prediction_images += [images[subset_start + index] for index in correct_prediction_indices]
            # save it for future use
            with open('{}_correct_predictions.pkl'.format(model_name), 'wb') as f:
                pickle.dump(correct_prediction_images, f)
            print(correct_prediction_images)

        subset_start = subset_end
        subset_end += batch_data.shape[0]
        ith_batch += 1


def retrieve_images(image_names, model_name):
    for img_name in image_names:
        src = os.getcwd() + '\ILSVRC2012_DATA' + '\\val\\' + img_name
        dst = os.getcwd() + '\\results\project_results\{}_correct_original\\'.format(model_name) + img_name
        copyfile(src, dst)

if __name__ == '__main__':
    import os, stat
    import pickle
    cwd = os.getcwd()
    # os.chdir(cwd + '/src')
    # evaluate()

    with open('denseNet_correct_predictions.pkl', 'rb') as f:
        denseNet_corrects = pickle.load(f)

    with open('resNet_correct_predictions.pkl', 'rb') as f:
        resNet_corrects = pickle.load(f)

    # resnet correct but not densenet
    resnet_win = np.array(list(set(resNet_corrects) - set(denseNet_corrects)))
    densenet_win = np.array(list(set(denseNet_corrects) - set(resNet_corrects)))
    retrieve_images(resnet_win, 'resNet')
    retrieve_images(densenet_win, 'denseNet')

    # get all labels
    with open(os.getcwd() + '\ILSVRC2012_validation_ground_truth.txt', "r") as f:
        all_labels = np.array([int(label[:-1]) for label in f.readlines()])

    # get all the images
    all_images = np.array(os.listdir(os.getcwd()+'\ILSVRC2012_DATA'+'\\val'))

    # find the corresponding indices or labels that resnet win and denseNet win
    resnet_win_labels = list(all_labels[np.searchsorted(all_images, resnet_win)])
    denseNet_win_labels = list(all_labels[np.searchsorted(all_images, densenet_win)])

    # generate_multiple_images_smooth_gradients(resnet_win_labels, resnet_win)
    generate_multiple_images_smooth_gradients(denseNet_win_labels, densenet_win)

    # ** do a experiment make sure that the model is variatant of the gradient final results
