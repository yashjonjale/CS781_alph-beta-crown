"""
A simple example for bounding neural network outputs under input perturbations.

This example serves as a skeleton for robustness verification of neural networks using auto_LiRPA.
"""

import os
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm, PerturbationLpNormLocalised
from auto_LiRPA.utils import Flatten

## Step 1: Define the computational graph by implementing forward()
# This simple model is adapted from the repository:
# https://github.com/locuslab/convex_adversarial
def mnist_model():
    """
    Defines a simple Convolutional Neural Network (CNN) for MNIST classification.
    
    Architecture:
    - Conv2d: 1 input channel, 16 output channels, kernel size 4, stride 2, padding 1
    - ReLU activation
    - Conv2d: 16 input channels, 32 output channels, kernel size 4, stride 2, padding 1
    - ReLU activation
    - Flatten layer to convert 2D feature maps to 1D feature vectors
    - Linear layer: 32*7*7 inputs, 100 outputs
    - ReLU activation
    - Linear layer: 100 inputs, 10 outputs (number of classes)
    """
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),  # First convolutional layer
        nn.ReLU(),                                # Activation
        nn.Conv2d(16, 32, 4, stride=2, padding=1),# Second convolutional layer
        nn.ReLU(),                                # Activation
        Flatten(),                                # Flattening the output for the linear layers
        nn.Linear(32*7*7, 100),                   # First fully connected layer
        nn.ReLU(),                                # Activation
        nn.Linear(100, 10)                        # Output layer
    )
    return model

# Instantiate the model
model = mnist_model()

# Optionally, load the pretrained weights.
# Assumes that 'pretrained/mnist_a_adv.pth' exists relative to this script's directory.
checkpoint = torch.load(
    os.path.join(os.path.dirname(__file__), 'pretrained/mnist_a_adv.pth'),
    map_location=torch.device('cpu'))  # Load on CPU by default
model.load_state_dict(checkpoint)     # Load the state dictionary into the model

## Step 2: Prepare the dataset as usual
# Load the MNIST test dataset with normalization to tensor format
test_data = torchvision.datasets.MNIST(
    './data', train=False, download=True,
    transform=torchvision.transforms.ToTensor())

# For illustration, we only use a subset (first N images) from the dataset
N = 11  # Number of images to use
n_classes = 10  # Number of classes in MNIST
image = test_data.data[:N].view(N, 1, 28, 28)  # Select first N images and reshape
# image = torch.repeat_interleave(image, 3, dim=1)  # Convert to 3-channel format
true_label = test_data.targets[:N]             # Corresponding true labels

# Normalize the image data to [0, 1] by converting to float and dividing by 255
image = image.to(torch.float32) / 255.0

# If a CUDA-compatible GPU is available, move the data and model to GPU for faster computation
if torch.cuda.is_available():
    image = image.cuda()
    model = model.cuda()

## Step 3: Wrap the model with auto_LiRPA
# The second parameter is a sample input used for constructing the trace of the computational graph.
# Its actual content is not important; it's primarily used internally by auto_LiRPA.
lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
print('Running on', image.device)  # Display the device being used (CPU or CUDA)

## Step 4: Compute bounds using LiRPA given a perturbation
# Define the perturbation parameters
eps = 0.3  # Perturbation magnitude
norm = float("inf")  # Use L-infinity norm for perturbations
# norm = 2
window_size = 5
# Create a perturbation object representing L-infinity norm perturbations with epsilon=0.3
ptb = PerturbationLpNorm(norm=norm, eps=eps)
# ptb = PerturbationLpNormLocalised(norm=norm, window_size=window_size,eps=eps)

# Wrap the input image tensor with the defined perturbation
image = BoundedTensor(image, ptb)
# f_0(x_0):    2.652 <= f_0(x_0+delta) <=   10.009 (ground-truth)
# f_1(x_0):  -16.613 <= f_1(x_0+delta) <=   -5.235 
# f_2(x_0):   -5.368 <= f_2(x_0+delta) <=    2.556 
# f_3(x_0):   -6.717 <= f_3(x_0+delta) <=    0.039 
# f_4(x_0):  -11.043 <= f_4(x_0+delta) <=   -0.852 
# f_5(x_0):   -6.962 <= f_5(x_0+delta) <=    1.268 
# f_6(x_0):   -7.928 <= f_6(x_0+delta) <=    2.510 
# f_7(x_0):  -11.158 <= f_7(x_0+delta) <=   -0.859 
# f_8(x_0):   -6.262 <= f_8(x_0+delta) <=    0.471 
# f_9(x_0):   -9.429 <= f_9(x_0+delta) <=   -0.646 

# Perform a forward pass through the model using the bounded input tensor
pred = lirpa_model(image)

# Obtain the predicted labels by taking the argmax of the output logits
label = torch.argmax(pred, dim=1).cpu().detach().numpy()
print('Demonstration 1: Bound computation and comparisons of different methods.\n')

## Step 5: Compute bounds for the final output
# Iterate over different bounding methods to compute and display bounds
for method in ['forward', 'forward+backward', 'CROWN-Optimized (alpha-CROWN)']:
    print('Bounding method:', method)
    
    if 'Optimized' in method:
        # If using an optimized bounding method, set optimization options
        # These options can include the number of iterations, learning rate, etc.
        lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
    
    # Compute the lower and upper bounds using the specified method
    lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
    
    # Iterate over each image in the batch (here, only one image)
    for i in range(1):
        print(f'Image {i} top-1 prediction {label[i]} ground-truth {true_label[i]}')
        
        # Iterate over each class to display bounds for each output neuron
        for j in range(n_classes):
            # Mark the ground-truth class for reference
            indicator = '(ground-truth)' if j == true_label[i] else ''
            
            # Display the lower and upper bounds for the j-th output neuron
            print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
                j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
        print()  # Newline for better readability

# print('Demonstration 2: Obtaining linear coefficients of the lower and upper bounds.\n')

# # Initialize a dictionary to specify which linear coefficients (A matrices) to retrieve
# required_A = defaultdict(set)
# # Specify that we want the linear coefficients of the output node with respect to the input node
# required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])

# # Iterate over different bounding methods to compute and display linear coefficients
# for method in [
#         'IBP+backward (CROWN-IBP)', 
#         'backward (CROWN)', 
#         'CROWN',
#         'CROWN-Optimized (alpha-CROWN)'
#     ]:
#     print("Bounding method:", method)
    
#     if 'Optimized' in method:
#         # If using an optimized bounding method, set optimization options
#         lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
    
#     # Compute the bounds and retrieve the A matrices (linear coefficients) if needed_A is True
#     lb, ub, A_dict = lirpa_model.compute_bounds(
#         x=(image,), 
#         method=method.split()[0], 
#         return_A=True, 
#         needed_A_dict=required_A
#     )
    
#     # Extract the lower bound linear coefficients and bias terms
#     lower_A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA']
#     lower_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']
    
#     # Extract the upper bound linear coefficients and bias terms
#     upper_A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['uA']
#     upper_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['ubias']
    
#     # Display information about the lower bound linear coefficients
#     print(f'lower bound linear coefficients size (batch, output_dim, *input_dims): {list(lower_A.size())}')
#     print(f'lower bound linear coefficients norm (smaller is better): {lower_A.norm()}')
#     print(f'lower bound bias term size (batch, output_dim): {list(lower_bias.size())}')
#     print(f'lower bound bias term sum (larger is better): {lower_bias.sum()}')
    
#     # Display information about the upper bound linear coefficients
#     print(f'upper bound linear coefficients size (batch, output_dim, *input_dims): {list(upper_A.size())}')
#     print(f'upper bound linear coefficients norm (smaller is better): {upper_A.norm()}')
#     print(f'upper bound bias term size (batch, output_dim): {list(upper_bias.size())}')
#     print(f'upper bound bias term sum (smaller is better): {upper_bias.sum()}')
    
#     print(f'These linear lower and upper bounds are valid everywhere within the perturbation radii.\n')

# ## Demonstration 3: Computing margin bounds
# # In compute_bounds() function, you can pass a specification matrix C, which applies a final linear transformation
# # to the network's output. This is useful for computing margins between classes, which can yield tighter bounds.

# # Re-wrap the model as a BoundedModule (optional, as it's already wrapped earlier)
# lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)

# # Initialize the specification matrix C with zeros
# C = torch.zeros(size=(N, 1, n_classes), device=image.device)

# # Prepare ground truth labels for constructing the margin
# groundtruth = true_label.to(device=image.device).unsqueeze(1).unsqueeze(1)  # Shape: (N, 1, 1)

# # Define target labels as (groundtruth + 1) modulo number of classes to avoid out-of-range indices
# target_label = (groundtruth + 1) % n_classes

# # Set C such that it computes the margin: f_groundtruth - f_target
# C.scatter_(dim=2, index=groundtruth, value=1.0)   # Assign +1 to the ground-truth class
# C.scatter_(dim=2, index=target_label, value=-1.0) # Assign -1 to the target class

# print('Demonstration 3: Computing bounds with a specification matrix.\n')
# print('Specification matrix:\n', C)

# # Iterate over different bounding methods to compute margin bounds
# for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized (alpha-CROWN)']:
#     print('Bounding method:', method)
    
#     if 'Optimized' in method:
#         # If using an optimized bounding method, set optimization options
#         lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }})
    
#     # Compute the lower and upper bounds with the specification matrix C
#     lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
    
#     # Iterate over each image in the batch
#     for i in range(N):
#         print('Image {} top-1 prediction {} ground-truth {}'.format(i, label[i], true_label[i]))
        
#         # Display the computed margin bounds
#         print('margin bounds: {l:8.3f} <= f_{j}(x_0+delta) - f_{target}(x_0+delta) <= {u:8.3f}'.format(
#             j=true_label[i], 
#             target=(true_label[i] + 1) % n_classes, 
#             l=lb[i][0].item(), 
#             u=ub[i][0].item()))
#     print()
