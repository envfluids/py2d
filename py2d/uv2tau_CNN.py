import numpy as np
import torch
import torch.nn as nn
import jax.numpy as jnp

# Function to initialize the model
def init_model(model_type, model_path):
    # Define the ConvNeuralNet classes for each model type
    if model_type == '1M':
        class ConvNeuralNet(nn.Module):

            # Determine what layers and their order in CNN object
            def __init__(self, num_classes):

                    super(ConvNeuralNet, self).__init__()

                    self.conv_layer1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer11 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding="same")
                    self.relu1 = nn.ReLU()
                    
            def forward(self, x):

                    out = self.conv_layer1(x) # Layer1 (Input Layer)
                    out = self.relu1(out)

                    ## Hidden Layers
                    out = self.conv_layer2(out) #Layer2
                    out = self.relu1(out)

                    out = self.conv_layer3(out) #Layer3
                    out = self.relu1(out)

                    out = self.conv_layer4(out) #Layer4
                    out = self.relu1(out)

                    out = self.conv_layer5(out) #Layer5
                    out = self.relu1(out)

                    out = self.conv_layer6(out) #Layer6
                    out = self.relu1(out)

                    out = self.conv_layer7(out) #Layer7
                    out = self.relu1(out)

                    out = self.conv_layer8(out) #Layer8
                    out = self.relu1(out)

                    out = self.conv_layer9(out) #Layer9
                    out = self.relu1(out)

                    out = self.conv_layer10(out) #Layer10
                    out = self.relu1(out)

                    out = self.conv_layer11(out) #Layer11 (Output Layer)
                    return out
    elif model_type == '1M-PiOmega':
        class ConvNeuralNet(nn.Module):

            # Determine what layers and their order in CNN object
            def __init__(self, num_classes):

                    super(ConvNeuralNet, self).__init__()

                    self.conv_layer1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer11 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding="same")
                    self.relu1 = nn.ReLU()
                    
            def forward(self, x):

                    out = self.conv_layer1(x) # Layer1 (Input Layer)
                    out = self.relu1(out)

                    ## Hidden Layers
                    out = self.conv_layer2(out) #Layer2
                    out = self.relu1(out)

                    out = self.conv_layer3(out) #Layer3
                    out = self.relu1(out)

                    out = self.conv_layer4(out) #Layer4
                    out = self.relu1(out)

                    out = self.conv_layer5(out) #Layer5
                    out = self.relu1(out)

                    out = self.conv_layer6(out) #Layer6
                    out = self.relu1(out)

                    out = self.conv_layer7(out) #Layer7
                    out = self.relu1(out)

                    out = self.conv_layer8(out) #Layer8
                    out = self.relu1(out)

                    out = self.conv_layer9(out) #Layer9
                    out = self.relu1(out)

                    out = self.conv_layer10(out) #Layer10
                    out = self.relu1(out)

                    out = self.conv_layer11(out) #Layer11 (Output Layer)
                    return out     
    elif model_type == 'mcwiliams-ani':
        class ConvNeuralNet(nn.Module):

            # Determine what layers and their order in CNN object
            def __init__(self, num_classes):

                    super(ConvNeuralNet, self).__init__()

                    self.conv_layer1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer11 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=5, padding="same")
                    self.relu1 = nn.ReLU()
                    
            def forward(self, x):

                    out = self.conv_layer1(x) # Layer1 (Input Layer)
                    out = self.relu1(out)

                    ## Hidden Layers
                    out = self.conv_layer2(out) #Layer2
                    out = self.relu1(out)

                    out = self.conv_layer3(out) #Layer3
                    out = self.relu1(out)

                    out = self.conv_layer4(out) #Layer4
                    out = self.relu1(out)

                    out = self.conv_layer5(out) #Layer5
                    out = self.relu1(out)

                    out = self.conv_layer6(out) #Layer6
                    out = self.relu1(out)

                    out = self.conv_layer7(out) #Layer7
                    out = self.relu1(out)

                    out = self.conv_layer8(out) #Layer8
                    out = self.relu1(out)

                    out = self.conv_layer9(out) #Layer9
                    out = self.relu1(out)

                    out = self.conv_layer10(out) #Layer10
                    out = self.relu1(out)

                    out = self.conv_layer11(out) #Layer11 (Output Layer)
                    return out     

    elif model_type == 'mcwiliams':
        class ConvNeuralNet(nn.Module):

            # Determine what layers and their order in CNN object
            def __init__(self, num_classes):

                    super(ConvNeuralNet, self).__init__()

                    self.conv_layer1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, padding="same")
                    # self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    # self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    # self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    # self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    # self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    # self.conv_layer7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    # self.conv_layer8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    # self.conv_layer9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    # self.conv_layer10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer11 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding="same")
                    self.relu1 = nn.ReLU()
                    
            def forward(self, x):

                    out = self.conv_layer1(x) # Layer1 (Input Layer)
                    out = self.relu1(out)

                    ## Hidden Layers
                    # out = self.conv_layer2(out) #Layer2
                    # out = self.relu1(out)

                    # out = self.conv_layer3(out) #Layer3
                    # out = self.relu1(out)

                    # out = self.conv_layer4(out) #Layer4
                    # out = self.relu1(out)

                    # out = self.conv_layer5(out) #Layer5
                    # out = self.relu1(out)

                    # out = self.conv_layer6(out) #Layer6
                    # out = self.relu1(out)

                    # out = self.conv_layer7(out) #Layer7
                    # out = self.relu1(out)

                    # out = self.conv_layer8(out) #Layer8
                    # out = self.relu1(out)

                    # out = self.conv_layer9(out) #Layer9
                    # out = self.relu1(out)

                    # out = self.conv_layer10(out) #Layer10
                    # out = self.relu1(out)

                    out = self.conv_layer11(out) #Layer11 (Output Layer)
                    return out 
                    
    elif model_type == '8M':
        class ConvNeuralNet(nn.Module):
            # Determine what layers and their order in CNN object
            def __init__(self, num_classes):

                    super(ConvNeuralNet, self).__init__()

                    self.conv_layer1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=15, padding="same")
                    self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=15, padding="same")
                    self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=15, padding="same")
                    self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=15, padding="same")
                    self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=15, padding="same")
                    self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=15, padding="same")
                    self.conv_layer7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=15, padding="same")
                    self.conv_layer8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=15, padding="same")
                    self.conv_layer9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=15, padding="same")
                    self.conv_layer10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=15, padding="same")
                    self.conv_layer11 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=15, padding="same")
                    self.relu1 = nn.ReLU()
                    
            def forward(self, x):

                    out = self.conv_layer1(x) # Layer1 (Input Layer)
                    out = self.relu1(out)

                    ## Hidden Layers
                    out = self.conv_layer2(out) #Layer2
                    out = self.relu1(out)

                    out = self.conv_layer3(out) #Layer3
                    out = self.relu1(out)

                    out = self.conv_layer4(out) #Layer4
                    out = self.relu1(out)

                    out = self.conv_layer5(out) #Layer5
                    out = self.relu1(out)

                    out = self.conv_layer6(out) #Layer6
                    out = self.relu1(out)

                    out = self.conv_layer7(out) #Layer7
                    out = self.relu1(out)

                    out = self.conv_layer8(out) #Layer8
                    out = self.relu1(out)

                    out = self.conv_layer9(out) #Layer9
                    out = self.relu1(out)

                    out = self.conv_layer10(out) #Layer10
                    out = self.relu1(out)

                    out = self.conv_layer11(out) #Layer11 (Output Layer)
                    return out
            
    elif model_type == 'shallow':
        class ConvNeuralNet(nn.Module):
            # Determine what layers and their order in CNN object
            def __init__(self, num_classes):

                    super(ConvNeuralNet, self).__init__()

                    self.conv_layer1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="same")
                    self.conv_layer11 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding="same")
                    self.relu1 = nn.ReLU()
                    
            def forward(self, x):

                    out = self.conv_layer1(x) # Layer1 (Input Layer)
                    out = self.relu1(out)

                    ## Hidden Layers
                    out = self.conv_layer2(out) #Layer2
                    out = self.relu1(out)

                    # out = self.conv_layer3(out) #Layer3
                    # out = self.relu1(out)

                    # out = self.conv_layer4(out) #Layer4
                    # out = self.relu1(out)

                    # out = self.conv_layer5(out) #Layer5
                    # out = self.relu1(out)

                    # out = self.conv_layer6(out) #Layer6
                    # out = self.relu1(out)

                    # out = self.conv_layer7(out) #Layer7
                    # out = self.relu1(out)

                    # out = self.conv_layer8(out) #Layer8
                    # out = self.relu1(out)

                    # out = self.conv_layer9(out) #Layer9
                    # out = self.relu1(out)

                    # out = self.conv_layer10(out) #Layer10
                    # out = self.relu1(out)

                    out = self.conv_layer11(out) #Layer11 (Output Layer)
                    return out
    else:
        print('Invalid model type. Please choose from "deep" or "shallow"')

    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model instance and move it to the specified device
    model = ConvNeuralNet(1).to(device=device)

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract the model_state_dict from the checkpoint
    model_state_dict = checkpoint['model_state_dict']

    # Load the state dictionary into the model
    model.load_state_dict(model_state_dict)

    # Move the model to the specified device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    return model


def evaluate_model(model, input_tensor):
    NX = input_tensor.shape[1]
    NY = input_tensor.shape[2]
    # Set device to GPU if available, otherwise use CPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert the JAX array to a NumPy array, then to a PyTorch tensor
    # input_np = np.array(input_tensor)
    # input_tensor = torch.from_numpy(input_tensor).float()
    input_tensor = j2t(input_tensor).float()

    # Convert input_tensor to a torch Variable and move it to the specified device
    # input_var = torch.autograd.Variable(input_tensor).to(device)
    
    # Perform forward pass of the model with the input variable
    # output = model(input_var)
    output = model(input_tensor)
    # print('Output shape right after model: ', output.shape)
    # Convert the output tensor back to a JAX array
    # output_np = output.detach().cpu().numpy()
    # output_jax = jnp.array(output_np)
    # output = torch.concatenate([output, output, output], dim=0)
    # Flatten the output tensor
    output = output.view(-1)
    output_jax = t2j(output)
    # Reshape the output tensor to the original shape
    output_jax = output_jax.reshape((-1,NX,NY))


    return output_jax

import torch
import torch.utils.dlpack
import jax
import jax.dlpack
import jax.numpy as np
import numpy as nnp
import time

# A generic mechanism for turning a JAX function into a PyTorch function.

def j2t(x_jax):
  x_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x_jax))
  return x_torch

def t2j(x_torch):
  x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082
  x_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch))
  return x_jax



'''
------------------------------------------- Examples of how to use the functions above -------------------------------------------
model_path = "/media/volume/sdb/cgft/cgft_shallow/train_model_0b898_00008_8_batch_size_train=32,learning_rate=0.0000,p_data=50_2023-04-17_23-57-20/model.pt"
input_tensor = torch.randn(5, 2, 128, 128)  # Example input tensor, modify according to your model input requirements

model = init_model(model_type='1M', model_path=model_path) 
output = evaluate_model(model, input_tensor)

print(output.shape)
'''


