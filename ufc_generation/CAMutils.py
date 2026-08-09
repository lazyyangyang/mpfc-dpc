import cv2
import numpy as np
import torch

def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
        
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    
    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()
    
    return heatmap, result


def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: Modified ResNet model
        target_layer_name (str): the name of layer with its hierarchical information.
            For example: 'layer4', 'layer1_bottleneck0', etc.
            
    Return:
        target_layer: found layer. This layer will be hooked to get forward/backward pass information.
    """
    # Ensure we're accessing the original model when using DataParallel
    if isinstance(arch, torch.nn.DataParallel):
        arch = arch.module  # Access the original model inside DataParallel

    # Print model architecture to identify correct layers
    print(arch)

    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        
        # Modify the layer check for ModifiedResNet
        if layer_num == 1:
            target_layer = arch.model.layer1  # layer1 in ModifiedResNet
        elif layer_num == 2:
            target_layer = arch.model.layer2
        elif layer_num == 3:
            target_layer = arch.model.layer3
        elif layer_num == 4:
            # Replace layer4 with the final layer you are using
            target_layer = arch.model.layer3  # Change this based on your architecture
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]
                
        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer






def find_efficientnet_layer(arch, target_layer_name):
    """Find EfficientNet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: EfficientNet_tiny_imagenet model
        target_layer_name (str): the name of the layer. Example: 'conv1', 'layers', 'linear'
            
    Return:
        target_layer: found layer. This layer will be hooked to get forward/backward pass information.
    """
    # Ensure we're accessing the original model if it's wrapped in DataParallel
    if isinstance(arch, torch.nn.DataParallel):
        arch = arch.module  # Access the original model inside DataParallel
    
    # If the target layer is 'conv1', return the first convolutional layer
    if target_layer_name == 'conv1':
        target_layer = arch.conv1

    # If the target layer is 'layers', return the sequential block layers
    elif target_layer_name == 'layers':
        target_layer = arch.layers

    # If the target layer is 'linear', return the final fully connected layer
    elif target_layer_name == 'linear':
        target_layer = arch.linear

    # If the target layer is inside the block layers, extract based on block and layer name
    elif 'layers' in target_layer_name:
        # Example: target_layer_name = 'layers_2'
        hierarchy = target_layer_name.split('_')
        if len(hierarchy) == 2 and hierarchy[0] == 'layers':
            block_idx = int(hierarchy[1])
            target_layer = arch.layers[block_idx]  # Return specific block layer

    else:
        raise ValueError(f"Unknown layer name: {target_layer_name}")

    return target_layer


def find_mobilenet_layer(arch, target_layer_name):
    """Find MobileNetV2 layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: MobileNetV2_tiny_imagenet model
        target_layer_name (str): the name of the layer. Example: 'conv1', 'layers', 'conv2', 'linear'
            
    Return:
        target_layer: found layer. This layer will be hooked to get forward/backward pass information.
    """
    # Ensure we're accessing the original model if it's wrapped in DataParallel
    if isinstance(arch, torch.nn.DataParallel):
        arch = arch.module  # Access the original model inside DataParallel

    # If the target layer is 'conv1', return the first convolutional layer
    if target_layer_name == 'conv1':
        target_layer = arch.conv1

    # If the target layer is 'layers', return the sequential block layers
    elif target_layer_name == 'layers':
        target_layer = arch.layers

    # If the target layer is 'conv2', return the second convolutional layer
    elif target_layer_name == 'conv2':
        target_layer = arch.conv2

    # If the target layer is 'linear', return the final fully connected layer
    elif target_layer_name == 'linear':
        target_layer = arch.linear

    # If the target layer is inside the layers blocks, extract based on block index
    elif 'layers' in target_layer_name:
        # Example: target_layer_name = 'layers_3'
        hierarchy = target_layer_name.split('_')
        if len(hierarchy) == 2 and hierarchy[0] == 'layers':
            block_idx = int(hierarchy[1])
            target_layer = arch.layers[block_idx]  # Return specific block layer

    else:
        raise ValueError(f"Unknown layer name: {target_layer_name}")

    return target_layer




def find_shufflenet_layer(arch, target_layer_name):
    """Find ShuffleNet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: ShuffleNet_tiny_imagenet model
        target_layer_name (str): the name of the layer. Example: 'conv1', 'layer1', 'layer2', 'layer3', 'linear'
            
    Return:
        target_layer: found layer. This layer will be hooked to get forward/backward pass information.
    """
    # Ensure we're accessing the original model if it's wrapped in DataParallel
    if isinstance(arch, torch.nn.DataParallel):
        arch = arch.module  # Access the original model inside DataParallel

    # If the target layer is 'conv1', return the first convolutional layer
    if target_layer_name == 'conv1':
        target_layer = arch.conv1

    # If the target layer is 'bn1', return the first batch normalization layer
    elif target_layer_name == 'bn1':
        target_layer = arch.bn1

    # If the target layer is 'layer1', return the first sequential layer
    elif target_layer_name == 'layer1':
        target_layer = arch.layer1

    # If the target layer is 'layer2', return the second sequential layer
    elif target_layer_name == 'layer2':
        target_layer = arch.layer2

    # If the target layer is 'layer3', return the third sequential layer
    elif target_layer_name == 'layer3':
        target_layer = arch.layer3

    # If the target layer is 'linear', return the final fully connected layer
    elif target_layer_name == 'linear':
        target_layer = arch.linear

    # If the target layer is inside the layers blocks, extract based on block index
    elif 'layer' in target_layer_name:
        # Example: target_layer_name = 'layer1_2'
        hierarchy = target_layer_name.split('_')
        if len(hierarchy) == 2 and hierarchy[0] == 'layer1':
            block_idx = int(hierarchy[1]) - 1  # Indexing is zero-based
            target_layer = arch.layer1[block_idx]  # Return specific block in layer1
        elif len(hierarchy) == 2 and hierarchy[0] == 'layer2':
            block_idx = int(hierarchy[1]) - 1
            target_layer = arch.layer2[block_idx]  # Return specific block in layer2
        elif len(hierarchy) == 2 and hierarchy[0] == 'layer3':
            block_idx = int(hierarchy[1]) - 1
            target_layer = arch.layer3[block_idx]  # Return specific block in layer3

    else:
        raise ValueError(f"Unknown layer name: {target_layer_name}")

    return target_layer
# def find_resnet_layer(arch, target_layer_name):
#     """Find resnet layer to calculate GradCAM and GradCAM++
    
#     Args:
#         arch: default torchvision densenet models
#         target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
#             target_layer_name = 'conv1'
#             target_layer_name = 'layer1'
#             target_layer_name = 'layer1_basicblock0'
#             target_layer_name = 'layer1_basicblock0_relu'
#             target_layer_name = 'layer1_bottleneck0'
#             target_layer_name = 'layer1_bottleneck0_conv1'
#             target_layer_name = 'layer1_bottleneck0_downsample'
#             target_layer_name = 'layer1_bottleneck0_downsample_0'
#             target_layer_name = 'avgpool'
#             target_layer_name = 'fc'
            
#     Return:
#         target_layer: found layer. this layer will be hooked to get forward/backward pass information.
#     """
#     if 'layer' in target_layer_name:
#         hierarchy = target_layer_name.split('_')
#         layer_num = int(hierarchy[0].lstrip('layer'))
#         if layer_num == 1:
#             target_layer = arch.layer1
#         elif layer_num == 2:
#             target_layer = arch.layer2
#         elif layer_num == 3:
#             target_layer = arch.layer3
#         elif layer_num == 4:
#             target_layer = arch.layer4
#         else:
#             raise ValueError('unknown layer : {}'.format(target_layer_name))

#         if len(hierarchy) >= 2:
#             bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
#             target_layer = target_layer[bottleneck_num]

#         if len(hierarchy) >= 3:
#             target_layer = target_layer._modules[hierarchy[2]]
                
#         if len(hierarchy) == 4:
#             target_layer = target_layer._modules[hierarchy[3]]

#     else:
#         target_layer = arch._modules[target_layer_name]

#     return target_layer


def find_densenet_layer(arch, target_layer_name):
    """Find densenet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer


def find_vgg_layer(arch, target_layer_name):
    """Find vgg layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_42'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_alexnet_layer(arch, target_layer_name):
    """Find alexnet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_0'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_squeezenet_layer(arch, target_layer_name):
    """Find squeezenet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features_12'
            target_layer_name = 'features_12_expand3x3'
            target_layer_name = 'features_12_expand3x3_activation'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2]+'_'+hierarchy[3]]

    return target_layer


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)
    
    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)
    
    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)