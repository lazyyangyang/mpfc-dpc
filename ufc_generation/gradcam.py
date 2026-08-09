import torch
import torch.nn.functional as F

from CAMutils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer

class GradCAM(object):
    """Calculate GradCAM saliency map."""

    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()

        # 定义梯度钩子
        def backward_hook(module, grad_input, grad_output):
            """保存梯度"""
            self.gradients['value'] = grad_output[0]
            print(f"Backward hook triggered! Gradients saved: {self.gradients['value'].shape}")  # 打印梯度的形状
            return None

        # 定义前向钩子
        def forward_hook(module, input, output):
            """保存激活值"""
            self.activations['value'] = output
            print(f"Forward hook triggered! Activations saved: {self.activations['value'].shape}")  # 打印激活值的形状
            return None

        # 选择对应的层
        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)

        print(f"Target layer: {target_layer}")  # 打印目标层，确认是否正确选择

        # 注册前向钩子和反向钩子
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        # 打印 saliency_map 的大小
        
        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                print('saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (batch_size, 3, H, W)
            class_idx (int or list of ints): class index for calculating GradCAM.
                If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        # 切换到训练模式，确保梯度计算正常
        self.model_arch.train()  # 强制进入训练模式

        # 执行前向传播
        logit = self.model_arch(input)

        # 如果没有提供 class_idx，则使用模型预测得分最高的类别
        if class_idx is None:
            class_idx = logit.max(1)[-1]  # 获取每个样本的最高预测类别的索引

        # 获取目标类别的预测得分
        score = logit[range(b), class_idx]  # 提取每个样本对应的类别得分

        # 确保 score 是一个标量，用于反向传播
        if score.dim() != 0:
            score = score.mean()  # 如果有多个类别，取平均

        # 执行反向传播计算梯度
        self.model_arch.zero_grad()  # 清除之前的梯度
        print(f"Score before backward: {score}")  # 打印 score，确保它有效
        score.backward(retain_graph=retain_graph)  # 保持计算图，以便后续使用

        # 检查 gradients['value'] 是否存在
        if 'value' not in self.gradients:
            print("Error: Gradients not found!")
            return None, logit  # 提供错误信息，防止后续出错

        # 获取梯度和激活值
        gradients = self.gradients['value']
        activations = self.activations['value']

        # 打印梯度和激活值
        print(f"Gradients shape: {gradients.shape}")
        print(f"Activations shape: {activations.shape}")

        # 计算权重和 saliency map
        b, k, u, v = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)  # 计算每个通道的权重
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)  # 计算 ReLU 激活
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        # 归一化 saliency map
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    # 添加 __call__ 方法，使得 GradCAM 实例可以像函数一样被调用
    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
# class GradCAM(object):
#     """Calculate GradCAM saliency map."""

#     def __init__(self, model_dict, verbose=False):
#         model_type = model_dict['type']
#         layer_name = model_dict['layer_name']
#         self.model_arch = model_dict['arch']

#         self.gradients = dict()
#         self.activations = dict()

#         def backward_hook(module, grad_input, grad_output):
#             """保存梯度"""
#             self.gradients['value'] = grad_output[0]
#             return None

#         def forward_hook(module, input, output):
#             """保存激活值"""
#             self.activations['value'] = output
#             return None

#         # 选择对应的层
#         if 'vgg' in model_type.lower():
#             target_layer = find_vgg_layer(self.model_arch, layer_name)
#         elif 'resnet' in model_type.lower():
#             target_layer = find_resnet_layer(self.model_arch, layer_name)
#         elif 'densenet' in model_type.lower():
#             target_layer = find_densenet_layer(self.model_arch, layer_name)
#         elif 'alexnet' in model_type.lower():
#             target_layer = find_alexnet_layer(self.model_arch, layer_name)
#         elif 'squeezenet' in model_type.lower():
#             target_layer = find_squeezenet_layer(self.model_arch, layer_name)

#         # 注册钩子
#         target_layer.register_forward_hook(forward_hook)
#         target_layer.register_backward_hook(backward_hook)

#         # 打印 saliency_map 的大小
#         if verbose:
#             try:
#                 input_size = model_dict['input_size']
#             except KeyError:
#                 print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
#             else:
#                 device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
#                 self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
#                 print('saliency_map size :', self.activations['value'].shape[2:])

#     def forward(self, input, class_idx=None, retain_graph=False):
#         """
#         Args:
#             input: input image with shape of (batch_size, 3, H, W)
#             class_idx (int or list of ints): class index for calculating GradCAM.
#                 If not specified, the class index that makes the highest model prediction score will be used.
#         Return:
#             mask: saliency map of the same spatial dimension with input
#             logit: model output
#         """
#         b, c, h, w = input.size()

#         # 执行前向传播
#         logit = self.model_arch(input)

#         # 如果没有提供 class_idx，则使用模型预测最高得分的类别
#         if class_idx is None:
#             class_idx = logit.max(1)[-1]  # 获取每个样本的最高预测类别的索引

#         # 获取目标类别的预测得分
#         score = logit[range(b), class_idx]  # 提取每个样本对应的类别得分

#         # 确保 score 是一个标量，用于反向传播
#         if score.dim() != 0:
#             score = score.mean()  # 如果有多个类别，取平均

#         # 进行反向传播来计算梯度
#         self.model_arch.zero_grad()  # 清除之前的梯度
#         score.backward(retain_graph=retain_graph)

#         # 获取梯度和激活值
#         gradients = self.gradients['value']
#         activations = self.activations['value']

#         # 计算权重和 saliency map
#         b, k, u, v = gradients.size()
#         alpha = gradients.view(b, k, -1).mean(2)  # 计算每个通道的权重
#         weights = alpha.view(b, k, 1, 1)

#         saliency_map = (weights * activations).sum(1, keepdim=True)
#         saliency_map = F.relu(saliency_map)  # 计算 ReLU 激活
#         saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

#         # 归一化 saliency map
#         saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
#         saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

#         return saliency_map, logit
    

#     def __call__(self, input, class_idx=None, retain_graph=False):
#         return self.forward(input, class_idx, retain_graph)

class GradCAMpp(GradCAM):
    """Calculate GradCAM++ saliency map."""

    def __init__(self, model_dict, verbose=False):
        super(GradCAMpp, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (batch_size, 3, H, W)
            class_idx (int or list of ints): class index for calculating GradCAM.
                If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()
    
        # Perform forward pass through the model
        logit = self.model_arch(input)
    
        # If class_idx is not specified, use the index of the class with the highest score
        if class_idx is None:
            # Get the index of the highest predicted class for each sample in the batch
            class_idx = logit.max(1)[-1]  # This returns a tensor with the index of the highest class per sample in the batch
    
        # Now, we select the score for the specified class or highest class
        score = logit[range(b), class_idx]  # Extract scores for the selected class indices from each sample in the batch
    
        # Ensure that score is a scalar for backward
        if score.dim() != 0:
            score = score.mean()  # Take the mean of the scores to make it a scalar
    
        # Perform backward pass to calculate gradients
        self.model_arch.zero_grad()  # Ensure gradients are zeroed
        score.backward(retain_graph=retain_graph)
    
        # Get gradients and activations
        gradients = self.gradients['value']
        activations = self.activations['value']
    
        # Compute the weights and saliency map
        b, k, u, v = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
    
        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
    
        # Normalize saliency map
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
    
        return saliency_map, logit
# class GradCAMpp(GradCAM):
#     """Calculate GradCAM++ salinecy map.

#     A simple example:

#         # initialize a model, model_dict and gradcampp
#         resnet = torchvision.models.resnet101(pretrained=True)
#         resnet.eval()
#         model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
#         gradcampp = GradCAMpp(model_dict)

#         # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
#         img = load_img()
#         normed_img = normalizer(img)

#         # get a GradCAM saliency map on the class index 10.
#         mask, logit = gradcampp(normed_img, class_idx=10)

#         # make heatmap from mask and synthesize saliency map using heatmap and img
#         heatmap, cam_result = visualize_cam(mask, img)


#     Args:
#         model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
#         verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
#     """
#     def __init__(self, model_dict, verbose=False):
#         super(GradCAMpp, self).__init__(model_dict, verbose)

#     def forward(self, input, class_idx=None, retain_graph=False):
#         """
#         Args:
#             input: input image with shape of (batch_size, 3, H, W)
#             class_idx (int or list of ints): class index for calculating GradCAM.
#                 If not specified, the class index that makes the highest model prediction score will be used.
#         Return:
#             mask: saliency map of the same spatial dimension with input
#             logit: model output
#         """
#         b, c, h, w = input.size()
    
#         # Perform forward pass through the model
#         logit = self.model_arch(input)
    
#         # If class_idx is not specified, use the index of the class with the highest score
#         if class_idx is None:
#             # Get the index of the highest predicted class for each sample in the batch
#             class_idx = logit.max(1)[-1]  # This returns a tensor with the index of the highest class per sample in the batch
    
#         # Now, we select the score for the specified class or highest class
#         score = logit[range(b), class_idx]  # Extract scores for the selected class indices from each sample in the batch
    
#         # Ensure that score is a scalar for backward
#         if score.dim() != 0:
#             score = score.mean()  # Take the mean of the scores to make it a scalar
    
#         # Perform backward pass to calculate gradients
#         self.model_arch.zero_grad()  # Ensure gradients are zeroed
#         score.backward(retain_graph=retain_graph)
    
#         # Get gradients and activations
#         gradients = self.gradients['value']
#         activations = self.activations['value']
    
#         # Compute the weights and saliency map
#         b, k, u, v = gradients.size()
#         alpha = gradients.view(b, k, -1).mean(2)
#         weights = alpha.view(b, k, 1, 1)
    
#         saliency_map = (weights * activations).sum(1, keepdim=True)
#         saliency_map = F.relu(saliency_map)
#         saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
    
#         # Normalize saliency map
#         saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
#         saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
    
#         return saliency_map, logit
   
