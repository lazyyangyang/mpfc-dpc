import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModifiedResNet(nn.Module):
    def __init__(self, model):
        super(ModifiedResNet, self).__init__()
        self.model = model

        # 禁用所有 ReLU 层的就地操作
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False  # 禁用 ReLU 的就地操作
            if isinstance(module, nn.Conv2d):
                module.register_backward_hook(self.backward_hook)  # 注册反向传播钩子
                module.register_forward_hook(self.forward_hook)    # 注册前向传播钩子

        # 1x1 卷积用于调整形状
        self.conv1x1_layer1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv1x1_layer2 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)  # 用于调整 layer2 的形状
        self.conv1x1_layer3 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)  # 用于调整 layer3 的形状
        self.conv1x1_layer4 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)  # 用于调整 layer4 的形状

    def forward(self, x):
        out = self.model.conv1(x)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        print(f"Shape after maxpool: {out.shape}")  # 打印每层输出的形状

        # 残差连接：处理 layer1
        for layer in self.model.layer1:
            identity = out
            out = layer(out)
            print(f"Shape after layer1: {out.shape}, identity: {identity.shape}")  # 打印 layer1 输出形状
            if out.size() != identity.size():
                identity = self.conv1x1_layer1(identity)  # 使用 1x1 卷积调整形状
                print(f"Shape after conv1x1 (identity): {identity.shape}")  # 打印卷积调整后的形状
            out = out + identity  # 使用非就地加法

        # 残差连接：处理 layer2
        for layer in self.model.layer2:
            identity = out
            out = layer(out)
            print(f"Shape after layer2: {out.shape}, identity: {identity.shape}")  # 打印 layer2 输出形状
            if out.size() != identity.size():
                identity = self.conv1x1_layer2(identity)  # 使用 1x1 卷积调整形状
                print(f"Shape after conv1x1 (identity): {identity.shape}")  # 打印卷积调整后的形状
            out = out + identity  # 使用非就地加法

        # 残差连接：处理 layer3
        for layer in self.model.layer3:
            identity = out
            out = layer(out)
            print(f"Shape after layer3: {out.shape}, identity: {identity.shape}")  # 打印 layer3 输出形状
            if out.size() != identity.size():
                identity = self.conv1x1_layer3(identity)  # 使用 1x1 卷积调整形状
                print(f"Shape after conv1x1 (identity): {identity.shape}")  # 打印卷积调整后的形状
            out = out + identity  # 使用非就地加法

        # 残差连接：处理 layer4
        for layer in self.model.layer4:
            identity = out
            out = layer(out)
            print(f"Shape after layer4: {out.shape}, identity: {identity.shape}")  # 打印 layer4 输出形状
            if out.size() != identity.size():
                identity = self.conv1x1_layer4(identity)  # 使用 1x1 卷积调整形状
                print(f"Shape after conv1x1 (identity): {identity.shape}")  # 打印卷积调整后的形状
            out = out + identity  # 使用非就地加法

        # 其他层
        out = self.model.avgpool(out)
        print(f"Shape after avgpool: {out.shape}")  # 打印 avgpool 输出形状
        out = torch.flatten(out, 1)
        out = self.model.fc(out)
        print(f"Shape after fc: {out.shape}")  # 打印 fc 输出形状

        return out
        def backward_hook(self, module, grad_input, grad_output):
            """保存梯度"""
            self.gradients = grad_output[0]
            print(f"Backward hook triggered! Gradients saved: {self.gradients.shape}")
            return None

        def forward_hook(self, module, input, output):
            """保存激活值"""
            self.activations = output
            print(f"Forward hook triggered! Activations saved: {self.activations.shape}")
            return None
# class ModifiedResNet(nn.Module):
#     def __init__(self, model):
#         super(ModifiedResNet, self).__init__()
#         self.model = model
        
#         # 禁用所有 ReLU 层的就地操作
#         for name, module in self.model.named_modules():
#             if isinstance(module, nn.ReLU):
#                 module.inplace = False  # 禁用 ReLU 的就地操作

#     def forward(self, x):
#         out = self.model.conv1(x)
#         out = self.model.bn1(out)
#         out = self.model.relu(out)
#         out = self.model.maxpool(out)
    
#         # 残差连接：处理 layer1
#         for layer in self.model.layer1:
#             identity = out
#             out = layer(out)
#             if out.size() != identity.size():
#                 # 使用 1x1 卷积调整 shape，确保通道数一致
#                 identity = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)(identity)  # 输入和输出通道数均为 64
#             out = out + identity  # 使用非就地加法
    
#         # 残差连接：处理 layer2
#         for layer in self.model.layer2:
#             identity = out
#             out = layer(out)
#             if out.size() != identity.size():
#                 # 使用 1x1 卷积调整 shape，确保通道数一致
#                 identity = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)(identity)  # 输入和输出通道数均为 128
#             out = out + identity  # 使用非就地加法
    
#         # 残差连接：处理 layer3
#         for layer in self.model.layer3:
#             identity = out
#             out = layer(out)
#             if out.size() != identity.size():
#                 # 使用 1x1 卷积调整 shape，确保通道数一致
#                 identity = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)(identity)  # 输入和输出通道数均为 256
#             out = out + identity  # 使用非就地加法
    
#         # 残差连接：处理 layer4
#         for layer in self.model.layer4:
#             identity = out
#             out = layer(out)
#             if out.size() != identity.size():
#                 # 使用 1x1 卷积调整 shape，确保通道数一致
#                 identity = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)(identity)  # 输入和输出通道数均为 512
#             out = out + identity  # 使用非就地加法
    
#         # 其他层
#         out = self.model.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.model.fc(out)
    
#         return out
# class ModifiedResNet(nn.Module):
#     def __init__(self, model):
#         super(ModifiedResNet, self).__init__()
#         self.model = model
        
#         # 禁用所有 ReLU 层的就地操作
#         for name, module in self.model.named_modules():
#             if isinstance(module, nn.ReLU):
#                 module.inplace = False  # 禁用 ReLU 的就地操作

#         # 1x1 卷积用于调整形状
#         # 修改 conv1x1，使其输入通道数为 64，输出通道数为 128（调整输入输出通道数）
#         self.conv1x1 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)

#     def forward(self, x):
#         out = self.model.conv1(x)
#         out = self.model.bn1(out)
#         out = self.model.relu(out)
#         out = self.model.maxpool(out)
    
#         # 残差连接：处理 layer1
#         for layer in self.model.layer1:
#             identity = out
#             out = layer(out)
#             if out.size() != identity.size():
#                 identity = self.conv1x1(identity)  # 使用 1x1 卷积调整形状
#             out = out + identity  # 使用非就地加法
    
#         # 残差连接：处理 layer2
#         for layer in self.model.layer2:
#             identity = out
#             out = layer(out)
#             if out.size() != identity.size():
#                 identity = self.conv1x1(identity)  # 使用 1x1 卷积调整形状
#             out = out + identity  # 使用非就地加法
    
#         # 残差连接：处理 layer3
#         for layer in self.model.layer3:
#             identity = out
#             out = layer(out)
#             if out.size() != identity.size():
#                 identity = self.conv1x1(identity)  # 使用 1x1 卷积调整形状
#             out = out + identity  # 使用非就地加法
    
#         # 残差连接：处理 layer4
#         for layer in self.model.layer4:
#             identity = out
#             out = layer(out)
#             if out.size() != identity.size():
#                 identity = self.conv1x1(identity)  # 使用 1x1 卷积调整形状
#             out = out + identity  # 使用非就地加法
    
#         # 其他层
#         out = self.model.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.model.fc(out)
    
#         return out
