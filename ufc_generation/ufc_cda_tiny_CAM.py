import sys
sys.path.append("./models/")
from mobilenetv2 import MobileNetV2_tiny_imagenet,MobileNetV2_cifar10,MobileNetV2_cifar100
from efficientnet import EfficientNetB0_tiny_imagenet,EfficientNetB0_cifar10,EfficientNetB0_cifar100
from shufflenet import ShuffleNet_tiny_imagenet_G2,ShuffleNetG2_cifar10,ShuffleNetG2_cifar100
from resnet import ModifiedResNet

import os
import random
import argparse
import collections
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
from PIL import Image
from utils import BNFeatureHook, lr_cosine_policy, save_images, clip_image, denormalize_image
import wandb
from torchvision import transforms

#new
from CAMutils import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp
import torch.nn.functional as F
import torchvision.models as models
#new


def get_gradcam_mask(model, input, class_idx=None):
    gradcam = GradCAM(model_dict={'type': 'resnet', 'arch': model, 'layer_name': 'layer4'})

    # 打印输入张量的形状，确保其符合要求
    print(f"Input shape before GradCAM: {input.shape}")
    
    # 确保输入张量是四维的 (N, C, H, W)
    if input.dim() == 3:  # 如果是三维张量 (C, H, W)，添加批次维度
        input = input.unsqueeze(0)  # (1, C, H, W)
    elif input.dim() == 2:  # 如果是二维张量 (H, W)，添加批次和通道维度
        input = input.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # 打印修改后的输入张量的形状
    print(f"Input shape after modification: {input.shape}")
    
    # 获取 saliency_map 和 logit
    saliency_map, logit = gradcam(input, class_idx=class_idx)
    
    # 如果 saliency_map 为 None，打印调试信息并返回一个空的张量
    if saliency_map is None:
        print("Error: saliency_map is None!")
        return torch.zeros((1, 64, 64))  # 返回一个空的张量，避免后续错误

    print(f"Saliency map shape: {saliency_map.shape}")  # 打印 saliency_map 的形状

    # 确保 saliency_map 是四维张量，如果是二维的 (H, W)，需要添加通道维度
    if len(saliency_map.shape) == 2:  # 如果是二维张量 (H, W)，需要添加批次和通道维度
        saliency_map = saliency_map.unsqueeze(0).unsqueeze(0)  # 变成 (1, 1, H, W)
    
    print(f"After unsqueeze, saliency_map shape: {saliency_map.shape}")
    
    # 使用 F.interpolate 进行插值操作，确保输入和输出维度一致
    gradcam_mask = F.interpolate(saliency_map, size=(64, 64), mode='bilinear', align_corners=False)
    
    # 返回结果，移除批次维度并转为 numpy 数组
    return gradcam_mask.squeeze(0).cpu().numpy()  # 移除批次维度
# def get_gradcam_mask(model, input, class_idx=None):
#     gradcam = GradCAM(model_dict={'type': 'resnet', 'arch': model, 'layer_name': 'layer4'})
#     saliency_map, logit = gradcam(input, class_idx=class_idx)

#     # 添加通道维度（从 [1, 64, 64] 到 [1, 1, 64, 64]）
#     gradcam_mask = saliency_map.unsqueeze(1)  # 添加通道维度（1, 64, 64） -> （1, 1, 64, 64）

#     # 如果输入张量的形状是 [200, 1, 1, 64, 64]，移除第一个维度（批次维度）
#     gradcam_mask = gradcam_mask.squeeze(0)  # 变成 [200, 1, 64, 64] -> [200, 1, 64, 64]

#     # 确保输出为4D张量，即[1, channels, height, width]
#     gradcam_mask = gradcam_mask.view(1, -1, 64, 64)  # 这里的 -1 表示动态推断第二个维度的大小（200）

#     # 使用 F.interpolate 进行插值操作
#     gradcam_mask = F.interpolate(gradcam_mask, size=(64, 64), mode='bilinear', align_corners=False)

#     # 返回结果，移除批次维度后转为 numpy 数组
#     return gradcam_mask.squeeze(0).cpu().numpy()  # 移除批次维度

def get_images(args, model_lists, hook_for_display, ipc_id):
    print("get_images call")
    save_every = 100
    batch_size = args.batch_size
    best_cost = 1e4

    # 初始化 loss_packed_features 和 targets_all，只需要创建一次
    loss_packed_features = [
        [BNFeatureHook(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        for model in model_lists
    ]

    targets_all = torch.LongTensor(np.arange(200))

    init_path = args.init_path
    for kk in range(0, args.num_class, batch_size):
        targets = targets_all[kk : min(kk + batch_size, args.num_class)].to("cuda")

        model_index = max(ipc_id // args.ipc_init - 1, 0)
        model_teacher = model_lists[model_index]
        
        # 确保所有 ReLU 操作都不使用就地操作
        for name, layer in model_teacher.named_modules():
            if isinstance(layer, nn.ReLU):
                layer.inplace = False  # 禁用就地操作
            if isinstance(layer, nn.BatchNorm2d):
                # 禁用 BatchNorm 中的 running_var 和 running_mean 视图问题
                layer.track_running_stats = False
        
        # 禁用 DataParallel，如果多卡训练时出错，可以检查单卡
        if isinstance(model_teacher, torch.nn.DataParallel):
            model_teacher = model_teacher.module  # 只使用模型的原始部分
        
        # 更新 loss_packed_features 和 targets_all 为每个 batch 计算一次
        loss_r_feature_layers = loss_packed_features[model_index]

        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        
        # Prepare images
        image_tensors = []
        sorted_folders = sorted(os.listdir(init_path))

        for folder_index in range(kk, min(kk + batch_size, 200)):
            folder_path = os.path.join(init_path, sorted_folders[folder_index])
            image_files = sorted(os.listdir(folder_path))
            image_files = [f for f in image_files if os.path.isfile(os.path.join(folder_path, f))]
            
            image_path = os.path.join(folder_path, image_files[ipc_id % (args.ipc_init)])
            image = Image.open(image_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0)
            image_tensors.append(img_tensor)

        input_original = torch.cat(image_tensors, dim=0).to("cuda")
        # Ensure input_original is not a view, clone if necessary
        input_original = input_original.clone()  # Ensure it's not a view
        input_original.requires_grad_()  # Ensures the tensor requires gradients

        # Run Grad-CAM for the images
        gradcam_mask = get_gradcam_mask(model_teacher, input_original)
        
        # Ensure gradcam_mask is not a view, clone it before interpolation
        gradcam_mask = gradcam_mask.clone()  # Ensure it's not a view
        gradcam_mask = F.interpolate(gradcam_mask, size=(64, 64), mode='bilinear', align_corners=False)

        # Split input_original into key regions and non-key regions
        mask_threshold = gradcam_mask > gradcam_mask.mean()  # Binary mask: True for key regions, False for non-key regions
        key_region = mask_threshold * input_original  # Keep only key regions from input_original
        non_key_region = (~mask_threshold) * input_original  # Keep only non-key regions from input_original

        # Apply the regions to input_original (no need to optimize modified_input)
        key_region = key_region.clone() + 0.1  # Ensure no in-place modification
        non_key_region = non_key_region.clone() - 0.05  # Ensure no in-place modification

        # Combine the modified regions back into one tensor (modified_input)
        modified_input = (key_region + non_key_region).clone()  # Ensure it's a new tensor

        # Ensure the modified input tensor requires gradients (but will not be optimized)
        modified_input.requires_grad_()  # Make sure it's a leaf tensor but no optimization

        # Initialize uni_perb as a tensor to be optimized
        uni_perb = torch.zeros((1, 3, 64, 64), requires_grad=True, device="cuda", dtype=torch.float)

        # Now create final input by adding uni_perb to modified_input (this is the tensor we will optimize)
        final_input = modified_input + uni_perb
        final_input.requires_grad_()
        # Initialize optimizer to optimize only uni_perb
        optimizer = optim.Adam([uni_perb], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, args.iteration)

        for iteration in range(args.iteration):
            optimizer.zero_grad()
            # Forward pass through model to calculate loss
            outputs = model_teacher(final_input)

            # Classification loss
            loss_ce = nn.CrossEntropyLoss()(outputs, targets)

            # Feature loss (assuming loss_r_bn_feature)
            loss_r_bn_feature = torch.stack([mod.r_feature for mod in loss_r_feature_layers]).sum()

            # Total loss
            loss = loss_ce + args.r_bn * loss_r_bn_feature

            loss.backward(retain_graph=True)  # Retain the graph to avoid freeing the intermediate tensors
            optimizer.step()

            if iteration % save_every == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}")

            # Update best cost
            if best_cost > loss.item() or iteration == 1:
                best_inputs = final_input.data.clone()

        # Save images if required
        if args.store_best_images:
            best_inputs = denormalize_image(best_inputs, args.dataset)
            save_images(args, best_inputs, targets, ipc_id)

        # Clear optimizer states
        optimizer.state = collections.defaultdict(dict)

    torch.cuda.empty_cache()


def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                 std=[0.2302, 0.2265, 0.2262])
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        normalize,
    ])
    return preprocess(image)



def generation(args, ipc_id):
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    # prepare archs for UFC
    dataset_models = {
        'cifar10': {
            'num_classes': args.num_class,
            'model_types': [torchvision.models.resnet18, MobileNetV2_cifar10, EfficientNetB0_cifar10, ShuffleNetG2_cifar10],  
            'model_paths': [
                "pretrained/cifar-10/resnet18_E200/ckpt.pth",
                "pretrained/cifar-10/mobilenetV2_E200/ckpt.pth",
                "pretrained/cifar-10/efficientnet_E200/ckpt.pth",  
                "pretrained/cifar-10/shufflenet_E200/ckpt.pth"
            ]
        },
        'cifar100': {
            'num_classes': args.num_class,
            'model_types': [torchvision.models.resnet18, MobileNetV2_cifar100, EfficientNetB0_cifar100, ShuffleNetG2_cifar100],  
            'model_paths': [
                "pretrained/cifar-100/resnet18_E200/ckpt.pth",
                "pretrained/cifar-100/mobilenetV2_E200/ckpt.pth",
                "pretrained/cifar-100/efficientnet_E200/ckpt.pth", 
                "pretrained/cifar-100/shufflenet_E200/ckpt.pth"
            ]
        },
        # 'tiny': {
        #     'num_classes': args.num_class,
        #     'model_types': [torchvision.models.resnet18, MobileNetV2_tiny_imagenet, EfficientNetB0_tiny_imagenet],  
        #     'model_paths': [
        #         "pretrained/tiny/resnet18_E200/ckpt.pth",
        #         "pretrained/tiny/mobilenetV2_E200/ckpt.pth",
        #         "pretrained/tiny/efficientnet_E200/ckpt.pth", 
        #     ]
        # }
        'tiny': {
            'num_classes': args.num_class,
            'model_types': [torchvision.models.resnet18, MobileNetV2_tiny_imagenet, EfficientNetB0_tiny_imagenet, ShuffleNet_tiny_imagenet_G2],  
            'model_paths': [
                "pretrained/tiny/resnet18_E200/ckpt.pth",
                "pretrained/tiny/mobilenetV2_E200/ckpt.pth",
                "pretrained/tiny/efficientnet_E200/ckpt.pth",  
                "pretrained/tiny/shufflenet_E200/ckpt.pth"
            ]
        }
    }

    assert args.dataset in dataset_models, f"Unknown dataset: {args.dataset}"
    dataset_config = dataset_models[args.dataset]

    num_classes = dataset_config['num_classes']
    model_types = dataset_config['model_types']
    model_paths = dataset_config['model_paths']

    model_lists = []

    for model_type, model_path in zip(model_types, model_paths):

        if model_type == torchvision.models.resnet18:
            model_teacher = models.resnet18(pretrained=False, num_classes=num_classes)
            # 使用自定义的 ModifiedResNet 类来避免就地操作问题
            model_teacher = ModifiedResNet(model_teacher)  # 使用修改后的 ResNet 模型
            # 如果使用多卡训练，使用 DataParallel
            model_teacher = nn.DataParallel(model_teacher).cuda()
        elif model_type == torchvision.models.resnet34:
            model_teacher = model_type(num_classes=num_classes)
            model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model_teacher.maxpool = nn.Identity()
        # elif model_type == MobileNetV2_cifar100:
        #     model_teacher = MobileNetV2_cifar100()
        # elif model_type == EfficientNetB0_cifar100:
        #     model_teacher = EfficientNetB0_cifar100()
        # elif model_type == ShuffleNetG2_cifar100:
        #     model_teacher = ShuffleNetG2_cifar100()
        else:
            model_teacher = model_type()

        model_teacher = nn.DataParallel(model_teacher).cuda()
        checkpoint = torch.load(model_path)
        model_teacher.load_state_dict(checkpoint["state_dict"], strict=False)

        model_teacher.eval()
        if hasattr(model_teacher.module, '_fc'):  # 检查是否有 _fc 层
            model_teacher.module._fc = nn.Linear(model_teacher.module._fc.in_features, 200)
        for p in model_teacher.parameters():
            p.requires_grad = False

        model_lists.append(model_teacher)

    hook_for_display = None
    get_images(args, model_lists, hook_for_display, ipc_id)

def get_args():
    parser = argparse.ArgumentParser(description="UFC: Generate Inter-class Feature Compensator")

    # General settings
    parser.add_argument("--dataset", default="cifar100", type=str, choices=["cifar10", "cifar100", "tiny" , "imagenet"],
                        help="Dataset selection: cifar10, cifar100, or imagenet")
    parser.add_argument("--M", default=4, type=int, help="Number of architectures involved in UFC generation")
    parser.add_argument("--init_path", type=str, default="",
                        help="Path to the initial synthetic data")
    parser.add_argument("--ipc", type=int, default=10,
                        help="IPC (images per class) setting")
    # Data saving parameters
    parser.add_argument("--exp-name", type=str, default="generated_results",
                        help="Experiment name (subfolder under --syn-data-path)")
    parser.add_argument("--wandb-name", type=str, default="cifar100-ipc10")
    parser.add_argument("--syn-data-path", type=str, default="./syn_data",
                        help="Root directory for storing synthetic data")
    parser.add_argument("--store-best-images", action="store_true",
                        help="Flag to store the best-generated images")

    # Optimization parameters
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of images optimized simultaneously")
    parser.add_argument("--iteration", type=int, default=1000,
                        help="Number of iterations for optimization")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for optimization")
    parser.add_argument("--jitter", type=int, default=4,
                        help="Random shift applied to synthetic data for augmentation")
    parser.add_argument("--r-bn", type=float, default=0.05,
                        help="Coefficient for batch normalization (BN) feature distribution regularization")
    parser.add_argument("--first-bn-multiplier", type=float, default=10.0,
                        help="Additional multiplier for the first BN layer in R_bn")

    #new
    parser.add_argument("--easy2hard-mode", default="cosine", type=str, choices=["step", "linear", "cosine"])
    parser.add_argument("--milestone", default=0, type=float)

    # Parse arguments
    args = parser.parse_args()

    # Update syn_data_path to include experiment name
    args.syn_data_path = os.path.join(args.syn_data_path, args.wandb_name, args.exp_name)

    return args


if __name__ == "__main__":

    """
    args.ipc：表示每个类的图像数（IPC），即每类数据应该有多少图像（images per class）。

    args.M：表示涉及 UFC 生成的模型数量。

    args.num_class：表示数据集的类别数（例如，CIFAR-10 的类别数为 10，CIFAR-100 为 100，ImageNet 为 1000）。

    args.ipc_init：这个值的计算方式是：args.ipc / (args.M / args.num_class + 1)，这会将每个类别的图像数 (args.ipc) 分配到 args.M 个模型中，然后得到每个模型的图像数。

    args.ipc_end：ipc_end 是通过将 ipc_init 乘以 (args.M + 1) 来计算的。这可能意味着你希望为每个模型处理多个迭代步骤，或者在最终的计算中有所扩展。
    """
    args = get_args()

    if not wandb.api.api_key:
        wandb.login(key='')
    wandb.init(project='UFC-generation', name=args.wandb_name)
    global wandb_metrics
    wandb_metrics = {}
    args.ipc_start = 0
    if args.dataset =='cifar10':
        args.num_class = 10
    elif args.dataset =='cifar100':
        args.num_class = 100
    elif args.dataset =='imagenet':
        args.num_class = 1000
    elif args.dataset =='tiny':
        args.num_class = 200

    # averaging UFC for fair comparison
    args.ipc_init = int(args.ipc/(args.M/args.num_class + 1))
    args.ipc_end = args.ipc_init * (args.M + 1)
    print('ipc_end = ', args.ipc_end)

    for ipc_id in range(args.ipc_start, args.ipc_end):
        print("ipc = ", ipc_id)
        wandb.log({'ipc_id': ipc_id})
        generation(args, ipc_id)

    wandb.finish()