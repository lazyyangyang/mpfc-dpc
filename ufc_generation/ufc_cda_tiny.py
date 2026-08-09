import sys
sys.path.append("./models/")
from mobilenetv2 import MobileNetV2_tiny_imagenet,MobileNetV2_cifar10,MobileNetV2_cifar100
from efficientnet import EfficientNetB0_tiny_imagenet,EfficientNetB0_cifar10,EfficientNetB0_cifar100
from shufflenet import ShuffleNet_tiny_imagenet_G2,ShuffleNetG2_cifar10,ShuffleNetG2_cifar100

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


def get_images(args, model_lists, hook_for_display, ipc_id):
    print("get_images call")
    save_every = 100
    batch_size = args.batch_size
    best_cost = 1e4

    loss_packed_features = [
        [BNFeatureHook(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        for model in model_lists
    ]

    if len(loss_packed_features) > 2 and len(loss_packed_features[2]) > 1:
        loss_packed_features[2].pop(1)

    targets_all = torch.LongTensor(np.arange(200))

    init_path = args.init_path
    for kk in range(0, args.num_class, batch_size):
        targets = targets_all[kk : min(kk + batch_size, args.num_class)].to("cuda")

        model_index = max(ipc_id // args.ipc_init - 1, 0)
        print(f"model_index: {model_index}, len(model_lists): {len(model_lists)}")
        if model_index >= len(model_lists) or model_index < 0:
            print("model_index is out of range!")
        model_teacher = model_lists[model_index]
        loss_r_feature_layers = loss_packed_features[model_index]

        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 调整图片大小，如果需要的话
            transforms.ToTensor(),  # 转换为 Tensor
        ])

        # # 加载图片并转换为 tensor
        # # 存储所有类别的图像 tensor
        # image_tensors = []
        
        # # 获取 init_path 下所有类别文件夹
        # category_folders = [f for f in os.listdir(args.init_path) if os.path.isdir(os.path.join(args.init_path, f))]
        
        # for category in category_folders:
        #     category_path = os.path.join(args.init_path, category)
        
        # # 获取当前类别文件夹中的所有图片文件
        #     image_files = [f for f in os.listdir(category_path) if f.endswith('.JPEG')]
        
        # # 确保文件夹中有 50 张图片
        #     if len(image_files) == 50:
        #     # 处理每张图片
        #         for img_file in image_files:
        #             img_path = os.path.join(category_path, img_file)  # 获取完整的图片路径
        #             img = Image.open(img_path).convert('RGB')  # 打开图片并转换为 RGB 格式
        #             img_tensor = transform(img).unsqueeze(0)  # 增加一个 batch 维度
        #             image_tensors.append(img_tensor)
        #     else:
        #         print(f"Warning: {category} folder does not have 50 images, it has {len(image_files)} images.")
        
        # # 将所有的图片 tensor 拼接成一个大 tensor
        # input_original = torch.cat(image_tensors, dim=0).to("cuda")  # 如果有 GPU 设备，转到 GPU 上
        targets_all = torch.LongTensor(np.arange(200))

        init_path = args.init_path
        for kk in range(0, 200, batch_size):
            targets = targets_all[kk:min(kk + batch_size, 200)].to('cuda')

            model_index = ipc_id // args.ipc_init - 1 
            model_teacher = model_lists[model_index]
            loss_r_feature_layers = loss_packed_features[model_index]

            inputs_list = []
            sorted_folders = sorted(os.listdir(init_path))
            for folder_index in range(kk, min(kk + batch_size, 200)):
            # #debug
            #     print(f"Total folders available: {len(sorted_folders)}")
            #     print(f"Accessing folder index: {folder_index}")

                folder_path = os.path.join(init_path, sorted_folders[folder_index])
                # image_path = os.path.join(folder_path, sorted(os.listdir(folder_path))[ipc_id%(args.ipc_init)])  
                # image = load_image(image_path)  
                # inputs_list.append(image)
                # 过滤掉文件夹而非图像文件
                image_files = sorted(os.listdir(folder_path))
                image_files = [f for f in image_files if os.path.isfile(os.path.join(folder_path, f))]
                
                # 获取该文件夹中对应的图像文件
                image_path = os.path.join(folder_path, image_files[ipc_id % (args.ipc_init)])
                image = load_image(image_path)
                inputs_list.append(image)

        inputs_ori = torch.stack(inputs_list).to('cuda')  
        inputs_ori.requires_grad_(False)
        inputs_ori = inputs_ori.to('cuda')
        uni_perb = torch.zeros((1, 3, 64, 64), requires_grad=True, device="cuda", dtype=torch.float)

        iterations_per_layer = args.iteration if ipc_id >= args.ipc_init else 0
        inputs = inputs_ori if iterations_per_layer == 0 else inputs_ori + uni_perb

        optimizer = optim.Adam([uni_perb], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)
        criterion = nn.CrossEntropyLoss().cuda()

        # Ensure DataParallel is applied before moving to CUDA
        model_teacher = nn.DataParallel(model_teacher).cuda()

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)
            inputs = inputs_ori + uni_perb
            #new
            min_crop = 0.08
            max_crop = 1.0
            if iteration < args.milestone * iterations_per_layer:
                if args.easy2hard_mode == "step":
                    min_crop = 1.0
                elif args.easy2hard_mode == "linear":
                    # min_crop linear decreasing: 1.0 -> 0.08
                    min_crop = 0.08 + (1.0 - 0.08) * (1 - iteration / (args.milestone * iterations_per_layer))
                elif args.easy2hard_mode == "cosine":
                    # min_crop cosine decreasing: 1.0 -> 0.08
                    min_crop = 0.08 + (1.0 - 0.08) * (1 + np.cos(np.pi * iteration / (args.milestone * iterations_per_layer))) / 2

            aug_function = transforms.Compose(
                [
                    transforms.RandomResizedCrop(64, scale=(min_crop, max_crop)),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            #new

            off1, off2 = random.randint(0, args.jitter), random.randint(0, args.jitter)
            inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))
            inputs_jit = inputs_jit.to("cuda")

            # forward pass
            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)

            # R_cross classification loss
            loss_ce = criterion(outputs, targets)

            # R_feature loss
            rescale = [args.first_bn_multiplier] + [1.0 for _ in range(len(loss_r_feature_layers) - 1)]

            loss_r_bn_feature = [
                mod.r_feature.to(loss_ce.device) * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)
            ]
            loss_r_bn_feature = torch.stack(loss_r_bn_feature).sum()

            loss_aux = args.r_bn * loss_r_bn_feature

            loss = loss_ce + loss_aux

            if iteration % save_every == 0:
                print("------------iteration {}----------".format(iteration))
                print("loss_ce", loss_ce.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("loss_total", loss.item())

            # do image update
            loss.backward()
            optimizer.step()
            # clip color outlayers
            inputs.data = clip_image(inputs.data, args.dataset)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone()  # using multicrop, save the last one
            best_inputs = denormalize_image(best_inputs, args.dataset)
            save_images(args, best_inputs, targets, ipc_id)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

    torch.cuda.empty_cache()


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
            model_teacher = model_type(num_classes=num_classes)
            model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model_teacher.maxpool = nn.Identity()
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