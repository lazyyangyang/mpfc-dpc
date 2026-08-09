import sys 
sys.path.append("./models/")
from mobilenetv2 import MobileNetV2_cifar100, MobileNetV2_cifar10
from efficientnet import EfficientNetB0_cifar100, EfficientNetB0_cifar10
from shufflenet import ShuffleNetG2_cifar100, ShuffleNetG2_cifar10

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
from torchvision import transforms
from PIL import Image
from utils import BNFeatureHook, lr_cosine_policy, save_images, clip_image, denormalize_image
import wandb
from torchvision import models

import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# ===================== DiRe losses =====================
def cosine_diversity_loss(synthetic_data, feature_extractor):
    embeddings = feature_extractor(synthetic_data).view(synthetic_data.size(0), -1)
    cosine_sim = torch.matmul(embeddings, embeddings.T)
    return 1 - cosine_sim.mean()

def cosine_distribution_matching_loss(synthetic_data, real_data, feature_extractor):
    synthetic_embeddings = feature_extractor(synthetic_data).view(synthetic_data.size(0), -1)
    real_embeddings = feature_extractor(real_data).view(real_data.size(0), -1)
    cosine_sim = F.cosine_similarity(synthetic_embeddings, real_embeddings, dim=1)
    return 1 - cosine_sim.mean()

def euclidean_distribution_matching_loss(synthetic_data, real_data, feature_extractor):
    synthetic_embeddings = feature_extractor(synthetic_data).view(synthetic_data.size(0), -1)
    real_embeddings = feature_extractor(real_data).view(real_data.size(0), -1)
    dist = torch.cdist(synthetic_embeddings, real_embeddings)
    return dist.mean()

# ===================== get_images =====================
def get_images(args, model_lists, hook_for_display, ipc_id):
    print("get_images call")
    save_every = 100
    batch_size = args.batch_size

    loss_packed_features = [
        [BNFeatureHook(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        for model in model_lists
    ]
    if len(loss_packed_features) > 2 and len(loss_packed_features[2]) > 1:
        loss_packed_features[2].pop(1)

    targets_all = torch.LongTensor(np.arange(args.num_class))
    image_size = 32

    device = "cuda"

    # 这里先别只取 avgpool，至少先把模型放到 GPU
    resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    resnet_model.eval()

    for kk in range(0, args.num_class, batch_size):
        targets = targets_all[kk:min(kk + batch_size, args.num_class)].to(device)
        model_index = max(ipc_id // args.ipc_init - 1, 0)
        model_teacher = model_lists[model_index]
        loss_r_feature_layers = loss_packed_features[model_index]

        loaded_tensor = torch.load(
            f"{args.init_path}/tensor_{ipc_id % args.ipc_init}.pt",
            map_location=device
        ).clone()
        input_original = loaded_tensor.detach()
        uni_perb = torch.zeros(
            (1, 3, image_size, image_size),
            requires_grad=True,
            device=device,
            dtype=torch.float
        )

        iterations_per_layer = args.iteration if ipc_id >= args.ipc_init else 0
        inputs = input_original if iterations_per_layer == 0 else input_original + uni_perb

        best_inputs = inputs.detach().clone()
        best_cost = float("inf")

        optimizer = optim.Adam([uni_perb], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, max(iterations_per_layer, 1))
        criterion = nn.CrossEntropyLoss().to(device)

        for iteration in range(iterations_per_layer):
            lr_scheduler(optimizer, iteration, iteration)
            inputs = input_original + uni_perb

            min_crop = 0.08
            max_crop = 1.0
            if iteration < args.milestone * iterations_per_layer:
                if args.easy2hard_mode == "step":
                    min_crop = 1.0
                elif args.easy2hard_mode == "linear":
                    min_crop = 0.08 + (1.0 - 0.08) * (1 - iteration / (args.milestone * iterations_per_layer))
                elif args.easy2hard_mode == "cosine":
                    min_crop = 0.08 + (1.0 - 0.08) * (1 + np.cos(np.pi * iteration / (args.milestone * iterations_per_layer))) / 2

            aug_function = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(min_crop, max_crop)),
                transforms.RandomHorizontalFlip(),
            ])
            inputs_jit = aug_function(inputs)
            off1, off2 = random.randint(0, args.jitter), random.randint(0, args.jitter)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)
            loss_ce = criterion(outputs, targets)

            rescale = [args.first_bn_multiplier] + [1.0 for _ in range(len(loss_r_feature_layers) - 1)]
            loss_r_bn_feature = []
            for idx, mod in enumerate(loss_r_feature_layers):
                if mod.r_feature is not None:
                    loss_r_bn_feature.append(mod.r_feature.to(loss_ce.device) * rescale[idx])

            loss_r_bn_feature = torch.stack(loss_r_bn_feature).sum() if loss_r_bn_feature else torch.tensor(0.0, device=device)
            loss_aux = args.r_bn * loss_r_bn_feature

            # 这里先保留你原来的 DiRe 调用方式
            cd_loss = cosine_diversity_loss(inputs, resnet_model.avgpool)
            cdm_loss = cosine_distribution_matching_loss(inputs, input_original, resnet_model.avgpool)
            edm_loss = euclidean_distribution_matching_loss(inputs, input_original, resnet_model.avgpool)

            total_loss = loss_ce + loss_aux + (args.rcd *cd_loss) + (args.rcdm * cdm_loss) + (args.redm * edm_loss)

            total_loss.backward()
            optimizer.step()

            inputs.data = clip_image(inputs.data, args.dataset)

            if total_loss.item() < best_cost:
                best_cost = total_loss.item()
                best_inputs = inputs.detach().clone()

        if args.store_best_images:
            best_inputs = denormalize_image(best_inputs, args.dataset)
            save_images(args, best_inputs, targets, ipc_id)

        optimizer.state = collections.defaultdict(dict)

    torch.cuda.empty_cache()


def generation(args, ipc_id):
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)
    
    # prepare archs for UFC
    dataset_models = {
        'cifar10': {
            'num_classes': args.num_class,
            'model_types': [torchvision.models.resnet18, MobileNetV2_cifar10, EfficientNetB0_cifar10,ShuffleNetG2_cifar10],
            'model_paths': [
                "pretrained/cifar-10/resnet18_E200/ckpt.pth",
                "pretrained/cifar-10/mobilenetV2_E200/ckpt.pth",
                "pretrained/cifar-10/efficientnet_E200/ckpt.pth",
                "pretrained/cifar-10/shufflenet_E200/ckpt.pth"
            ]
        },
        'cifar100': {
            'num_classes': args.num_class,
            'model_types': [torchvision.models.resnet18, MobileNetV2_cifar100, EfficientNetB0_cifar100,ShuffleNetG2_cifar100],
            'model_paths': [
                "pretrained/cifar-100/resnet18_E200/ckpt.pth",
                "pretrained/cifar-100/mobilenetV2_E200/ckpt.pth",
                "pretrained/cifar-100/efficientnet_E200/ckpt.pth",
                "pretrained/cifar-100/shufflenet_E200/ckpt.pth"
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
        elif model_type == torchvision.models.resnet50:
            model_teacher = model_type(num_classes=num_classes)
            model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model_teacher.maxpool = nn.Identity() 
        elif model_type == torchvision.models.resnet101:
            model_teacher = model_type(num_classes=num_classes)
            model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model_teacher.maxpool = nn.Identity() 
        else:
            model_teacher = model_type()

        model_teacher = nn.DataParallel(model_teacher).cuda()
        checkpoint = torch.load(model_path)
        model_teacher.load_state_dict(checkpoint["state_dict"])

        model_teacher.eval()
        # model_teacher.module.fc = nn.Linear(model_teacher.module.fc.in_features, 100)
        for p in model_teacher.parameters():
            p.requires_grad = False

        model_lists.append(model_teacher)

    hook_for_display = None
    get_images(args, model_lists, hook_for_display, ipc_id)



def get_args():
    parser = argparse.ArgumentParser(description="UFC: Generate Inter-class Feature Compensator")

    # General settings
    parser.add_argument("--dataset", default="cifar100", type=str, choices=["cifar10", "cifar100", "imagenet"],
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

    parser.add_argument("--rcd", default=1.0, type=float, help="Weight for cosine diversity loss")

    parser.add_argument("--rcdm", default=1.0, type=float, help="Weight for cosine distribution matching loss")
    
    parser.add_argument("--redm", default=0.5, type=float, help="Weight for euclidean distribution matching loss")

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

    # averaging UFC for fair comparison
    args.ipc_init = int(args.ipc/(args.M/args.num_class + 1))
    args.ipc_end = args.ipc_init * (args.M + 1)
    print('ipc_end = ', args.ipc_end)

    for ipc_id in range(args.ipc_start, args.ipc_end):
        print("ipc = ", ipc_id)
        wandb.log({'ipc_id': ipc_id})
        generation(args, ipc_id)

    wandb.finish()
