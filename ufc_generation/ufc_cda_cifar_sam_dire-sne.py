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

import time
import subprocess

import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
# t-sne new
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob

class SyntheticImageDataset(Dataset):
    def __init__(self, syn_data_path, dataset="cifar10"):
        self.samples = []

        # 你的 save_images 一般会保存成：
        # syn_data_path/new000/class000_id000.jpg
        # syn_data_path/new001/class001_id000.jpg
        for class_id in range(10):
            class_dir = os.path.join(syn_data_path, f"new{class_id:03d}")
            image_paths = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))[:10]

            for path in image_paths:
                self.samples.append((path, class_id))

        if len(self.samples) == 0:
            raise RuntimeError(f"No synthetic images found in {syn_data_path}")

        # 注意：这里要和你的 CIFAR-10 teacher 训练时的 normalize 保持一致
        if dataset == "cifar10":
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label

def build_cifar_resnet18_feature_extractor(ckpt_path, num_classes=10, device="cuda"):
    model = torchvision.models.resnet18(num_classes=num_classes)

    # 这部分要和你 generation() 里面的 ResNet18 CIFAR 改法一致
    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    # 因为你的 teacher 是 nn.DataParallel 保存/加载的，
    # state_dict 里可能带 module. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k.replace("module.", "")] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 去掉最后的 fc，只保留特征提取部分
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()

    return feature_extractor

def plot_tsne_for_synthetic_data(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    syn_dataset = SyntheticImageDataset(
        syn_data_path=args.syn_data_path,
        dataset=args.dataset
    )

    syn_loader = DataLoader(
        syn_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    feature_extractor = build_cifar_resnet18_feature_extractor(
        ckpt_path="pretrained/cifar-10/resnet18_E200/ckpt.pth",
        num_classes=10,
        device=device
    )

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in syn_loader:
            images = images.to(device)

            features = feature_extractor(images)
            features = features.view(features.size(0), -1)

            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    print("t-SNE input feature shape:", features.shape)
    print("t-SNE label shape:", labels.shape)

    # CIFAR-10 IPC=10 时大概只有 100 个点，perplexity 不要太大
    perplexity = min(30, max(5, (features.shape[0] - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        init="pca",
        random_state=42
    )

    features_2d = tsne.fit_transform(features)

    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    plt.figure(figsize=(7, 6))

    for class_id in range(10):
        idx = labels == class_id
        plt.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            s=35,
            alpha=0.8,
            label=cifar10_classes[class_id]
        )

    plt.legend(
        fontsize=8,
        markerscale=1.2,
        frameon=True,
        loc="best"
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE Visualization of Distilled CIFAR-10 Images")

    plt.tight_layout()

    save_path_png = os.path.join(args.syn_data_path, "tsne_synthetic_cifar10.png")
    save_path_pdf = os.path.join(args.syn_data_path, "tsne_synthetic_cifar10.pdf")

    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_pdf, bbox_inches="tight")
    plt.close()

    print(f"Saved t-SNE figure to: {save_path_png}")
    print(f"Saved t-SNE figure to: {save_path_pdf}")


def get_gpu_utilization():
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            encoding="utf-8"
        )
        util, mem_used, mem_total = result.strip().split("\n")[0].split(", ")
        return float(util), float(mem_used), float(mem_total)
    except Exception:
        return -1.0, -1.0, -1.0


def format_gb(x):
    return x / 1024 ** 3


def sharpness_aware_minimization(optimizer, model, loss, epsilon=0.1):
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    for param in model.parameters():
        if param.grad is not None:
            param.data += epsilon * param.grad.data.sign()
    perturbed_loss = loss
    perturbed_loss.backward(retain_graph=True)
    for name, param in model.named_parameters():
        param.data = original_params[name].data

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
    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    gpu_util_start, gpu_mem_used_start, gpu_mem_total = get_gpu_utilization()
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
            cd_loss = cosine_diversity_loss(uni_perb, resnet_model.avgpool)
            cdm_loss = cosine_distribution_matching_loss(uni_perb, input_original, resnet_model.avgpool)
            edm_loss = euclidean_distribution_matching_loss(uni_perb, input_original, resnet_model.avgpool)

            total_loss = loss_ce + loss_aux + cd_loss + cdm_loss + edm_loss

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

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    peak_allocated = format_gb(torch.cuda.max_memory_allocated())
    peak_reserved = format_gb(torch.cuda.max_memory_reserved())
    current_allocated = format_gb(torch.cuda.memory_allocated())
    current_reserved = format_gb(torch.cuda.memory_reserved())

    gpu_util_end, gpu_mem_used_end, gpu_mem_total = get_gpu_utilization()

    print("========== Resource Usage ==========")
    print(f"IPC ID: {ipc_id}")
    print(f"Generation Time: {elapsed_time:.2f} seconds")
    print(f"Peak GPU Memory Allocated: {peak_allocated:.2f} GB")
    print(f"Peak GPU Memory Reserved: {peak_reserved:.2f} GB")
    print(f"Current GPU Memory Allocated: {current_allocated:.2f} GB")
    print(f"Current GPU Memory Reserved: {current_reserved:.2f} GB")
    print(f"GPU Utilization Start: {gpu_util_start:.2f}%")
    print(f"GPU Utilization End: {gpu_util_end:.2f}%")
    print(f"nvidia-smi Memory Start: {gpu_mem_used_start:.2f} MB / {gpu_mem_total:.2f} MB")
    print(f"nvidia-smi Memory End: {gpu_mem_used_end:.2f} MB / {gpu_mem_total:.2f} MB")
    print("====================================")

    wandb.log({
        "resource/ipc_id": ipc_id,
        "resource/generation_time_sec": elapsed_time,
        "resource/peak_memory_allocated_GB": peak_allocated,
        "resource/peak_memory_reserved_GB": peak_reserved,
        "resource/current_memory_allocated_GB": current_allocated,
        "resource/current_memory_reserved_GB": current_reserved,
        "resource/gpu_util_start_percent": gpu_util_start,
        "resource/gpu_util_end_percent": gpu_util_end,
        "resource/nvidia_smi_memory_start_MB": gpu_mem_used_start,
        "resource/nvidia_smi_memory_end_MB": gpu_mem_used_end,
    })
    torch.cuda.empty_cache()
# def get_images(args, model_lists, hook_for_display, ipc_id):
#     print("get_images call")
#     save_every = 100
#     batch_size = args.batch_size
#     best_cost = 1e4

#     loss_packed_features = [
#         [BNFeatureHook(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
#         for model in model_lists
#     ]
#     if len(loss_packed_features) > 2 and len(loss_packed_features[2]) > 1:
#         loss_packed_features[2].pop(1)  

#     targets_all = torch.LongTensor(np.arange(args.num_class))

#     image_size = 32  # CIFAR 图片尺寸

#     # 初始化特征提取器 (用于 DiRe losses)
#     resnet_model = models.resnet18(pretrained=True)
#     resnet_model.eval()
#     feature_extractor = resnet_model.avgpool

#     for kk in range(0, args.num_class, batch_size):
#         targets = targets_all[kk : min(kk + batch_size, args.num_class)].to("cuda")
#         model_index = max(ipc_id // args.ipc_init - 1, 0)
#         model_teacher = model_lists[model_index]
#         loss_r_feature_layers = loss_packed_features[model_index]

#         # 加载 .pt 张量
#         loaded_tensor = torch.load(f"{args.init_path}/tensor_{ipc_id % args.ipc_init}.pt").clone()
#         input_original = loaded_tensor.to("cuda").detach()
#         uni_perb = torch.zeros((1, 3, image_size, image_size), requires_grad=True, device="cuda", dtype=torch.float)

#         iterations_per_layer = args.iteration if ipc_id >= args.ipc_init else 0
#         inputs = input_original if iterations_per_layer == 0 else input_original + uni_perb

#         optimizer = optim.Adam([uni_perb], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
#         lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)
#         criterion = nn.CrossEntropyLoss().cuda()       

#         model_teacher = nn.DataParallel(model_teacher).to("cuda:0")

#         for iteration in range(iterations_per_layer):
#             lr_scheduler(optimizer, iteration, iteration)
#             inputs = input_original + uni_perb

#             # 数据增强
#             min_crop = 0.08
#             max_crop = 1.0
#             if iteration < args.milestone * iterations_per_layer:
#                 if args.easy2hard_mode == "step":
#                     min_crop = 1.0
#                 elif args.easy2hard_mode == "linear":
#                     min_crop = 0.08 + (1.0 - 0.08) * (1 - iteration / (args.milestone * iterations_per_layer))
#                 elif args.easy2hard_mode == "cosine":
#                     min_crop = 0.08 + (1.0 - 0.08) * (1 + np.cos(np.pi * iteration / (args.milestone * iterations_per_layer))) / 2

#             aug_function = transforms.Compose([
#                 transforms.RandomResizedCrop(image_size, scale=(min_crop, max_crop)),
#                 transforms.RandomHorizontalFlip(),
#             ])
#             inputs_jit = aug_function(inputs)
#             off1, off2 = random.randint(0, args.jitter), random.randint(0, args.jitter)
#             inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
#             inputs_jit = inputs_jit.to("cuda")

#             # forward pass
#             optimizer.zero_grad()
#             outputs = model_teacher(inputs_jit)

#             # 分类损失
#             loss_ce = criterion(outputs, targets)

#             # BN 特征损失
#             rescale = [args.first_bn_multiplier] + [1.0 for _ in range(len(loss_r_feature_layers) - 1)]
#             loss_r_bn_feature = []
#             for idx, mod in enumerate(loss_r_feature_layers):
#                 if mod.r_feature is not None:
#                     loss_r_bn_feature.append(mod.r_feature.to(loss_ce.device) * rescale[idx])
#             loss_r_bn_feature = torch.stack(loss_r_bn_feature).sum() if loss_r_bn_feature else 0
#             loss_aux = args.r_bn * loss_r_bn_feature

#             # === DiRe losses ===
#             cd_loss = cosine_diversity_loss(uni_perb, feature_extractor)
#             cdm_loss = cosine_distribution_matching_loss(uni_perb, input_original, feature_extractor)
#             edm_loss = euclidean_distribution_matching_loss(uni_perb, input_original, feature_extractor)

#             # 总损失
#             total_loss = loss_ce + loss_aux + cd_loss + cdm_loss + edm_loss

#             if iteration % save_every == 0:
#                 print(f"------------iteration {iteration}----------")
#                 print("loss_ce", loss_ce.item())
#                 print("loss_r_bn_feature", loss_r_bn_feature.item())
#                 print("loss_total", total_loss.item())

#             # === SAM 优化 ===
#             sharpness_aware_minimization(optimizer, model_teacher, total_loss)

#             total_loss.backward()
#             optimizer.step()

#             inputs.data = clip_image(inputs.data, args.dataset)

#             if best_cost > total_loss.item() or iteration == 1:
#                 best_inputs = inputs.data.clone()

#         if args.store_best_images:
#             best_inputs = denormalize_image(best_inputs, args.dataset)
#             save_images(args, best_inputs, targets, ipc_id)

#         optimizer.state = collections.defaultdict(dict)

#     torch.cuda.empty_cache()

def generation(args, ipc_id):
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)
    
    # prepare archs for UFC
    dataset_models = {
        # 'cifar10': {
        #     'num_classes': args.num_class,
        #     'model_types': [torchvision.models.resnet18, torchvision.models.resnet34, torchvision.models.resnet50, torchvision.models.resnet101],
        #     'model_paths': [
        #         "pretrained/cifar-10/resnet18_E200/ckpt.pth",
        #         "pretrained/cifar-10/resnet34_E200/ckpt.pth",
        #         "pretrained/cifar-10/resnet50_E200/ckpt.pth",
        #         "pretrained/cifar-10/resnet101_E200/ckpt.pth"
        #     ]
        # },
        # 'cifar100': {
        #     'num_classes': args.num_class,
        #     'model_types': [torchvision.models.resnet18, torchvision.models.resnet34, torchvision.models.resnet50, torchvision.models.resnet101],
        #     'model_paths': [
        #         "pretrained/cifar-100/resnet18_E200/ckpt.pth",
        #         "pretrained/cifar-100/resnet34_E200/ckpt.pth",
        #         "pretrained/cifar-100/resnet50_E200/ckpt.pth",
        #         "pretrained/cifar-100/resnet101_E200/ckpt.pth"
        #     ]
        # }
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
        #debug
        # print(checkpoint.keys())
        # model_teacher.load_state_dict(checkpoint["state_dict"])
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
    parser.add_argument(
    "--plot-tsne",
    action="store_true",
    help="Plot t-SNE visualization for generated synthetic CIFAR-10 images"
    )

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

    if args.plot_tsne and args.dataset == "cifar10":
        plot_tsne_for_synthetic_data(args)

    wandb.finish()