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

    targets_all = torch.LongTensor(np.arange(args.num_class))


    for kk in range(0, args.num_class, batch_size):
        targets = targets_all[kk : min(kk + batch_size, args.num_class)].to("cuda")

        # model_index = ipc_id // args.ipc_init - 1
        model_index = max(ipc_id // args.ipc_init - 1, 0)
        
        model_teacher = model_lists[model_index]
        loss_r_feature_layers = loss_packed_features[model_index]

        # initialization
        loaded_tensor = torch.load(f"{args.init_path}/tensor_{ipc_id % args.ipc_init}.pt").clone()
        input_original = loaded_tensor.to("cuda").detach()
        uni_perb = torch.zeros((1, 3, 32, 32), requires_grad=True, device="cuda", dtype=torch.float)

        iterations_per_layer = args.iteration if ipc_id >= args.ipc_init else 0
        inputs = input_original if iterations_per_layer == 0 else input_original + uni_perb

        optimizer = optim.Adam([uni_perb], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)
        criterion = nn.CrossEntropyLoss().cuda()       

        #new
        model_teacher = nn.DataParallel(model_teacher).to("cuda:0")

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)
            inputs = input_original + uni_perb
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
                    # transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                    # transforms.RandomResizedCrop(64, scale=(min_crop, max_crop)),
                    transforms.RandomResizedCrop(32, scale=(min_crop, max_crop)),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            inputs_jit = aug_function(inputs)
            off1, off2 = random.randint(0, args.jitter), random.randint(0, args.jitter)
            inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))
            #new
            
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
                # comment below line can speed up the training (no validation process)
                # if hook_for_display is not None:
                #     acc_jit, _ = hook_for_display(inputs_jit, targets)
                #     acc_image, loss_image = hook_for_display(inputs, targets)

                #     metrics = {
                #         'crop/acc_crop': acc_jit,
                #         'image/acc_image': acc_image,
                #         'image/loss_image': loss_image,
                #     }
                #     wandb_metrics.update(metrics)

                # metrics = {
                #     'crop/loss_ce': loss_ce.item(),
                #     'crop/loss_r_bn_feature': loss_r_bn_feature.item(),
                #     'crop/loss_total': loss.item(),
                # }
                # wandb_metrics.update(metrics)
                # wandb.log(wandb_metrics)

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