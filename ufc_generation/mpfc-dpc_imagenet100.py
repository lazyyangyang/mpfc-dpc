import sys
sys.path.append("./models/")

import argparse
import collections
import json
import os
import random
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights, ResNet18_Weights

import wandb
from utils import BNFeatureHook, clip_image, denormalize_image, lr_cosine_policy, save_images


IMAGENET_CLASS_INDEX_URL = (
    "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
)


def load_imagenet_class_index(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(IMAGENET_CLASS_INDEX_URL, path)

    with open(path, "r", encoding="utf-8") as f:
        class_index = json.load(f)

    return {wnid: int(idx) for idx, (wnid, _) in class_index.items()}


def load_selected_wnids(args):
    with open(args.class_list, "r", encoding="utf-8") as f:
        wnids = [line.strip() for line in f if line.strip()]
    return wnids[:args.num_class]


def load_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)


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


def get_images(args, model_lists, ipc_id):
    print("get_images call")
    save_every = 100
    batch_size = args.batch_size
    image_size = args.image_size
    device = "cuda"

    wnid_to_imagenet_idx = load_imagenet_class_index(args.imagenet_class_index)
    selected_wnids = load_selected_wnids(args)
    global_targets_all = torch.LongTensor([wnid_to_imagenet_idx[wnid] for wnid in selected_wnids])
    local_targets_all = torch.LongTensor(np.arange(args.num_class))

    loss_packed_features = [
        [BNFeatureHook(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        for model in model_lists
    ]

    feature_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    feature_model.eval()

    for kk in range(0, args.num_class, batch_size):
        batch_wnids = selected_wnids[kk:min(kk + batch_size, args.num_class)]
        targets = global_targets_all[kk:min(kk + batch_size, args.num_class)].to(device)
        save_targets = local_targets_all[kk:min(kk + batch_size, args.num_class)].to(device)

        model_index = max(ipc_id // args.ipc_init - 1, 0)
        model_teacher = model_lists[model_index]
        loss_r_feature_layers = loss_packed_features[model_index]

        inputs_list = []
        for wnid in batch_wnids:
            folder_path = os.path.join(args.init_path, wnid)
            image_files = sorted(
                f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            )
            image_path = os.path.join(folder_path, image_files[ipc_id % args.ipc_init])
            inputs_list.append(load_image(image_path, image_size))

        input_original = torch.stack(inputs_list).to(device).detach()
        uni_perb = torch.zeros(
            (1, 3, image_size, image_size),
            requires_grad=True,
            device=device,
            dtype=torch.float,
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
                    min_crop = 0.08 + (1.0 - 0.08) * (
                        1 + np.cos(np.pi * iteration / (args.milestone * iterations_per_layer))
                    ) / 2

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
            loss_r_bn_feature = (
                torch.stack(loss_r_bn_feature).sum()
                if loss_r_bn_feature
                else torch.tensor(0.0, device=device)
            )
            loss_aux = args.r_bn * loss_r_bn_feature

            cd_loss = cosine_diversity_loss(inputs, feature_model.avgpool)
            cdm_loss = cosine_distribution_matching_loss(inputs, input_original, feature_model.avgpool)
            edm_loss = euclidean_distribution_matching_loss(inputs, input_original, feature_model.avgpool)
            total_loss = loss_ce + loss_aux + (args.rcd * cd_loss) + (args.rcdm * cdm_loss) + (args.redm * edm_loss)

            if iteration % save_every == 0:
                print("------------iteration {}----------".format(iteration))
                print("loss_ce", loss_ce.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("loss_total", total_loss.item())

            total_loss.backward()
            optimizer.step()
            inputs.data = clip_image(inputs.data, "imagenet")

            if total_loss.item() < best_cost:
                best_cost = total_loss.item()
                best_inputs = inputs.detach().clone()

        if args.store_best_images:
            best_inputs = denormalize_image(best_inputs, "imagenet")
            save_images(args, best_inputs, save_targets, ipc_id)

        optimizer.state = collections.defaultdict(dict)

    torch.cuda.empty_cache()


def generation(args, ipc_id):
    os.makedirs(args.syn_data_path, exist_ok=True)

    model_specs = [
        ("resnet18", ResNet18_Weights.IMAGENET1K_V1),
        ("mobilenet_v2", MobileNet_V2_Weights.IMAGENET1K_V1),
        ("efficientnet_b0", EfficientNet_B0_Weights.IMAGENET1K_V1),
    ]

    model_lists = []
    for model_name, weights in model_specs:
        model_teacher = models.__dict__[model_name](weights=weights)
        model_teacher = nn.DataParallel(model_teacher).cuda()
        model_teacher.eval()
        for p in model_teacher.parameters():
            p.requires_grad = False
        model_lists.append(model_teacher)

    get_images(args, model_lists, ipc_id)


def get_args():
    parser = argparse.ArgumentParser(description="MPFC-DPC: Generate ImageNet-100 distilled images")
    parser.add_argument("--dataset", default="imagenet100", type=str, choices=["imagenet100"])
    parser.add_argument("--M", default=3, type=int, help="Number of architectures involved in generation")
    parser.add_argument("--init_path", type=str, default="init_images/tiny/")
    parser.add_argument("--class-list", type=str, default="data/wnids.txt")
    parser.add_argument("--imagenet-class-index", type=str, default="data/imagenet_class_index.json")
    parser.add_argument("--num-class", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--ipc", type=int, default=10)

    parser.add_argument("--exp-name", type=str, default="generated_results")
    parser.add_argument("--wandb-name", type=str, default="imagenet100-ipc10")
    parser.add_argument("--syn-data-path", type=str, default="./syn")
    parser.add_argument("--store-best-images", action="store_true")

    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--iteration", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--jitter", type=int, default=32)
    parser.add_argument("--r-bn", type=float, default=0.05)
    parser.add_argument("--first-bn-multiplier", type=float, default=10.0)

    parser.add_argument("--easy2hard-mode", default="cosine", type=str, choices=["step", "linear", "cosine"])
    parser.add_argument("--milestone", default=0, type=float)
    parser.add_argument("--rcd", default=1.0, type=float)
    parser.add_argument("--rcdm", default=1.0, type=float)
    parser.add_argument("--redm", default=0.5, type=float)

    args = parser.parse_args()
    args.syn_data_path = os.path.join(args.syn_data_path, args.wandb_name, args.exp_name)
    return args


if __name__ == "__main__":
    args = get_args()

    if not wandb.api.api_key:
        os.environ.setdefault("WANDB_MODE", "offline")
    wandb.init(project="UFC-generation", name=args.wandb_name)

    args.ipc_start = 0
    args.ipc_init = max(1, int(args.ipc / (args.M / args.num_class + 1)))
    args.ipc_end = args.ipc_init * (args.M + 1)
    print("ipc_end = ", args.ipc_end)

    for ipc_id in range(args.ipc_start, args.ipc_end):
        print("ipc = ", ipc_id)
        wandb.log({"ipc_id": ipc_id})
        generation(args, ipc_id)

    wandb.finish()
