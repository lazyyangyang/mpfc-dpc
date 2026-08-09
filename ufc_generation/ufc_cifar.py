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
import torchvision.transforms as transforms  # ✅ 加这一行
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
from PIL import Image
from utils import BNFeatureHook, lr_cosine_policy, save_images, clip_image, denormalize_image
import wandb

#new
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob
from scipy.spatial import ConvexHull

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm"
]


def parse_tsne_classes(args):
    """
    解析需要绘制 t-SNE 的类别编号。
    - CIFAR-10 默认跳过 airplane，即绘制 1-9 类。
    - CIFAR-100 默认选择 7 个语义差异较大的类别，便于观察清晰的类间分布。
    """
    if args.tsne_classes.strip():
        selected_classes = [
            int(x.strip()) for x in args.tsne_classes.split(",")
            if x.strip() != ""
        ]
    else:
        if args.dataset == "cifar10":
            selected_classes = list(range(1, 10))  # 默认跳过 airplane
        elif args.dataset == "cifar100":
            # CIFAR-100 fine label ids:
            # 0: apple
            # 1: aquarium_fish
            # 5: bed
            # 6: bee
            # 13: bus
            # 33: forest
            # 69: rocket
            # 这 7 类语义差异较大，适合 t-SNE 可视化类间边界。
            selected_classes = [0, 1, 5, 6, 13, 33, 69]
        else:
            raise ValueError(
                f"t-SNE currently supports cifar10 and cifar100, got {args.dataset}"
            )

    num_classes = 10 if args.dataset == "cifar10" else 100
    for class_id in selected_classes:
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(
                f"Invalid class id {class_id} for {args.dataset}. "
                f"Valid range is [0, {num_classes - 1}]."
            )

    print(f"Selected t-SNE classes for {args.dataset}: {selected_classes}")

    return selected_classes


class SyntheticImageDataset(Dataset):
    def __init__(self, syn_data_path, dataset="cifar10", selected_classes=None, max_per_class=-1):
        self.samples = []

        if selected_classes is None:
            selected_classes = list(range(10 if dataset == "cifar10" else 100))

        # save_images 一般会保存成：
        # syn_data_path/new000/*.jpg
        # syn_data_path/new001/*.jpg
        for class_id in selected_classes:
            class_dir = os.path.join(syn_data_path, f"new{class_id:03d}")
            image_paths = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))
            image_paths = image_paths[10:]

            if max_per_class is not None and max_per_class > 0:
                image_paths = image_paths[:max_per_class]

            print(f"[t-SNE] class {class_id:03d}: found {len(image_paths)} images in {class_dir}")

            for path in image_paths:
                self.samples.append((path, class_id))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No synthetic images found in {syn_data_path}. "
                f"Please check --syn-data-path, --wandb-name, --exp-name and --tsne-classes."
            )

        # 注意：这里要和对应数据集 teacher 训练时的 normalize 保持一致
        if dataset == "cifar10":
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                )
            ])
        elif dataset == "cifar100":
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])
        else:
            raise ValueError(f"t-SNE currently supports cifar10 and cifar100, got {dataset}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


def build_cifar_resnet18_feature_extractor(ckpt_path, num_classes=10, device="cuda"):
    model = torchvision.models.resnet18(num_classes=num_classes)

    # 这部分要和 generation() 里面的 ResNet18 CIFAR 改法一致
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

    # 因为 teacher 可能是 nn.DataParallel 保存/加载的，
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

    # 去掉最后的 fc，只保留特征提取部分，输出 [N, 512, 1, 1]
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()

    return feature_extractor


def get_tsne_feature_extractor(args, device):
    if args.dataset == "cifar10":
        ckpt_path = "pretrained/cifar-10/resnet18_E200/ckpt.pth"
        num_classes = 10
    elif args.dataset == "cifar100":
        ckpt_path = "pretrained/cifar-100/resnet18_E200/ckpt.pth"
        num_classes = 100
    else:
        raise ValueError(f"t-SNE currently supports cifar10 and cifar100, got {args.dataset}")

    return build_cifar_resnet18_feature_extractor(
        ckpt_path=ckpt_path,
        num_classes=num_classes,
        device=device
    )


def get_class_name(dataset, class_id):
    if dataset == "cifar10":
        return CIFAR10_CLASSES[class_id]
    if dataset == "cifar100":
        return CIFAR100_CLASSES[class_id]
    return f"class_{class_id}"


def plot_tsne_for_synthetic_data(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    selected_classes = parse_tsne_classes(args)

    syn_dataset = SyntheticImageDataset(
        syn_data_path=args.syn_data_path,
        dataset=args.dataset,
        selected_classes=selected_classes,
        max_per_class=args.tsne_max_per_class
    )

    syn_loader = DataLoader(
        syn_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    feature_extractor = get_tsne_feature_extractor(args, device)

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

    print(f"[t-SNE] dataset: {args.dataset}")
    print(f"[t-SNE] selected classes: {selected_classes}")
    print("[t-SNE] input feature shape:", features.shape)
    print("[t-SNE] label shape:", labels.shape)

    if features.shape[0] < 4:
        raise RuntimeError("Too few samples for t-SNE. Please select more classes or more images per class.")

    perplexity = min(30, max(2, (features.shape[0] - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        init="pca",
        random_state=42
    )

    features_2d = tsne.fit_transform(features)

    # ===================== 美化版 t-SNE 绘图 =====================
    plt.figure(figsize=(8.5, 6.8))

    # tab10 适合 <=10 类，tab20 适合更多类别
    if len(selected_classes) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_classes)))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, len(selected_classes)))

    for color_idx, class_id in enumerate(selected_classes):
        idx = labels == class_id
        points = features_2d[idx]

        if points.shape[0] == 0:
            continue

        color = colors[color_idx]
        class_name = get_class_name(args.dataset, class_id)

        # 1) 使用类别内 85% 近中心点绘制 Convex Hull 区域，避免离群点把区域拉得过大
        hull_input = points
        if points.shape[0] >= 5:
            center = points.mean(axis=0)
            dist = np.linalg.norm(points - center, axis=1)
            hull_input = points

        # 2) 绘制类别半透明区域和边界
        if hull_input.shape[0] >= 3:
            try:
                hull = ConvexHull(hull_input)
                hull_points = hull_input[hull.vertices]

                plt.fill(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    color=color,
                    alpha=0.14,
                    edgecolor=color,
                    linewidth=1.6,
                    zorder=1
                )
            except Exception:
                pass

        # 3) 绘制散点
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=55,
            color=color,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.7,
            label=class_name,
            zorder=2
        )

        # 4) 在类别中心添加类别名称标签
        center = points.mean(axis=0)
        plt.text(
            center[0],
            center[1],
            class_name,
            fontsize=8.2,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=color,
                alpha=0.78
            ),
            zorder=3
        )

    plt.legend(
        fontsize=8,
        markerscale=1.0,
        frameon=True,
        loc="best",
        borderpad=0.6
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("t-SNE Dimension 1", fontsize=13)
    plt.ylabel("t-SNE Dimension 2", fontsize=13)

    class_ids_str = "_".join([str(x) for x in selected_classes])
    plt.title(
        f"t-SNE Visualization of Distilled {args.dataset.upper()} Images",
        fontsize=16
    )

    plt.grid(True, linestyle="--", alpha=0.2)
    plt.tight_layout()

    save_path_png = os.path.join(args.syn_data_path, f"tsne_synthetic_{args.dataset}_classes_{class_ids_str}.png")
    save_path_pdf = os.path.join(args.syn_data_path, f"tsne_synthetic_{args.dataset}_classes_{class_ids_str}.pdf")

    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_pdf, bbox_inches="tight")
    plt.close()

    print(f"Saved t-SNE figure to: {save_path_png}")
    print(f"Saved t-SNE figure to: {save_path_pdf}")


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
        #new
        # print(f"model_index: {model_index}, len(model_lists): {len(model_lists)}")
        # if model_index >= len(model_lists) or model_index < 0:
        #     print("model_index is out of range!")
        #new
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

        
        model_teacher = nn.DataParallel(model_teacher).to("cuda:0")
        
        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)
            inputs = input_original + uni_perb

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
        #debug
        # if hasattr(model_teacher.module, 'fc'):
        #     model_teacher.module.fc = nn.Linear(model_teacher.module.fc.in_features, 100)
        # else:
        #     print("Model does not have 'fc' layer")

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
    parser.add_argument("--plot-tsne", action="store_true", help="Plot t-SNE visualization for generated synthetic CIFAR-10 or CIFAR-100 images")
    parser.add_argument("--tsne-classes", default="", type=str, help="Comma-separated class ids for t-SNE. Example: '0,1,2,3,4'. " "For CIFAR-10, empty means classes 1-9 without airplane. ""For CIFAR-100, empty means classes 0-9.")
    parser.add_argument(
        "--tsne-max-per-class",
        default=-1,
        type=int,
        help="Maximum number of synthetic images used per selected class for t-SNE. "
             "Use -1 to use all images."
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

    if args.plot_tsne and args.dataset in ["cifar10", "cifar100"]:
        plot_tsne_for_synthetic_data(args)

    wandb.finish()

# 什么时候需要调试？当结果不符合预期时