import argparse
import json
import os
import time
import sys
sys.path.append("./models/")

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from PIL import Image
from mobilenetv2 import MobileNetV2_cifar100, MobileNetV2_cifar10, MobileNetV2_tiny_imagenet
from efficientnet import EfficientNetB0_cifar100, EfficientNetB0_cifar10, EfficientNetB0_tiny_imagenet
from shufflenet import ShuffleNetG2_cifar100, ShuffleNetG2_cifar10, ShuffleNet_tiny_imagenet_G2
from convnet import ConvNet, get_default_convnet_setting
from imagenet_ipc import ImageFolderIPC
from datetime import datetime
from tinyimagenet import TinyImageNet

parser = argparse.ArgumentParser(description="UFC: Validate with Dynamic Labeling")
parser.add_argument("--dataset", default="cifar100", type=str, choices=["cifar10", "cifar100", "tiny", "imagenet100"], help="Dataset")
parser.add_argument("--M", default=4, type=int, help="Number of architectures involved in UFC generation")
parser.add_argument("--networks", default='resnet18', type=str, help="Model architecture: resnet18, resnet34, resnet50, resnet101, mobilenetV2, efficientnet, shufflenet")
parser.add_argument("--epochs", default=200, type=int, help="Training epochs")
parser.add_argument("--batch-size", default=128, type=int, help="Batch size")
parser.add_argument("--lr", default=0.15, type=float, help="Learning rate")
parser.add_argument("--temperature", default=30, type=float, help="Temperature")
parser.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay")
parser.add_argument("--syn-data-path", default="./syn", type=str, help="Path to synthetic data")
parser.add_argument("--output-dir", default="./save", type=str, help="Directory to save results")
parser.add_argument("--resume", "-r", action="store_true", help="Resume from checkpoint")
parser.add_argument("--check-ckpt", default=None, type=str, help="Checkpoint to evaluate")
parser.add_argument("--ipc", default=5, type=int, help="IPC setting")
parser.add_argument("--ipc-total", default=None, type=int, help="Override generated images per class if it differs from the UFC formula")
parser.add_argument("--imagenet100-class-list", default="data/wnids.txt", type=str, help="WordNet ID list for ImageNet-100")
parser.add_argument("--imagenet100-num-class", default=100, type=int, help="Number of WordNet IDs to use for ImageNet-100")
parser.add_argument("--imagenet-class-index", default="data/imagenet_class_index.json", type=str, help="ImageNet-1K class index JSON")
parser.add_argument("--imagenet100-val-root", default="data/val", type=str, help="Tiny/ImageNet-style val directory for ImageNet-100")
parser.add_argument('--wandb-project', type=str, default='UFC-validation', help='WandB project name')
parser.add_argument('--wandb-api-key', type=str, default=None, help='WandB API key')
parser.add_argument('--wandb-name', type=str, default="cifar100-ipc10", help='WandB run name')
 
args = parser.parse_args()

if args.wandb_api_key:
    wandb.login(key=args.wandb_api_key)
else:
    os.environ.setdefault("WANDB_MODE", "offline")
wandb.init(project=args.wandb_project, name=f"{args.wandb_name}_{datetime.now().strftime('%m/%d, %H:%M:%S')}")

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.check_ckpt:
    checkpoint = torch.load(args.check_ckpt)
    print(f"==> Loaded checkpoint: {args.check_ckpt}, Acc: {checkpoint['acc']}, Epoch: {checkpoint['epoch']}")
    exit()

os.makedirs(args.output_dir, exist_ok=True)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")
def load_imagenet100_wnids(class_list, num_class):
    with open(class_list, "r", encoding="utf-8") as f:
        wnids = [line.strip() for line in f if line.strip()]
    return wnids[:num_class]


def load_imagenet_class_index(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is required for ImageNet-100 dynamic labels. "
            "Run the ImageNet-100 generation script once or provide --imagenet-class-index."
        )
    with open(path, "r", encoding="utf-8") as f:
        class_index = json.load(f)
    return {wnid: int(idx) for idx, (wnid, _) in class_index.items()}


class ImageNet100Val(torch.utils.data.Dataset):
    def __init__(self, root, wnids, transform=None):
        self.root = root
        self.transform = transform
        self.wnid_to_local = {wnid: idx for idx, wnid in enumerate(wnids)}
        annotations = os.path.join(root, "val_annotations.txt")
        image_dir = os.path.join(root, "images")
        self.samples = []

        with open(annotations, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, wnid = parts[0], parts[1]
                if wnid in self.wnid_to_local:
                    self.samples.append((os.path.join(image_dir, image_name), self.wnid_to_local[wnid]))

        if not self.samples:
            raise RuntimeError(f"No ImageNet-100 validation images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


imagenet100_wnids = None
imagenet100_teacher_indices = None
if args.dataset == "imagenet100":
    imagenet100_wnids = load_imagenet100_wnids(args.imagenet100_class_list, args.imagenet100_num_class)
    wnid_to_imagenet_idx = load_imagenet_class_index(args.imagenet_class_index)
    imagenet100_teacher_indices = torch.LongTensor([wnid_to_imagenet_idx[wnid] for wnid in imagenet100_wnids]).to(device)

dataset_config = {
    "cifar10": {
        "num_classes": 10,
        "transform_test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "transform_train": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    },
    "cifar100": {
        "num_classes": 100,
        "transform_test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]),
        "transform_train": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )        
    },
    "tiny": {
        "num_classes": 200,
        "transform_test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ]),
        "transform_train": transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
        ]
    )        
    },
    "imagenet100": {
        "num_classes": args.imagenet100_num_class,
        "transform_test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        "transform_train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    }
}

transform_test = dataset_config[args.dataset]["transform_test"]
transform_train = dataset_config[args.dataset]["transform_train"]
num_classes = dataset_config[args.dataset]["num_classes"]

args.ipc_init = max(1, int(args.ipc/(args.M/num_classes + 1)))
args.ipc_total = args.ipc_total if args.ipc_total is not None else args.ipc_init * (args.M + 1)

def check_files_per_folder(path, ipc_total):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")

    folder_counts = []

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):  
            num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            folder_counts.append(num_files)

            assert num_files == ipc_total, f"Error: Folder '{folder}' contains {num_files} files, expected {ipc_total}."

    if folder_counts:
        avg_files = sum(folder_counts) / len(folder_counts)
        print(f"✅ Average number of files per folder: {avg_files:.2f} (Expected: {ipc_total})")
    else:
        print("⚠ No subdirectories found.")

check_files_per_folder(args.syn_data_path, args.ipc_total)

trainset = ImageFolderIPC(root=args.syn_data_path, transform=transform_train, ipc=args.ipc_total)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)


if args.dataset == "tiny":
    testset = TinyImageNet(root="./data/", transform=transform_test)
elif args.dataset == "imagenet100":
    testset = ImageNet100Val(root=args.imagenet100_val_root, wnids=imagenet100_wnids, transform=transform_test)
elif args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=transform_test)
elif args.dataset == "cifar100":
    testset = torchvision.datasets.CIFAR100(root="../data", train=False, download=True, transform=transform_test)
else:
    raise ValueError("Unsupported dataset type")

testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

print(f"Test loader size: {len(testloader)}")

print(f"==> Building model: {args.networks}")
def build_student(network_name, dataset, num_classes):
    if "resnet" in network_name:
        model = getattr(torchvision.models, network_name)(num_classes=num_classes)
        if dataset != "imagenet100":
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        return model

    if dataset == "cifar10":
        model_classes = {
            "mobilenetV2": MobileNetV2_cifar10,
            "efficientnet": EfficientNetB0_cifar10,
            "shufflenet": ShuffleNetG2_cifar10,
        }
        return model_classes[network_name]()
    if dataset == "cifar100":
        model_classes = {
            "mobilenetV2": MobileNetV2_cifar100,
            "efficientnet": EfficientNetB0_cifar100,
            "shufflenet": ShuffleNetG2_cifar100,
        }
        return model_classes[network_name]()
    if dataset == "tiny":
        model_classes = {
            "mobilenetV2": MobileNetV2_tiny_imagenet,
            "efficientnet": EfficientNetB0_tiny_imagenet,
            "shufflenet": ShuffleNet_tiny_imagenet_G2,
        }
        return model_classes[network_name]()
    if dataset == "imagenet100":
        model_classes = {
            "mobilenetV2": lambda: torchvision.models.mobilenet_v2(num_classes=num_classes),
            "efficientnet": lambda: torchvision.models.efficientnet_b0(num_classes=num_classes),
            "shufflenet": lambda: torchvision.models.shufflenet_v2_x1_0(num_classes=num_classes),
        }
        return model_classes[network_name]()
    raise ValueError(f"Unsupported dataset/network: {dataset}/{network_name}")

model_student = build_student(args.networks, args.dataset, num_classes).to(device)

if device == "cuda":
    model_student = torch.nn.DataParallel(model_student)
    cudnn.benchmark = True

def build_imagenet100_teachers():
    specs = [
        ("resnet18", torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
        ("mobilenet_v2", torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1),
        ("efficientnet_b0", torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    ]
    teachers = []
    for name, weights in specs:
        model = torchvision.models.__dict__[name](weights=weights)
        model = nn.DataParallel(model).cuda()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        teachers.append(model)
    return teachers


dataset_models = {
    'cifar10': {
        'num_classes': 10,
        # 'model_types': [torchvision.models.resnet18,torchvision.models.resnet34,torchvision.models.resnet50,torchvision.models.resnet101],
        'model_types': [torchvision.models.resnet18,MobileNetV2_cifar10, EfficientNetB0_cifar10, ShuffleNetG2_cifar10], 
        'model_paths': [
            "pretrained/cifar-10/resnet18_E200/ckpt.pth",
            # "pretrained/cifar-10/resnet34_E200/ckpt.pth",
            # "pretrained/cifar-10/resnet50_E200/ckpt.pth",
            # "pretrained/cifar-10/resnet101_E200/ckpt.pth"
            "pretrained/cifar-10/mobilenetV2_E200/ckpt.pth",
            "pretrained/cifar-10/efficientnet_E200/ckpt.pth",
            "pretrained/cifar-10/shufflenet_E200/ckpt.pth"
        ]
    },
    'cifar100': {
        'num_classes': 100,
        'model_types': [torchvision.models.resnet18, MobileNetV2_cifar100, EfficientNetB0_cifar100, ShuffleNetG2_cifar100],
        'model_paths': [
            "pretrained/cifar-100/resnet18_E200/ckpt.pth",
            "pretrained/cifar-100/mobilenetV2_E200/ckpt.pth",
            "pretrained/cifar-100/efficientnet_E200/ckpt.pth",
            "pretrained/cifar-100/shufflenet_E200/ckpt.pth"
        ]
    },
    'tiny': {
            'num_classes': 200,
            'model_types': [torchvision.models.resnet18, MobileNetV2_tiny_imagenet, EfficientNetB0_tiny_imagenet, ShuffleNet_tiny_imagenet_G2],  
            'model_paths': [
                "pretrained/tiny/resnet18_E200/ckpt.pth",
                "pretrained/tiny/mobilenetV2_E200/ckpt.pth",
                "pretrained/tiny/efficientnet_E200/ckpt.pth",  
                "pretrained/tiny/shufflenet_E200/ckpt.pth"
        ]
    }
    # 'tiny': {
    #         'num_classes': 200,
    #         'model_types': [torchvision.models.resnet18, MobileNetV2_tiny_imagenet, EfficientNetB0_tiny_imagenet],
    #         'model_paths': [
    #             "pretrained/tiny/resnet18_E200/ckpt.pth",
    #             "pretrained/tiny/mobilenetV2_E200/ckpt.pth",
    #             "pretrained/tiny/efficientnet_E200/ckpt.pth"
    #             # ,torchvision.models.resnet34
    #             # "pretrained/tiny/resnet34_E200/ckpt.pth",
    #             # "pretrained/cifar-100/resnet50_E200/ckpt.pth",
    #             # "pretrained/cifar-100/resnet101_E200/ckpt.pth"
    #         ]
        # }
    # 'cifar100': {
    #         'num_classes': 100,
    #         'model_types': [torchvision.models.resnet18,torchvision.models.resnet34,torchvision.models.resnet50,torchvision.models.resnet101],
    #         'model_paths': [
    #             "pretrained/cifar-100/resnet18_E200/ckpt.pth",
    #             "pretrained/cifar-100/resnet34_E200/ckpt.pth",
    #             "pretrained/cifar-100/resnet50_E200/ckpt.pth",
    #             "pretrained/cifar-100/resnet101_E200/ckpt.pth"
    #         ]
    #     }
}

model_lists = []
if args.dataset == "imagenet100":
    model_lists = build_imagenet100_teachers()
else:
    assert args.dataset in dataset_models, f"Unknown dataset: {args.dataset}"
    dataset_config = dataset_models[args.dataset]
    num_classes = dataset_config['num_classes']
    model_types = dataset_config['model_types']
    model_paths = dataset_config['model_paths']

for model_type, model_path in zip(dataset_models.get(args.dataset, {}).get('model_types', []), dataset_models.get(args.dataset, {}).get('model_paths', [])):
    # if model_type == torchvision.models.resnet18:
    #     model_teacher = model_type(num_classes=num_classes)
    #     model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    #     model_teacher.maxpool = nn.Identity() 
    # else:
    #     model_teacher = model_type()
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
    model_teacher.load_state_dict(checkpoint["state_dict"],strict=False)
    model_teacher.eval()

    if hasattr(model_teacher.module, '_fc'):  # 检查是否有 _fc 层
        model_teacher.module._fc = nn.Linear(model_teacher.module._fc.in_features, 100)
    
    for p in model_teacher.parameters():
        p.requires_grad = False

    model_lists.append(model_teacher)

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    model_student.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model_student.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
loss_function_kl = nn.KLDivLoss(reduction="batchmean")


def mixup_data(x, y, alpha=0.8):
    """
    Returns mixed inputs, mixed targets, and mixing coefficients.
    For normal learning
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

# Train
def train(epoch,wandb_metrics):
    model_student.train()
    train_loss = 0
    correct = 0
    total = 0
    df1_sum = 0
    df2_sum = 0
    df3_sum = 0
    for batch_idx, (inputs, targets, indices) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        indices = indices.to(device)

        inputs, target_a, target_b, lam, mix_index = mixup_data(inputs, targets)
        
        soft_label_avg = [] 

        for ii in range(len(model_lists)):
            model_teacher = model_lists[ii]
            soft_label = model_teacher(inputs).detach()
            if args.dataset == "imagenet100":
                soft_label = soft_label.index_select(1, imagenet100_teacher_indices)
            soft_label_avg.append(soft_label.clone().detach())
        soft_label = sum(soft_label_avg) / len(soft_label_avg)

        optimizer.zero_grad()
        outputs = model_student(inputs)
        outputs_ = F.log_softmax(outputs / args.temperature, dim=1)

        soft_label_ = F.softmax(soft_label / args.temperature, dim=1)

        # crucial to make synthetic data and labels more aligned
        loss = loss_function_kl(outputs_, soft_label_)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f"Epoch: [{epoch}], Acc@1 {100.*correct/total:.3f}, Loss {train_loss/(batch_idx+1):.4f}")
    metrics = {
        "train/loss": float(f"{train_loss/(batch_idx+1):.4f}"),
        "train/Top1": float(f"{100.*correct/total:.3f}"),
        "train/epoch": epoch,
        "train/df1":float(f"{df1_sum:.4f}"),
        "train/df2":float(f"{df2_sum:.4f}"),
        "train/df3":float(f"{df3_sum:.4f}"),}
    wandb_metrics.update(metrics)
    wandb.log(wandb_metrics)


# Test
def test(epoch,wandb_metrics):
    global best_acc
    model_student.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_student(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Test: Acc@1 {100.*correct/total:.3f}, Loss {test_loss/(batch_idx+1):.4f}")

    acc = 100.0 * correct / total
    with open("test_acc_ori.csv", "a") as f:
        f.write(f"{epoch},{acc:.3f}\n")
    if acc > best_acc:
        best_acc = acc

    metrics = {
        'val/loss': float(f"{test_loss/(batch_idx+1):.4f}"),
        'val/top1': float(f"{100.*correct/total:.3f}"),
        'val/epoch': epoch,
        'val/best_acc':best_acc,
    }
    wandb_metrics.update(metrics)
    wandb.log(wandb_metrics)
    print(f"Best: Acc@1 {best_acc:.3f}")



    # Save checkpoint.
    # save last checkpoint
    if True:
        state = {
            "state_dict": model_student.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')

        path = os.path.join(args.output_dir, "./ckpt.pth")
        torch.save(state, path)
        # best_acc = acc
# def test(epoch, wandb_metrics):
#     global best_acc
#     model_student.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     batch_count = 0  # Initialize batch counter to track batches

#     try:
#         with torch.no_grad():
#             for batch_idx, (inputs, targets) in enumerate(testloader):
#                 if inputs.size(0) == 0:  # Skip empty batches
#                     print(f"Warning: Empty batch encountered at batch {batch_idx}")
#                     continue

#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model_student(inputs)
#                 loss = criterion(outputs, targets)

#                 test_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()

#                 batch_count += 1  # Increment batch counter

#         if batch_count > 0:  # Check if any batch was processed
#             print(f"Test: Acc@1 {100.*correct/total:.3f}, Loss {test_loss/batch_count:.4f}")
#         else:
#             print("No batches processed, skipping evaluation.")

#         acc = 100.0 * correct / total
#         if acc > best_acc:
#             best_acc = acc

#         metrics = {
#             'val/loss': float(f"{test_loss/batch_count:.4f}") if batch_count > 0 else 0.0,
#             'val/top1': float(f"{100.*correct/total:.3f}"),
#             'val/epoch': epoch,
#             'val/best_acc': best_acc,
#         }
#         wandb_metrics.update(metrics)
#         wandb.log(wandb_metrics)
#         print(f"Best: Acc@1 {best_acc:.3f}")

#         # Save checkpoint
#         state = {
#             "state_dict": model_student.state_dict(),
#             "acc": acc,
#             "epoch": epoch,
#         }
#         path = os.path.join(args.output_dir, "./ckpt.pth")
#         torch.save(state, path)

#     except Exception as e:
#         print(f"An error occurred during testing: {str(e)}")
#         # Log the error to a file if needed
#         with open("error_log.txt", "a") as f:
#             f.write(f"Error at epoch {epoch}: {str(e)}\n")
#         raise  # Re-raise the exception to stop execution if you want




start_time = time.time()
for epoch in range(start_epoch, start_epoch + args.epochs):
    global wandb_metrics
    wandb_metrics = {}

    train(epoch, wandb_metrics)
    # fast test
    if epoch % 10 == 0 or epoch == args.epochs - 1:
        test(epoch, wandb_metrics)
    scheduler.step()
end_time = time.time()
wandb.finish()
print(f"total time: {end_time - start_time} s")


