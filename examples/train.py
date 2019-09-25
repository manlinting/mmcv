import logging
from argparse import ArgumentParser
from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import resnet_cifar
from mmcv import Config
from mmcv.runner import Runner

def parse_args():
    parser = ArgumentParser(description="Train CIFAR-10 classification")
    parser.add_argument("-i", "--config", default="config/dev.yaml", type=str, metavar="PATH", help="path of config)")
    parser.add_argument("-c", "--checkpoint", default="checkpoint", type=str, metavar="PATH", help="path to save checkpoint")
    return parser.parse_args()
args = parse_args()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def batch_processor(model, data, train_mode):
    img, label = data
    label = label.to(args.device)
    pred = model(img)
    loss = F.cross_entropy(pred, label)
    acc_top1, acc_top5 = accuracy(pred, label, topk=(1, 5))
    log_vars = OrderedDict()
    log_vars["loss"] = loss.item()
    log_vars["acc_top1"] = acc_top1.item()
    log_vars["acc_top5"] = acc_top5.item()
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=img.size(0))
    return outputs


def get_logger(log_level):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=log_level)
    logger = logging.getLogger()
    return logger

def main():

    cfg = Config.fromfile(args.config)
    print(cfg)

    logger = get_logger(cfg.log_level)
    logger.info("start training....")

    if len(cfg.gpus) > 0 and torch.cuda.is_available():
        args.device = torch.device('cuda')
        logger.info("train on gpu {}".format(str(cfg.gpus)))
    else:
        args.device = torch.device('cpu')
        logger.info("train on cpu")

    # build datasets and dataloaders
    normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
    train_dataset = datasets.CIFAR10(
        root=cfg.data_root, train=True, transform=transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    )
    val_dataset = datasets.CIFAR10(root=cfg.data_root, train=False, transform=transforms.Compose([transforms.ToTensor(), normalize]))

    num_workers = cfg.data_workers * len(cfg.gpus)
    batch_size = cfg.batch_size
    shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # build model
    model = getattr(resnet_cifar, cfg.model)()
    model = DataParallel(model, device_ids=cfg.gpus).to(args.device)

    # build runner and register hooks
    runner = Runner(model, batch_processor, cfg.optimizer, args.checkpoint, log_level=cfg.log_level)
    runner.register_training_hooks(lr_config=cfg.lr_config, optimizer_config=cfg.optimizer_config, checkpoint_config=cfg.checkpoint_config, log_config=cfg.log_config)

    # load param (if necessary) and run
    if cfg.get("resume_from") is not None:
        runner.resume(cfg.resume_from)
    elif cfg.get("load_from") is not None:
        runner.load_checkpoint(cfg.load_from)

    runner.run([train_loader, val_loader], cfg.workflow, cfg.total_epochs)


if __name__ == "__main__":
    main()
