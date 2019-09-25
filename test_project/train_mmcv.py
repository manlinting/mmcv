import logging
import os
import warnings
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms

import torchvision.models as models
# from dataset_tar import DatasetTar
from imagelistdataset import ImageListDataset
from mmcv import accuracy
from mmcv.runner import Runner

from base import INIT_CONFIG, GET_CONFIG

warnings.filterwarnings("ignore")

def parse_args():
    parser = ArgumentParser(description="Train CIFAR-10 classification")
    parser.add_argument("-i", "--config", default="config/dev.yaml", type=str, metavar="PATH", help="path of config)")
    parser.add_argument("-c", "--checkpoint", default="checkpoint", type=str, metavar="PATH", help="path to save checkpoint")
    return parser.parse_args()


# global variables
args = parse_args()

INIT_CONFIG(args.config)
cfg = GET_CONFIG()
print(cfg)

def batch_processor(model, data, train_mode):

    img, label = data
    label = label.to(args.device)
    pred = model(img)
    # from IPython import embed; embed()
    loss = torch.nn.CrossEntropyLoss().to(args.device)(pred, label)
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

    logger = get_logger(cfg.log_level)
    logger.info("start training....")

    if len(cfg.gpus) > 0 and torch.cuda.is_available():
        args.device = torch.device("cuda")
        logger.info("train on gpu {}".format(str(cfg.gpus)))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cfg.gpus])
    else:
        args.device = torch.device("cpu")
        logger.info("train on cpu")

    # build datasets and dataloaders
    normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
    num_workers = cfg.data_workers * len(cfg.gpus)

    loaders = []
    class_num = 0
    for input_size in cfg.input_size:

        train_dataset = ImageListDataset( cfg.root_path, cfg.train_data, transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(int(input_size), (0.64, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(0.02, 0.02, 0.02, 0.02),
            transforms.ToTensor(),
            normalize,
        ]))

        val_dataset = ImageListDataset( cfg.root_path, cfg.val_data, transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ]))
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=num_workers)
        loaders.append(train_loader)

        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers)
        loaders.append(val_loader)

        class_num = train_dataset.get_class_num()
        logger.info('class_num:'+str(class_num))

    # build model
    model = getattr(models, cfg.model)( pretrained = False, num_classes = class_num )
    model = DataParallel(model).to(args.device)

    # build runner and register hooks
    runner = Runner(model, batch_processor, cfg.optimizer, args.checkpoint, log_level=cfg.log_level)
    runner.register_training_hooks(lr_config=cfg.lr_config, optimizer_config=cfg.optimizer_config, checkpoint_config=cfg.checkpoint_config, log_config=cfg.log_config)

    # load param (if necessary) and run
    if cfg.get("resume_from") is not None:
        runner.resume(cfg.resume_from)
    elif cfg.get("load_from") is not None:
        runner.load_checkpoint(cfg.load_from)

    runner.run(loaders, cfg.workflow, cfg.total_epochs)


if __name__ == "__main__":
    main()
