# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from torch.utils.data import Dataset
from collections import OrderedDict
import json
import os
from os import listdir
from sklearn import metrics

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image, ImageFile

import datasets
import models
from tokenizer import SimpleTokenizer
import utils


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_args_parser():
    parser = argparse.ArgumentParser(description='SLIP 0-shot evaluations', add_help=False)
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--batch-size', default=256, type=int, help='batch_size')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    return parser


class COCOvalDataset(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.transform = transform

        # load image_path, annotations
        image_path = os.path.join(root, "val2017")
        annotation_path = os.path.join(root, "annotations-2/captions_val2017.json")

        # key is image_key, value is (path, [])
        self.datas = {}
        for path in listdir(image_path):
            key = int(path.split(".")[0])
            self.datas[key] = (os.path.join(image_path, path), [])

        # load captions
        with open(annotation_path) as json_file:
            json_data = json.load(json_file)
            annotations = json_data['annotations']

        err_cnt = 0
        for ann in annotations:
            key = ann['image_id']
            caption = ann['caption']
            try:
                self.datas[key][1].append(caption)
            except:
                err_cnt += 1

        self.datas = list(self.datas.values())
        print(f"COCO Length: {len(self.datas)}")
        print(f"Total errors in COCO: {err_cnt}")

    def __getitem__(self, item):
        # print(f"item: {item}")
        image_path, texts = self.datas[item]

        image = pil_loader(image_path)
        image = self.transform(image)
        # if len(texts) != 5:
        #     print(len(texts))

        # ret = {"image": image, "text1": texts[0],}
        # return ret
        return image, "\t".join(texts[:5])

    def get_items(self, idx):
        images = []
        texts = []
        for i in idx:
            image, text = self.__getitem__(i)
            images.append(image)
            texts.append(text)

        return images, texts

    def __len__(self):
        return len(self.datas)



def main(args):
    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        ckpt_path = args.resume
    elif os.path.isfile(os.path.join(args.output_dir, 'checkpoint_best.pt')):
        ckpt_path = os.path.join(args.output_dir, 'checkpoint_best.pt')
    else:
        raise Exception('no checkpoint found')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    
    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    model = getattr(models, old_args.model)(rand_embed=False,
        ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
    model.cuda()
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    cudnn.benchmark = True

    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'dataset_catalog.json')) as f:
        catalog = json.load(f)

    with open(os.path.join(cwd, 'templates.json')) as f:
        all_templates = json.load(f)

    with open(os.path.join(cwd, 'labels.json')) as f:
        all_labels = json.load(f)

    # Data loading code
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    val_dataset = None # COCO dataset

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=4, drop_last=False)

    validate_zeroshot(val_loader, model, tokenizer)

def validate_zeroshot(val_loader, model, tokenizer):
    # switch to evaluate mode
    model.eval()

    print('=> encoding captions')
    with torch.no_grad():
        text_features = []
        image_features = []

        for images, texts in val_loader:
            images = images.cuda(non_blocking=True)
            texts = [item for t in texts for item in t.split('\t')]
            texts = tokenizer(texts).cuda(non_blocking=True)

            # encode images
            image_feature = utils.get_model(model).encode_image(images)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

            text_feature = utils.get_model(model).encode_text(texts)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

            image_features.append(image_feature)
            text_features.append(text_feature)

        image_features = torch.cat(image_features, dim=0)
        text_features = torch.cat(text_features, dim=0)

        logits_per_image = image_features @ text_features.t()
        logits_per_text = text_features @ image_features.t()

        image_labels = torch.arange(logits_per_image.shape[0]).cuda(non_blocking=True)
        text_labels = torch.arange(logits_per_text.shape[0]).cuda(non_blocking=True)

        r1, r5, r10 = rank(logits_per_image, image_labels, topk=(1, 5, 10))
        rt1, rt5, rt10 = rank(logits_per_image, text_labels, topk=(1, 5, 10))

        print(f"R1: {r1}, R5: {r5}, R10: {r10}, RT1: {rt1}, RT5: {rt5}, RT10: {rt10}")

def rank(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t() // 5

        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].t()
            correct_k = correct_k.sum(1, keepdim=True)
            correct_k = (correct_k > 0).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_per_class(outputs, targets):
    pred = outputs.argmax(1)
    confusion_matrix = metrics.confusion_matrix(targets, pred)
    per_classes = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

    return 100 * per_classes.mean()


def roc_auc(outputs, targets):
    pos_score = outputs[:, 1] - outputs[:, 0]
    metric = metrics.roc_auc_score(targets, pos_score)

    return 100 * metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SLIP 0-shot evaluations', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
