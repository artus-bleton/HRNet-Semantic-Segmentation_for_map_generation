# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Visorando(BaseDataset):
    def __init__(self,
                root,
                list_path,
                num_classes=2,
                multi_scale=True,
                flip = True,
                downsample_rate = 1,
                scale_factor=16,
                num_samples=None,
                ignore_label=-1,
                base_size=1024,
                crop_size=(512,512),
                mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5],
                auto_weight = False,
                auto_stats = False):

        # on passe NUM_CLASSES=2
        super(Visorando, self).__init__(
            ignore_label, base_size, crop_size,
            downsample_rate=downsample_rate, scale_factor=scale_factor,
            mean=mean, std=std
        )

        self.list_path = list_path
        self.root = root
        self.img_list = [l.strip().split() for l in open(root + list_path)]
        self.files = self.read_files()



        if num_samples:
            self.files = self.files[:num_samples]

        self.num_classes = num_classes


        self.label_mapping = {
            0: 0,    #fond
            254: 1   #chemin
        }

        # poids optionnels, ici uniformes
        self.class_weights = torch.FloatTensor([1.0, 1.0]).to(device)

        self.multi_scale = multi_scale
        self.flip = flip

        if auto_weight:
            self.compute_class_weights()

        if auto_stats:
            self.compute_mean_std()



    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        out = np.zeros_like(temp, dtype=np.uint8)
        if not inverse:
            for src, dst in self.label_mapping.items():
                out[temp == src] = dst
        else:
            inv = {v: k for k, v in self.label_mapping.items()}
            for dst, src in inv.items():
                out[temp == dst] = src
        return out


    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(os.path.join(self.root, item["img"]),
                           cv2.IMREAD_COLOR)

        # print(f"[DEBUG] __getitem__ : '{item['name']}'")
        # print(f"  path   : {os.path.join(self.root, item['img'])}")
        # print(f"  dtype  : {image.dtype}")
        # print(f"  shape  : {image.shape}")
        # print(f"  min    : {image.min()}, max: {image.max()}, mean: {image.mean():.2f}, std: {image.std():.2f}")


        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root, item["label"]),
                           cv2.IMREAD_GRAYSCALE)

        raw_label = cv2.imread(os.path.join(self.root, item["label"]),
                               cv2.IMREAD_GRAYSCALE)

        label = self.convert_label(label)



        image, label = self.gen_sample(image, label)

        # print(f"[DEBUG] __getitem__ (après gen_sample) : '{item['name']}'")
        # print(f"  path   : {os.path.join(self.root, item['img'])}")
        # print(f"  dtype  : {image.dtype}")
        # print(f"  shape  : {image.shape}")
        # print(f"  min    : {image.min()}, max: {image.max()}, mean: {image.mean():.2f}, std: {image.std():.2f}")


        return image.copy(), label.copy(), np.array(size), name



    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        print("Unique values after convert_label:", np.unique(preds))


        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            print("Unique classes in pred[{}]:".format(i), np.unique(preds[i]))

            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))



    def convert_pred_to_color(self, pred):
        """
        Convertit une prédiction (H, W) en image couleur RGB (3, H, W)
        Classe 0 : noir (fond), Classe 1 : vert (chemin)
        """
        if pred.ndim == 3:
            pred = pred.squeeze(0)

        palette = {
            0: [0, 0, 0],     # fond : noir
            1: [0, 255, 0],   # chemin : vert
        }

        h, w = pred.shape
        color_image = np.zeros((3, h, w), dtype=np.uint8)
        for class_id, color in palette.items():
            mask = pred == class_id
            for i in range(3):  # R, G, B
                color_image[i][mask] = color[i]
        return color_image
