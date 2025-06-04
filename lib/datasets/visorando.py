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
                root="data",
                list_path= "list/visorando",
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
        # image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]),
        #                    cv2.IMREAD_COLOR)
        image = cv2.imread(os.path.join(self.root, item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        # label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]),
        #                    cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(self.root, item["label"]),
                           cv2.IMREAD_GRAYSCALE)


        #debug -----
        raw_label = cv2.imread(os.path.join(self.root, item["label"]),
                                       cv2.IMREAD_GRAYSCALE)
        #print(f"[DEBUG] '{name}' - valeurs brutes :", np.unique(raw_label))
                #-----
        label = self.convert_label(label)

        # 🧪 Debug pour vérifier que les classes 0 et 1 existent après mapping
        #print(f"[DEBUG] '{name}' - uniques dans label après mapping :", np.unique(label))

        image, label = self.gen_sample(image, label,
                                self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).to(device)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).to(device)
                count = torch.zeros([1,1, new_h, new_w]).to(device)

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred

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

    def compute_class_weights(self):
        class_counts = np.zeros(self.num_classes, dtype=np.int64)

        print("📊 Calcul des pixels moyens par classe...")
        for item in self.files:
            label_path = os.path.join(self.root, item["label"])
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = self.convert_label(label)

            for cls in range(self.num_classes):
                class_counts[cls] += np.sum(label == cls)

        total_pixels = np.sum(class_counts)
        mean_pixels_per_class = class_counts / len(self.files)

        print("Nombre moyen de pixels par classe :")
        for cls in range(self.num_classes):
            print(f"  Classe {cls} : {mean_pixels_per_class[cls]:.2f} pixels")

        # Optionnel : calcul des poids inverses des fréquences
        class_freq = class_counts / total_pixels
        weights = 1.0 / (class_freq + 1e-6)
        weights = weights / np.sum(weights)

        self.class_weights = torch.FloatTensor(weights).to(device)
        print(f"Poids calculés : {self.class_weights}")



    def compute_mean_std(self):
        print("Calcul de la moyenne et de l'écart-type...")
        channel_sum = np.zeros(3)
        channel_squared_sum = np.zeros(3)
        pixel_count = 0

        for item in self.files:
            img_path = os.path.join(self.root, item["img"])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            img = img[:, :, ::-1]  # BGR -> RGB

            channel_sum += img.reshape(-1, 3).sum(axis=0)
            channel_squared_sum += (img.reshape(-1, 3) ** 2).sum(axis=0)
            pixel_count += img.shape[0] * img.shape[1]

        mean = channel_sum / pixel_count
        std = np.sqrt(channel_squared_sum / pixel_count - mean ** 2)

        print(f"Mean: {mean}")
        print(f"Std: {std}")

        self.mean = mean.tolist()
        self.std = std.tolist()
