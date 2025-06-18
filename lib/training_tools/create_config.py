import os
import yaml
import itertools
import copy
import glob
import re

from .naming import make_run_name

class ConfigGenerator:
    def __init__(self, px_size:list[float],loss_list : list[str], batch_size_list:list[int], lr_list:list[float], epochs_list:list[int],  output_dir="./experiments/visorando/", ):
        self.loss_list = loss_list
        self.batch_size_list = batch_size_list
        self.lr_list = lr_list
        self.epochs = epochs_list
        self.output_dir = output_dir
        self.px_size = px_size
        os.makedirs(self.output_dir, exist_ok=True)
        self.base_config = self._get_base_config()

    def _get_base_config(self):
        return {
            "CUDNN": {"BENCHMARK": True, "DETERMINISTIC": False, "ENABLED": True},
            "GPUS": [0],
            "OUTPUT_DIR": "output/visorando",
            "LOG_DIR": "log",
            "WORKERS": 8,
            "PRINT_FREQ": 10,
            "DATASET": {
                "DATASET": "Visorando",
                "ROOT": "data/",
                "TRAIN_SET": "list/visorando/train.lst",
                "TEST_SET": "list/visorando/val.lst",
                "NUM_CLASSES": 2,
                "IGNORE_LABEL": -1,
                "BASE_SIZE": 320,
                "CROP_SIZE": [320, 320],
                "SCALE_FACTOR": 16,
                "FLIP": True,
                "MULTI_SCALE": True,
                "DOWNSAMPLE_RATE": 1,
                "MEAN": [0.5, 0.5, 0.5],
                "STD": [0.5, 0.5, 0.5],
                "AUTO_WEIGHT": False,
                "AUTO_STATS": False,
                "NUM_SAMPLES": None,
            },
            "MODEL": {
                "NAME": "seg_hrnet",
                "PRETRAINED": "",
                "ALIGN_CORNERS": True,
                "NUM_OUTPUTS": 1,
                "EXTRA": {
                    "FINAL_CONV_KERNEL": 1,
                    "STAGE1": {
                        "NUM_MODULES": 1,
                        "NUM_BRANCHES": 1,
                        "BLOCK": "BOTTLENECK",
                        "NUM_BLOCKS": [4],
                        "NUM_CHANNELS": [64],
                        "FUSE_METHOD": "SUM",
                    },
                    "STAGE2": {
                        "NUM_MODULES": 1,
                        "NUM_BRANCHES": 2,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [4, 4],
                        "NUM_CHANNELS": [48, 96],
                        "FUSE_METHOD": "SUM",
                    },
                    "STAGE3": {
                        "NUM_MODULES": 4,
                        "NUM_BRANCHES": 3,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [4, 4, 4],
                        "NUM_CHANNELS": [48, 96, 192],
                        "FUSE_METHOD": "SUM",
                    },
                    "STAGE4": {
                        "NUM_MODULES": 3,
                        "NUM_BRANCHES": 4,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [4, 4, 4, 4],
                        "NUM_CHANNELS": [48, 96, 192, 384],
                        "FUSE_METHOD": "SUM",
                    },
                },
            },
            "LOSS": {
                "TYPE": "di_tl",
                "LBD": 0.002,
                "BALANCE_WEIGHTS": [1],
            },
            "TRAIN": {
                "IMAGE_SIZE": [320, 320],
                "BASE_SIZE": 320,
                "BATCH_SIZE_PER_GPU": 32,
                "SHUFFLE": True,
                "BEGIN_EPOCH": 0,
                "END_EPOCH": 10,
                "RESUME": "",
                "OPTIMIZER": "sgd",
                "LR": 0.0005,
                "WD": 0.0005,
                "MOMENTUM": 0.9,
                "NESTEROV": False,
                "FLIP": True,
                "MULTI_SCALE": True,
                "DOWNSAMPLERATE": 1,
                "IGNORE_LABEL": -1,
                "SCALE_FACTOR": 16,
            },
            "TEST": {
                "IMAGE_SIZE": [320, 320],
                "BASE_SIZE": 320,
                "BATCH_SIZE_PER_GPU": 32,
                "NUM_SAMPLES": None,
                "FLIP_TEST": True,
                "MULTI_SCALE": True,
            },
        }

    def generate_configs(self):
        for loss, batch_size, lr, epoch, px_size in itertools.product(
            self.loss_list, self.batch_size_list, self.lr_list, self.epochs, self.px_size
        ):
            self._generate_single_config(loss, batch_size, lr, epoch, px_size)

    def _generate_single_config(self, loss, batch_size, lr, epoch, px_size):
        filename = make_run_name(loss=loss, batch_size=batch_size, lr=lr, epoch=epoch, px_size=px_size) + ".yaml"
        filepath = os.path.join(self.output_dir, filename)

        config = copy.deepcopy(self.base_config)
        config["LOSS"]["TYPE"] = loss
        config["TRAIN"]["BATCH_SIZE_PER_GPU"] = batch_size
        config["TEST"]["BATCH_SIZE_PER_GPU"] = batch_size
        config["TRAIN"]["LR"] = lr
        config["TRAIN"]["END_EPOCH"] = epoch
        config["DATASET"]["TRAIN_SET"] = f"list/visorando{int(px_size * 10)}/train.lst"
        config["DATASET"]['TEST_SET'] = f"list/visorando{int(px_size * 10)}/val.lst"

        anciens_epochs = self._find_previous_epochs(loss, batch_size, lr, epoch, px_size)
        if anciens_epochs:
            config["TRAIN"]["RESUME"] = "output/visorando/Visorando/" + make_run_name(loss=loss, batch_size=batch_size, lr=lr, epoch=max(anciens_epochs), px_size=px_size) + "/"


        with open(filepath, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        print(f"Fichier généré : {filepath}")

    def _find_previous_epochs(self, loss, batch_size, lr: float, current_epoch: int, px_size:float):
        # Obtenir le préfixe sans epoch
        prefix = make_run_name(loss, batch_size, lr, epoch=None, px_size=px_size)
        pattern = re.compile(rf"^{re.escape(prefix)}_epch(\d+)\.yaml$")

        anciens = []

        for fname in os.listdir(self.output_dir):
            match = pattern.match(fname)
            if match:
                e = int(match.group(1))
                if e < current_epoch:
                    anciens.append(e)

        return anciens
