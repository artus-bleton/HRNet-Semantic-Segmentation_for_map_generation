import os
import yaml
import itertools

# paramètres à tester
loss_list = ["dice"]
batch_size_list = [4, 16, 32]
lr_list = [1e-4, 5e-4, 1e-3]

# template commun (copié depuis ton message)
base_config = {
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
        "TYPE": "dice",  # sera modifié
        "BALANCE_WEIGHTS": [1],
    },
    "TRAIN": {
        "IMAGE_SIZE": [320, 320],
        "BASE_SIZE": 320,
        "BATCH_SIZE_PER_GPU": 32,  # sera modifié
        "SHUFFLE": True,
        "BEGIN_EPOCH": 0,
        "END_EPOCH": 10,
        "RESUME": False,
        "OPTIMIZER": "sgd",
        "LR": 0.001,  # sera modifié
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
        "BATCH_SIZE_PER_GPU": 32,  # sera modifié
        "NUM_SAMPLES": None,
        "FLIP_TEST": True,
        "MULTI_SCALE": True,
    },
}

# dossier de sortie des configs
output_dir = "./experiments/visorando/"
os.makedirs(output_dir, exist_ok=True)

# boucle sur toutes les combinaisons
for loss, batch_size, lr in itertools.product(loss_list, batch_size_list, lr_list):
    # formatage du lr pour le nom
    lr_str = f"{lr:.0e}" if lr < 1e-3 else f"{lr:.1e}"

    # nom du fichier
    filename = f"config_viso_{loss}_new_dataset_batchsize{batch_size}_lr{lr_str}.yaml"
    filepath = os.path.join(output_dir, filename)

    # création de la config spécifique
    config = base_config.copy()
    config["LOSS"]["TYPE"] = loss
    config["TRAIN"]["BATCH_SIZE_PER_GPU"] = batch_size
    config["TEST"]["BATCH_SIZE_PER_GPU"] = batch_size
    config["TRAIN"]["LR"] = lr

    # écriture YAML
    with open(filepath, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"✅ Fichier généré : {filepath}")
