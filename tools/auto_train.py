from genericpath import commonprefix
import os
import itertools
import yaml
import argparse

import _init_paths

from training_tools.create_config import ConfigGenerator
from training_tools.naming import make_run_name

# Argument parser
parser = argparse.ArgumentParser(description="Run grid search with param grid YAML")
parser.add_argument("--cfg", type=str, required=True, help="Path to param_grid.yaml")

args = parser.parse_args()

# lecture du param_grid.yaml
with open(args.cfg, "r") as f:
    param_grid = yaml.safe_load(f)

loss_list = param_grid["loss_list"]
batch_size_list = param_grid["batch_size_list"]
lr_list = param_grid["lr_list"]
epochs = param_grid["epochs"]
px_size = param_grid["px_size"]

if "tl" in loss_list:
    alpha = param_grid["alpha"]
    beta = param_grid["beta"]

    cfg_gen = ConfigGenerator(
        loss_list=loss_list,
        batch_size_list=batch_size_list,
        lr_list=lr_list,
        epochs_list=epochs,
        px_size=px_size,
        alpha = alpha,
        beta = beta
    )

if "ftl" in loss_list:
    alpha = param_grid["alpha"]
    beta = param_grid["beta"]
    gamma = param_grid["gamma"]
    cfg_gen = ConfigGenerator(
        loss_list=loss_list,
        batch_size_list=batch_size_list,
        lr_list=lr_list,
        epochs_list=epochs,
        px_size=px_size,
        alpha = alpha,
        beta = beta,
        gamma = gamma
    )

else :
# Creation des fichiers YAML
    cfg_gen = ConfigGenerator(
        loss_list=loss_list,
        batch_size_list=batch_size_list,
        lr_list=lr_list,
        epochs_list=epochs,
        px_size=px_size,
        alpha = None,
        beta = None,
        gamma = None
    )





cfg_gen.generate_configs()


# Execution des scripts de training
train_script = "python3 tools/train.py"  # à adapter si besoin

# boucle sur toutes les combinaisons
for loss, batch_size, lr, epoch, pxs in itertools.product(loss_list, batch_size_list, lr_list, epochs, px_size):

    lr_str = str(lr)[2:]
    config_name = make_run_name(loss=loss, batch_size=batch_size, lr=lr, epoch=epoch, px_size=pxs) + ".yaml"

    cmd = f"{train_script} --cfg experiments/visorando/{config_name}"

    print(f"=== Running : {cmd} ===")
    os.system(cmd)
