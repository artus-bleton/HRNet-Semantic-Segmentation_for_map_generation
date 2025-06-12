from genericpath import commonprefix
import os
import itertools
import yaml
import argparse

from lib.training_tools.create_config import ConfigGenerator

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

# Creation des fichiers YAML
cfg_gen = ConfigGenerator(
    loss_list=loss_list,
    batch_size_list=batch_size_list,
    lr_list=lr_list,
    epochs_list=epochs
)
cfg_gen.generate_configs()


# Execution des scripts de training
train_script = "python3 tools/train.py"  # à adapter si besoin

# boucle sur toutes les combinaisons
for loss, batch_size, lr, epoch in itertools.product(loss_list, batch_size_list, lr_list, epochs):

    lr_str = str(lr)[2:]
    config_name = f"{loss}_bs{batch_size}_lr{lr_str}_epch{epoch}.yaml"

    cmd = f"{train_script} --cfg experiments/visorando/{config_name}"

    print(f"=== Running : {cmd} ===")
    os.system(cmd)
