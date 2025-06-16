import os
import shutil
import itertools
import yaml
import argparse

import _init_paths
from training_tools.naming import make_run_name

# Argument parser
parser = argparse.ArgumentParser(description="Copy final_state.pth from trained models")
parser.add_argument("--cfg", type=str, required=True, help="Path to param_grid.yaml")
parser.add_argument("--output", type=str, required=True, help="Destination folder for copied .pth files")
parser.add_argument("--root", type=str, default=".", help="Root folder containing trained model folders")

args = parser.parse_args()

# Lecture du fichier YAML
with open(args.cfg, "r") as f:
    param_grid = yaml.safe_load(f)

loss_list = param_grid["loss_list"]
batch_size_list = param_grid["batch_size_list"]
lr_list = param_grid["lr_list"]
epochs = param_grid["epochs"]
px_size = param_grid["px_size"]

# Création du dossier de destination
os.makedirs(args.output, exist_ok=True)

# Pour chaque combinaison de paramètres
for loss, batch_size, lr, epoch in itertools.product(loss_list, batch_size_list, lr_list, epochs):
    run_name = make_run_name(loss=loss, batch_size=batch_size, lr=lr, epoch=epoch, px_size=px_size)
    model_folder = os.path.join(args.root, run_name)
    model_path = os.path.join(model_folder, "final_state.pth")
    dest_path = os.path.join(args.output, f"{run_name}.pth")

    if os.path.exists(dest_path):
        print(f"⏭️  Déjà présent, ignoré : {dest_path}")
        continue

    if os.path.isfile(model_path):
        shutil.copy(model_path, dest_path)
        print(f"✅ Copié : {model_path} -> {dest_path}")
    else:
        print(f"⚠️  Fichier manquant : {model_path}")
