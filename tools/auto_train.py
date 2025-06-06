import os
import itertools

# listes des paramètres à tester
loss_list = ["dice"]
batch_size_list = [4, 16, 32]
lr_list = [1e-4, 5e-4, 1e-3]

# chemin vers ton script de training
train_script = "python3 tools/train.py"  # à adapter si besoin

# on boucle sur le produit cartésien
for loss, batch_size, lr in itertools.product(loss_list, batch_size_list, lr_list):
    # formater le lr pour le nom de fichier (ex: 0.001 → 1e-3)
    lr_str = f"{lr:.0e}" if lr < 1e-3 else f"{lr:.1e}"

    # créer le nom de config
    config_name = f"config_viso_{loss}_new_dataset_batchsize{batch_size}_lr{lr_str}.yaml"

    # commande à exécuter
    cmd = f"{train_script} --cfg experiments/visorando/{config_name} --comp"

    print(f"=== Running : {cmd} ===")
    os.system(cmd)
