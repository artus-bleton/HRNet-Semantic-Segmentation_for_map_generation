def make_run_name(loss:str, batch_size:int, lr:float, epoch:int, px_size:float):
    """
    Génère un nom unique à partir des paramètres d'entraînement.
    Exemple : dice_bs32_lr0005_epch10
    """
    lr_str = str(lr).replace('.', '')
    return f"pxs{px_size}_{loss}_bs{batch_size}_lr{lr_str}_epch{epoch}"
