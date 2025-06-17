def make_run_name(loss: str, batch_size: int, lr: float, epoch: int | None, px_size: float = 2.7) -> str:
    """
    Génère un nom de run. Si epoch est None, retourne le préfixe jusqu'au lr.
    """
    lr_str = str(lr).replace('0.', '')

    name = f"pxs{px_size}_{loss}_bs{batch_size}_lr{lr_str}"
    if epoch is not None:
        name += f"_epch{epoch}"
    return name
