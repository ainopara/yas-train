config = {
    "height": 32,
    "train_width": 384,
    "batch_size": 128,
    "epoch": 150,
    "print_per": 100,
    "save_per": 4000,

    "train_size": 512000,
    "validate_size": 12800,

    "pretrain": False,
    "pretrain_name": "model_training.pt",

    # Set according to your CPU
    "dataloader_workers": 6,
    # Generate data online for train/val
    "online_train": False,
    "online_val": True,
    # Freeze the cnn backbone in the first few epochs
    # set 0 to disable
    "unfreeze_backbone_epoch": 50,

    # Select model type: Genshin or StarRail
    "model_type": "StarRail"
}
