## Sarthak Uday Talwadkar

class Config():
    # Model Parameters
    scale = 4
    num_resblocks = 16
    num_feature_maps = 64
    res_scale = 0.1

    # Training Parameters
    batch_size = 16
    patch_size = 48
    lr = 1e-4
    epochs = 300
    checkpoint_dir = '/home/sunny/Projects/EDSR/'

    # Data Parameters
    train_dir = '/home/sunny/Projects/EDSR/data/Testing/'
    test_dir = '/home/sunny/Projects/EDSR/data/Testing/'

    # DIV2K Dataset
    rgb_range = 255
    rgb_mean = (0.4488*255, 0.4371*255, 0.4040*255)

config = Config()


