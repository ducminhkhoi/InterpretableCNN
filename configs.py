from dataset import CUBDataset, VOCPartDataset


settings = {
    'CUB': {
        'num_classes': 200,
        'img_size': 224,
        'root': 'data/CUB',
        'train_transform': CUBDataset.train_transform,
        'test_transform': CUBDataset.test_transform,
    },

    'VOCPart': {
        'num_classes': 6,
        'img_size': 224,
        'root': 'data/VOCPart',
        'train_transform': VOCPartDataset.train_transform,
        'test_transform': VOCPartDataset.test_transform,
    },
}

