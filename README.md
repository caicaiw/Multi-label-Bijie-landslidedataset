
# Multilabel Bijie Landslide Dataset

The dataset performs multi-label annotation on the existing Bijie landslide dataset to support in-depth research on landslide detection.

## Background

Traditional landslide detection methods mainly rely on optical remote sensing images and are implemented through scene classification technology. Compared to object detection and semantic segmentation, scene classification simplifies the annotation process and provides a comprehensive understanding of the landslide phenomenon. This data set aims to provide a more detailed description of the landslide area through a multi-label classification method, thereby breaking through the limitations of optical remote sensing in landslide detection.

## Dataset content

- **IMAGE**: JPEG format。
- **LABEL**: Contained in three CSV files：`multilabel.csv` `train.csv` `valid.csv`

## Dataset Structure

```
multilabel-bijie-landslide-dataset/
├── image/
│   ├── ddfgf20157.jpg
│   ├── ddfgf20158.jpg
│   └── ...
├── multilabel.csv
├── train.csv
└── valid.csv
```

## Usage Example

```python
from dataloader import TrainValidData
from torch.utils.data import DataLoader
import os

# Configure data path
kw = 'multilabel_bijie_landslide_dataset'
root_path = r'D:\data_set\'
data_dir = os.path.join(root_path, kw)
train_path = os.path.join(data_dir, 'train.csv')
val_path = os.path.join(data_dir, 'valid.csv')
img_path = os.path.join(data_dir, 'image/')

# Initialize dataset
train_dataset = TrainValidData(train_path, img_path)
val_dataset = TrainValidData(val_path, img_path)

# Initialize data loader
trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, drop_last=True)
```

## Get the Dataset

The dataset will be available for download immediately after the relevant article is published.

## Acknowledgments

We sincerely thank Professor Ji Shunping's team for their outstanding contributions in the field of landslide research and their support in the construction of this data set.
