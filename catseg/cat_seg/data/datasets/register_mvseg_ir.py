import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.colormap import colormap

CLASSES = ("background(unlabelled)", 
           "Car", 
           "Bus", 
           "Motorcycle", 
           "Bicycle", 
           "Pedestrian", 
           "Motorcycle", 
           "Bicyclist", 
           "Cart", 
           "Bench", 
           "Umbrella", 
           "Box", 
           "Pole", 
           "Street_lamp", 
           "Traffic_light", 
           "Traffic_sign", 
           "Car_stop", 
           "Color_cone", 
           "Sky", 
           "Road", 
           "Sidewalk", 
           "Curb", 
           "Vegetation", 
           "Terrain", 
           "Building", 
           "Ground")


def register_dataset(root):
    ds_name = 'mvseg_ir'
    root = os.path.join(root, 'MVSeg/ir/')

    for split, image_dirname, sem_seg_dirname, class_names in [
        ('train', 'train/images', 'train/labels', CLASSES),
        ('val', 'test/images', 'test/labels', CLASSES),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        print(image_dir)
        print(gt_dir)

        full_name = f'{ds_name}_sem_seg_{split}'
        print(f"FULL NAME-------------> {full_name}")
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='png', image_ext='jpg'
            ),
        )
        MetadataCatalog.get(full_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type='sem_seg',
            ignore_label=255,
            stuff_classes=class_names,
            stuff_colors=colormap(rgb=True),
            classes_of_interest=list(range(1, len(class_names))),
            background_class=0,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
