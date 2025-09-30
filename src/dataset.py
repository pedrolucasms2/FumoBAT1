import os
import torch
from torchvision.datasets import VisionDataset
from PIL import Image
from pycocotools.coco import COCO
import numpy as np

class PestDetectionDataset(VisionDataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        super().__init__(root_dir, transforms=transforms)
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # **FILTRO DE SANIDADE ADICIONADO AQUI**
        # Garante que as caixas delimitadoras tenham área > 0
        anns = [ann for ann in anns if ann['bbox'][2] > 0 and ann['bbox'][3] > 0]

        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert("RGB")

        num_objs = len(anns)
        
        if num_objs > 0:
            boxes = [ann['bbox'] for ann in anns]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2] # x2 = x1 + w
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3] # y2 = y1 + h

            labels = [ann['category_id'] for ann in anns]
            labels = torch.as_tensor(labels, dtype=torch.int64)

            masks = [coco.annToMask(ann) for ann in anns]
            masks = np.array(masks)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        else:
            # Se não houver objetos após a filtragem, crie tensores vazios com as formas corretas
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img.height, img.width), dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
        
    def get_classes(self):
        """Retorna um dicionário de mapeamento de ID para nome da classe."""
        return self.coco.cats

