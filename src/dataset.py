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
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert("RGB")

        # --- FILTRO DE SANIDADE FINAL E DEFINITIVO ---
        # Listas para guardar apenas as anotações 100% válidas
        valid_boxes = []
        valid_labels = []
        valid_masks = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            # 1. Verifica se a caixa tem dimensões válidas
            if w > 0 and h > 0:
                # 2. Gera a máscara
                mask = coco.annToMask(ann)
                # 3. VERIFICAÇÃO CRUCIAL: Garante que a máscara não está vazia.
                #    Isto apanha polígonos de segmentação inválidos.
                if mask.sum() > 0:
                    valid_boxes.append([x, y, x + w, y + h]) # Converte para [x1, y1, x2, y2]
                    valid_labels.append(ann['category_id'])
                    valid_masks.append(mask)

        num_objs = len(valid_boxes)

        if num_objs > 0:
            boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
            labels = torch.as_tensor(valid_labels, dtype=torch.int64)
            masks = np.array(valid_masks)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        else:
            # Cria tensores vazios com as formas corretas se não houver objetos válidos
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

