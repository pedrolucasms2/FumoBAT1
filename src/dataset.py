import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

class PestDetectionDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        """
        Args:
            root_dir (string): Diretório com todas as imagens.
            annotation_file (string): Caminho para o arquivo de anotação COCO.
            transforms (callable, optional): Transformações opcionais a serem aplicadas
                                             em uma amostra.
        """
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        # Obtém o ID da imagem
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_anns = coco.loadAnns(ann_ids)

        # Caminho para a imagem
        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root_dir, path)
        img = Image.open(img_path).convert('RGB')

        # Extrai as bounding boxes
        boxes = []
        for ann in coco_anns:
            # O formato COCO é [x_min, y_min, width, height]
            # O formato do PyTorch é [x_min, y_min, x_max, y_max]
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            xmax = xmin + ann['bbox'][2]
            ymax = ymin + ann['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

        # Converte para tensor, garantindo a forma correta [N, 4] mesmo se N=0
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            # Se não houver caixas, crie um tensor com a forma correta [0, 4]
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # Extrai os labels (IDs das classes)
        labels = torch.as_tensor([ann['category_id'] for ann in coco_anns], dtype=torch.int64)

        # Extrai as máscaras de segmentação
        masks = []
        for ann in coco_anns:
            masks.append(coco.annToMask(ann))
        
        # Converte a lista de máscaras em um único array numpy antes de criar o tensor
        if masks:
            masks = np.stack(masks, axis=0)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        else: # Caso não haja anotações na imagem
            masks = torch.empty((0, img.height, img.width), dtype=torch.uint8)

        # Monta o dicionário de 'targets'
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])
        
        # Calcula a área das bounding boxes
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Supõe que não há instâncias "crowd"
        target["iscrowd"] = torch.zeros((len(coco_anns),), dtype=torch.int64)

        # Aplica as transformações, se houver
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def get_classes(self):
        """
        Retorna uma lista de nomes de classes do arquivo de anotação COCO.
        """
        if self.coco:
            cats = self.coco.loadCats(self.coco.getCatIds())
            # O len() disso nos dará o número de classes.
            return cats
        return []

