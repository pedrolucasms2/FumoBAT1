import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

class PestDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        """
        Args:
            root_dir (string): Diretório com todas as imagens.
            annotation_file (string): Caminho para o arquivo de anotação COCO.
            transforms (callable, optional): Transformações a serem aplicadas na imagem e anotações.
        """
        self.root = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Retorna uma amostra do dataset.
        """
        # Pega o ID da imagem
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_anns = coco.loadAnns(ann_ids)
        
        # Pega o caminho e carrega a imagem
        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert("RGB")

        # Pega as bounding boxes
        boxes = []
        if coco_anns: # Garante que há anotações
            for ann in coco_anns:
                # O formato COCO é [x_min, y_min, width, height]
                # O formato do PyTorch é [x_min, y_min, x_max, y_max]
                x_min, y_min, width, height = ann['bbox']
                x_max = x_min + width
                y_max = y_min + height
                boxes.append([x_min, y_min, x_max, y_max])
        
        # Evita criar um tensor vazio se não houver anotações
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # Pega os rótulos (neste caso, todos são da classe 1, "pest")
        # Classe 0 é reservada para o fundo (background)
        labels = torch.ones((len(coco_anns),), dtype=torch.int64)
        
        # Pega as máscaras de segmentação
        masks = []
        if coco_anns and 'segmentation' in coco_anns[0]:
            for ann in coco_anns:
                mask = coco.annToMask(ann)
                masks.append(mask)
        
        if masks:
            # Converte a lista de arrays numpy em um único array numpy antes de criar o tensor
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        else:
            # Cria um placeholder para máscaras se não houver anotações
            h, w = img.size
            masks = torch.zeros((0, w, h), dtype=torch.uint8)


        # Cria o dicionário de anotações
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])
        
        # Aplica as transformações, se existirem
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

