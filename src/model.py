import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from torchvision.models.detection import MaskRCNN
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models.detection.anchor_utils import AnchorGenerator
import timm

class TimmToVisionFPN(nn.Module):
    """
    Ponte para usar um backbone da biblioteca `timm` com a FPN da `torchvision`.
    """
    def __init__(self, backbone_timm):
        super().__init__()
        self.backbone = backbone_timm
        
        in_channels_list = [
            self.backbone.feature_info.channels(i) for i in self.backbone.feature_info.out_indices
        ]
        
        out_channels = 256
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )
        
        self.out_channels = out_channels

    def forward(self, x):
        features = self.backbone(x)
        features_dict = OrderedDict(zip([str(i) for i in range(len(features))], features))
        features_fpn = self.fpn(features_dict)
        return features_fpn

def create_model(num_classes, image_size=640):
    """
    Cria o modelo Mask R-CNN com um backbone ConvNeXt V2 customizado.
    """
    backbone_timm = timm.create_model(
        'convnextv2_large.fcmae_ft_in22k_in1k',
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    
    backbone_with_fpn = TimmToVisionFPN(backbone_timm)
    
    anchor_sizes = ((16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Parâmetros para desativar completamente a transformação interna do modelo
    image_mean = [0.0] * 3
    image_std = [1.0] * 3

    model = MaskRCNN(
        backbone_with_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=roi_pooler,
        # **A CORREÇÃO DEFINITIVA ESTÁ AQUI**
        # Força o modelo a usar o tamanho de imagem que nós definimos,
        # desativando o seu redimensionamento interno que causa o estouro de memória.
        min_size=image_size,
        max_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
    )

    return model

