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
        
        # Extrai os canais de saída de cada estágio do backbone
        in_channels_list = [
            self.backbone.feature_info.channels(i) for i in self.backbone.feature_info.out_indices
        ]
        
        # A FPN da torchvision espera um OrderedDict de tensores
        # O backbone do timm com `features_only=True` já retorna uma lista (ou tupla) de tensores
        # Esta classe vai converter a saída para o formato esperado.
        
        out_channels = 256 # Tamanho da saída de cada nível da FPN
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )
        
        # Adiciona o atributo out_channels que o MaskRCNN espera encontrar
        self.out_channels = out_channels

    def forward(self, x):
        # Passa a imagem pelo backbone do timm
        features = self.backbone(x)
        
        # Converte a lista de tensores de saída para um OrderedDict
        # Os nomes ('0', '1', '2', '3') são os esperados pela FPN
        # se não forem especificados de outra forma.
        features_dict = OrderedDict(zip([str(i) for i in range(len(features))], features))
        
        # Passa os features pelo FPN
        features_fpn = self.fpn(features_dict)
        
        return features_fpn

def create_model(num_classes):
    """
    Cria o modelo Mask R-CNN com um backbone ConvNeXt V2 customizado.
    """
    # Carrega o backbone pré-treinado do timm
    backbone_timm = timm.create_model(
        'convnextv2_large.fcmae_ft_in22k_in1k',
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3), # Pega saídas de 4 estágios
    )
    
    # Cria a nossa ponte FPN
    backbone_with_fpn = TimmToVisionFPN(backbone_timm)
    
    # --- Criação do Gerador de Âncoras customizado ---
    # Nossa FPN produz 4 mapas de features. O gerador de âncoras padrão do
    # MaskRCNN espera 5. Precisamos de criar um que corresponda à nossa saída.
    # Usaremos âncoras menores para ajudar na detecção de objetos pequenos.
    anchor_sizes = ((16,), (32,), (64,), (128,)) # Tamanhos para 4 níveis de features
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )

    # O RoIAlign precisa de saber os nomes das features, o tamanho da saída e a amostragem
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'], # Nomes correspondem às chaves do dict da FPN
        output_size=7,
        sampling_ratio=2
    )

    # Cria o modelo Mask R-CNN
    model = MaskRCNN(
        backbone_with_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator, # Passa o gerador customizado
        box_roi_pool=roi_pooler,
        mask_roi_pool=roi_pooler
    )

    return model

