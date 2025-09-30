import torch
import timm
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork

class TimmToVisionFPN(torch.nn.Module):
    """
    Wrapper para converter a saída de um backbone 'timm' para o formato
    esperado pelo FPN da torchvision.
    Esta classe serve como uma "ponte" entre um backbone carregado via `timm`
    e o restante da arquitetura de detecção da `torchvision`.
    """
    def __init__(self, backbone_timm):
        super().__init__()
        self.backbone = backbone_timm
        
        # O FPN da torchvision espera um dicionário de tensores como entrada.
        # As chaves '0', '1', '2', '3' são nomes arbitrários que damos para
        # as saídas dos `out_indices` do backbone.
        self.return_layers = {f'{i}': f'{i}' for i in range(4)}
        
        # Pega o número de canais de saída de cada feature map do backbone
        in_channels_list = self.backbone.feature_info.channels()
        
        # O FPN irá produzir feature maps com 256 canais, um valor padrão
        # em muitas arquiteturas de detecção.
        out_channels_fpn = 256
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels_fpn,
        )
        
        # CORREÇÃO: Definimos `out_channels` manualmente, pois a classe FPN
        # não possui este atributo, mas o MaskRCNN espera encontrá-lo.
        self.out_channels = out_channels_fpn

    def forward(self, x):
        # Passa a imagem pelo backbone para extrair os feature maps
        features = self.backbone(x)
        
        # Organiza os feature maps em um dicionário no formato esperado pelo FPN
        # Ex: {'0': tensor1, '1': tensor2, ...}
        features_dict = {str(i): f for i, f in enumerate(features)}
        
        # Passa os feature maps pelo FPN
        output = self.fpn(features_dict)
        
        return output

def create_model(num_classes):
    """
    Cria o modelo Mask R-CNN com um backbone ConvNeXt V2 customizado.
    
    Args:
        num_classes (int): O número de classes de objetos (incluindo o fundo).
        
    Returns:
        torch.nn.Module: O modelo Mask R-CNN pronto para treinamento.
    """
    # Carrega o backbone ConvNeXt V2-Large pré-treinado do timm
    # features_only=True e out_indices garantem que teremos 4 feature maps de saída
    backbone_timm = timm.create_model(
        'convnextv2_large.fcmae_ft_in22k_in1k',
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    
    # Envolve o backbone do timm na nossa classe de compatibilidade com FPN
    backbone_with_fpn = TimmToVisionFPN(backbone_timm)
    
    # Define o gerador de âncoras para a Region Proposal Network (RPN)
    # Estas são as "caixas de chute inicial" que o modelo usa para encontrar objetos
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    
    # Define as camadas que farão a predição final das caixas e máscaras
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=[ '0', '1', '2', '3' ],
        output_size=7,
        sampling_ratio=2
    )
    
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=[ '0', '1', '2', '3' ],
        output_size=14,
        sampling_ratio=2
    )

    # Cria o modelo Mask R-CNN final
    model = MaskRCNN(
        backbone_with_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler
    )
    
    return model


