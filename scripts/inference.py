import os
import sys
import argparse
import torch
from PIL import Image
from sahi.predict import get_sliced_prediction

# Adiciona o diretório raiz do projeto ao sys.path
# Isso permite que o script encontre os módulos na pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import create_model
from src.utils import draw_predictions_on_image

def main(args):
    """
    Função principal para executar a inferência com tiling.
    """
    # Determina o dispositivo (GPU ou CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Usando o dispositivo: {device}")

    # Cria o diretório de saída se não existir
    os.makedirs(args.output_dir, exist_ok=True)

    # Carrega o modelo treinado
    # Supondo que o número de classes salvo no checkpoint seja o mesmo do treino
    # NOTA: Para um sistema mais robusto, o número de classes deveria ser salvo junto com o modelo.
    # Por simplicidade, estamos assumindo que é 2 (praga + fundo).
    num_classes = 2 
    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Carregando imagem de: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")

    print("Executando inferência com SAHI (tiling)...")
    # Executa a predição com tiling usando a biblioteca SAHI
    result = get_sliced_prediction(
        image,
        model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    print(f"Foram encontradas {len(result.object_prediction_list)} detecções.")

    # Desenha as predições na imagem original
    image_with_preds = draw_predictions_on_image(image, result.object_prediction_list)

    # Salva a imagem resultante
    output_filename = os.path.basename(args.image_path)
    output_path = os.path.join(args.output_dir, output_filename)
    image_with_preds.save(output_path)
    print(f"Imagem com as detecções salva em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa a inferência em uma imagem usando um modelo treinado.")
    parser.add_argument('--model_path', type=str, required=True, help='Caminho para o arquivo .pth do modelo treinado.')
    parser.add_argument('--image_path', type=str, required=True, help='Caminho para a imagem de entrada.')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Diretório para salvar a imagem com as detecções.')

    args = parser.parse_args()
    main(args)

