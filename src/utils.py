import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def collate_fn(batch):
    """
    Função de agrupamento para o DataLoader do PyTorch.
    Como as imagens em um lote podem ter um número diferente de objetos,
    simplesmente agrupamos as amostras em uma lista de tuplas.
    """
    return tuple(zip(*batch))

def save_plots(train_losses, valid_maes, output_dir):
    """
    Salva os gráficos de perda de treino e MAE de validação.
    """
    # Gráfico da Perda de Treinamento
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss per Epoch')
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

    # Gráfico do MAE de Validação
    plt.figure(figsize=(10, 5))
    plt.plot(valid_maes, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (Count)')
    plt.legend()
    plt.title('Validation MAE per Epoch')
    plt.savefig(os.path.join(output_dir, 'mae_plot.png'))
    plt.close()

def calculate_count_mae(predictions, targets):
    """
    Calcula o Erro Médio Absoluto (MAE) da contagem de objetos.
    """
    pred_counts = [len(p['boxes']) for p in predictions]
    target_counts = [len(t['boxes']) for t in targets]
    
    errors = [abs(p - t) for p, t in zip(pred_counts, target_counts)]
    
    if not errors:
        return 0.0
        
    return sum(errors) / len(errors)

def draw_predictions_on_image(image_np, predictions, confidence_threshold=0.5):
    """
    Desenha as caixas delimitadoras e scores em uma imagem.
    
    Args:
        image_np (np.array): A imagem no formato NumPy (OpenCV).
        predictions (list): A lista de predições do modelo.
        
    Returns:
        np.array: A imagem com as predições desenhadas.
    """
    img_copy = image_np.copy()

    for pred in predictions:
        # A lógica da SAHI retorna um objeto de predição diferente
        # do que o Torchvision retorna durante a validação.
        # Este if/else lida com os dois formatos.
        if isinstance(pred, dict): # Formato do Torchvision (validação)
            scores = pred['scores']
            boxes = pred['boxes']
        else: # Formato da SAHI (inferência)
            scores = [pred.score.value]
            bbox_obj = pred.bbox
            boxes = [[bbox_obj.minx, bbox_obj.miny, bbox_obj.maxx, bbox_obj.maxy]]

        for i, score in enumerate(scores):
            if score < confidence_threshold:
                continue
            
            box = boxes[i]
            # Converte para numpy se for tensor
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()

            x_min, y_min, x_max, y_max = map(int, box)
            
            # Desenha o retângulo
            cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Adiciona o texto com a confiança
            label = f"Pest: {score:.2f}"
            cv2.putText(img_copy, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_copy

