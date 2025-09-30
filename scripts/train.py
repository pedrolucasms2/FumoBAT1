import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.v2 as T

# Adiciona o diretório raiz do projeto ao sys.path
# Isso permite que o script encontre os módulos na pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import PestDetectionDataset
from src.model import create_model
from src.utils import collate_fn, save_plots, calculate_count_mae

def get_transform(train):
    """Define as transformações de imagem (data augmentation)."""
    transforms = []
    # Converte a imagem para Tensor e normaliza para [0, 1]
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    if train:
        # Durante o treino, aplica um flip horizontal aleatório com 50% de chance
        # A API v2 automaticamente ajusta as bounding boxes
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main(args):
    """
    Função principal para treinar o modelo.
    """
    # Determina o dispositivo (GPU ou CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Usando o dispositivo: {device}")

    # Cria os diretórios de saída se não existirem
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True) # Cria a pasta de resultados

    # Caminhos para os dados
    train_data_path = os.path.join(args.data_path, 'train')
    val_data_path = os.path.join(args.data_path, 'valid')
    train_ann_file = os.path.join(train_data_path, '_annotations.coco.json')
    val_ann_file = os.path.join(val_data_path, '_annotations.coco.json')

    # Cria os datasets
    dataset_train = PestDetectionDataset(
        root_dir=train_data_path,
        annotation_file=train_ann_file,
        transforms=get_transform(train=True)
    )
    dataset_valid = PestDetectionDataset(
        root_dir=val_data_path,
        annotation_file=val_ann_file,
        transforms=get_transform(train=False)
    )

    # Cria os DataLoaders
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8, # Otimizado para A100
        collate_fn=collate_fn,
        pin_memory=True # Melhora a performance de transferência para a GPU
    )
    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=1, # Validação é feita imagem por imagem
        shuffle=False,
        num_workers=8, # Otimizado para A100
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Pega o número de classes do dataset de treino
    # Adicionamos 1 para a classe de fundo (background)
    num_classes = len(dataset_train.coco.cats) + 1

    # Cria o modelo
    model = create_model(num_classes=num_classes)
    model.to(device)

    # Configura o otimizador
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Listas para guardar o histórico de perdas e métricas
    train_loss_history = []
    val_mae_history = []

    print("\nIniciando o treinamento...")
    # Loop de treinamento
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(data_loader_train, desc=f"--- Época {epoch+1}/{args.epochs} ---")

        optimizer.zero_grad() # Zera o gradiente no início da época

        for i, (images, targets) in enumerate(progress_bar):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Calcula as perdas
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Normaliza a perda para a acumulação
            losses = losses / args.accumulation_steps
            
            # Backpropagation
            losses.backward()

            # Atualiza os pesos do modelo a cada 'accumulation_steps'
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += losses.item() * args.accumulation_steps # Re-escala a perda para o log
            progress_bar.set_postfix(loss=f"{losses.item() * args.accumulation_steps:.4f}")

        avg_epoch_loss = epoch_loss / len(data_loader_train)
        train_loss_history.append(avg_epoch_loss)
        print(f"Perda média da época: {avg_epoch_loss:.4f}")

        # Loop de validação
        model.eval()
        all_true_counts = []
        all_pred_counts = []
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader_valid, desc="Validando..."):
                images = list(image.to(device) for image in images)
                
                # Para validação, o modelo retorna as predições
                predictions = model(images)
                
                # targets[0] porque o batch_size da validação é 1
                true_count = len(targets[0]['labels'])
                pred_count = len(predictions[0]['labels'])
                
                all_true_counts.append(true_count)
                all_pred_counts.append(pred_count)

        mae = calculate_count_mae(all_true_counts, all_pred_counts)
        val_mae_history.append(mae)
        print(f"MAE de Contagem na Validação: {mae:.4f}")

        # Salva o checkpoint do modelo
        checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Modelo salvo em: {checkpoint_path}")
        print("-" * 50)

    print("Treinamento concluído.")

    # Salva os gráficos de desempenho na pasta de resultados
    save_plots(train_loss_history, val_mae_history, args.results_dir)
    print(f"Gráficos de desempenho salvos em {args.results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina um modelo Mask R-CNN para detecção de pragas.")
    parser.add_argument('--data_path', type=str, required=True, help='Caminho para a pasta do dataset (que contém as pastas train e valid).')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas de treinamento.')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamanho do batch de treinamento (real, por passo da GPU).')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Número de passos para acumular gradientes. Batch size efetivo = batch_size * accumulation_steps.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Taxa de aprendizado (learning rate). Um valor menor é mais seguro para fine-tuning.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Decaimento de peso (weight decay).')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Diretório para salvar os checkpoints do modelo.')
    parser.add_argument('--results_dir', type=str, default='results', help='Diretório para salvar os gráficos de resultados.')

    args = parser.parse_args()
    main(args)

