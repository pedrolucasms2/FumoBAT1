import sys
import os

# Adiciona a pasta raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as T
from torch.optim.lr_scheduler import LinearLR
# Importações para Treinamento com Precisão Mista (AMP)
from torch.cuda.amp import GradScaler, autocast

from src.dataset import PestDetectionDataset
from src.model import create_model
from src.utils import collate_fn, save_plots, calculate_count_mae

def get_transforms(is_train, image_size=640):
    """Retorna as transformações de imagem apropriadas."""
    transforms = []
    transforms.append(T.Resize((image_size, image_size)))
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    if is_train:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    return T.Compose(transforms)

def main(args):
    # --- Configuração do Dispositivo ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        # Fallback para CPU se nem CUDA nem MPS estiverem disponíveis
        device = torch.device('cpu')
    print(f"Usando o dispositivo: {device}\n")

    image_size = 640

    # --- Preparação dos Dados ---
    train_data_path = os.path.join(args.data_path, 'train')
    val_data_path = os.path.join(args.data_path, 'valid')
    train_ann_file = os.path.join(train_data_path, '_annotations.coco.json')
    val_ann_file = os.path.join(val_data_path, '_annotations.coco.json')

    dataset_train = PestDetectionDataset(
        root_dir=train_data_path,
        annotation_file=train_ann_file,
        transforms=get_transforms(is_train=True, image_size=image_size)
    )
    dataset_valid = PestDetectionDataset(
        root_dir=val_data_path,
        annotation_file=val_ann_file,
        transforms=get_transforms(is_train=False, image_size=image_size)
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # --- Criação do Modelo ---
    num_classes = len(dataset_train.get_classes()) + 1
    model = create_model(num_classes=num_classes, image_size=image_size)
    model.to(device)

    # --- Configuração do Otimizador, Agendador e GradScaler ---
    params_backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    params_head = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {'params': params_backbone, 'lr': args.lr_backbone},
            {'params': params_head, 'lr': args.lr_head}
        ],
        weight_decay=1e-4
    )
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=len(data_loader_train))
    
    # **GRADSCALER ADICIONADO AQUI PARA ESTABILIZAR O TREINAMENTO**
    scaler = GradScaler()

    # --- Loop de Treinamento ---
    train_losses = []
    val_maes = []
    print("Iniciando o treinamento...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(data_loader_train, desc=f"--- Época {epoch}/{args.epochs} ---")
        for i, (images, targets) in enumerate(progress_bar):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Usa autocast para executar o forward pass com precisão mista
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"\nERRO: Perda infinita ou NaN detetada no passo {i}. Pulando atualização.")
                print(f"Detalhes da Perda: {loss_dict}")
                optimizer.zero_grad()
                continue

            loss_value = losses.item()
            
            # Usa o scaler para escalar a perda antes do backward pass
            scaler.scale(losses / args.accumulation_steps).backward()

            if (i + 1) % args.accumulation_steps == 0:
                # Usa o scaler para desescalar os gradientes antes do clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Usa o scaler para executar o passo do otimizador
                scaler.step(optimizer)
                # Atualiza a escala para o próximo passo
                scaler.update()
                
                optimizer.zero_grad()
            
            if epoch == 1:
                # A LÓGICA DO SCHEDULER FOI CORRIGIDA PARA ATUALIZAR APENAS APÓS O PASSO DO OTIMIZADOR
                # A verificação de (i+1) % acc.. garante que isto aconteça
                if (i + 1) % args.accumulation_steps == 0:
                    warmup_scheduler.step()

            running_loss += loss_value
            progress_bar.set_postfix(loss=f"{loss_value:.4f}")

        epoch_loss = running_loss / len(data_loader_train)
        train_losses.append(epoch_loss)

        val_mae = calculate_count_mae(model, data_loader_valid, device)
        val_maes.append(val_mae)

        print(f"Fim da Época {epoch}: Perda de Treino = {epoch_loss:.4f}, MAE de Validação = {val_mae:.4f}")

        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        torch.save(model.state_dict(), f'outputs/model_epoch_{epoch}.pth')

    print("\nTreinamento concluído.")
    if not os.path.exists('results'):
        os.makedirs('results')
    save_plots(train_losses, val_maes, 'results/performance_plots.png')
    print("Gráficos de desempenho salvos em 'results/performance_plots.png'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treina um modelo de detecção de pragas.")
    parser.add_argument('--data_path', type=str, required=True, help='Caminho para a pasta principal do dataset.')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas de treinamento.')
    parser.add_argument('--batch_size', type=int, default=2, help='Tamanho do batch de imagens por passo.')
    parser.add_argument('--num_workers', type=int, default=8, help='Número de workers para carregar os dados.')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Passos para acumulação de gradiente (batch_size_efetivo = 16).')
    parser.add_argument('--lr_head', type=float, default=1e-4, help='Taxa de aprendizado para a cabeça do modelo.')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Taxa de aprendizado para o backbone.')
    
    args = parser.parse_args()
    main(args)

