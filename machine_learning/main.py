import torch
from ultralytics import YOLO
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

def train_yolov8(config_path, epochs=50, batch_size=16, img_size=640, device='cuda'):
    # Currently using the last trained model. If not available, you can use a default
    # 'yolov8n.pt' or 'yolo11n.pt' input file
    model = YOLO("last.pt")  
    
    # Adds support to multiple GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    current_time = datetime.now().strftime('%H:%M:%S')
    print("Início: ", current_time)

    model.train(data=config_path, epochs=epochs, imgsz=img_size, batch=batch_size, device=device)

    # Mixed Precision to improve performance
    scaler = GradScaler() 

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        data_loader = model.train_dataloader(pin_memory=True, workers=4)

        for batch_idx, (imgs, targets, paths, shapes) in enumerate(data_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            model.optimizer.zero_grad()
            
            with autocast():
                loss = model(imgs, targets)
            
            # Backpropagation with Mixed Precision
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
            
            running_loss += loss.item()

            if batch_idx % 50 == 0: 
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(data_loader)}, Loss: {running_loss / (batch_idx + 1)}")
        
        model.save(f"yolov8_epoch_{epoch+1}.pt")

    current_time = datetime.now().strftime('%H:%M:%S')
    print("Treinamento concluído! ", current_time)

config_path = "./custom_dataset.yaml"

if __name__ == '__main__':
    train_yolov8(config_path, epochs=30, batch_size=16, img_size=640, device=0)