from ultralytics import YOLO
import os
import shutil
from pathlib import Path
import random

def prepare_dataset():
    # Define paths
    dataset_path = Path('dataset/images')
    output_path = Path('prepared_dataset')
    
    # Create train/val/test directories
    train_path = output_path / 'train'
    val_path = output_path / 'val'
    test_path = output_path / 'test'
    
    for path in [train_path, val_path, test_path]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    
    # Get all category directories
    categories = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    for category in categories:
        # Create category directories in train/val/test
        (train_path / category.name).mkdir(exist_ok=True)
        (val_path / category.name).mkdir(exist_ok=True)
        (test_path / category.name).mkdir(exist_ok=True)
        
        # Get all images from both default and real_world subdirectories
        images = []
        for subdir in ['default', 'real_world']:
            subdir_path = category / subdir
            if subdir_path.exists():
                images.extend(list(subdir_path.glob('*.jpg')))
                images.extend(list(subdir_path.glob('*.jpeg')))
                images.extend(list(subdir_path.glob('*.png')))
        
        # Shuffle images
        random.shuffle(images)
        
        # Split images (70% train, 20% val, 10% test)
        n_images = len(images)
        n_train = int(0.7 * n_images)
        n_val = int(0.2 * n_images)
        
        # Copy images to respective directories
        for idx, img_path in enumerate(images):
            if idx < n_train:
                dest_dir = train_path
            elif idx < n_train + n_val:
                dest_dir = val_path
            else:
                dest_dir = test_path
            
            shutil.copy2(img_path, dest_dir / category.name / img_path.name)
    
    return output_path

def train_model(data_path):
    # Load the YOLOv8n-cls model
    model = YOLO('yolov8n-cls.pt')
    
    # Train the model
    results = model.train(
        data=str(data_path),
        epochs=50,
        imgsz=224,
        batch=16,
        name='waste_classifier'
    )
    
    return results

def validate_model():
    # Load the trained model
    model = YOLO('runs/classify/waste_classifier/weights/best.pt')
    
    # Validate the model
    results = model.val()
    return results

if __name__ == "__main__":
    print("Preparing dataset...")
    prepared_data_path = prepare_dataset()
    print("Dataset preparation completed!")
    
    print("Starting model training...")
    train_results = train_model(prepared_data_path)
    print("Training completed!")
    
    print("Starting validation...")
    val_results = validate_model()
    print("Validation completed!")
    print(f"Validation Accuracy: {val_results.top1}")  # top-1 accuracy
