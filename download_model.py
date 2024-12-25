import os
import gdown
import torch as t

def download_pretrained_model():
    os.makedirs('checkpoints', exist_ok=True)
    model_path = 'checkpoints/squeezenet.pth'
    
    if not os.path.exists(model_path):
        print("Downloading pre-trained model...")
        # You would need to provide a valid URL to your pre-trained model
        url = "YOUR_PRETRAINED_MODEL_URL"
        gdown.download(url, model_path, quiet=False)
    
    return model_path

if __name__ == "__main__":
    download_pretrained_model() 