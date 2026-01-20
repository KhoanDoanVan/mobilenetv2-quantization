import torchvision.models as models
import torch
import os
from config import Config

if __name__ == "__main__":

    config = Config()
    
    # Create directories
    os.makedirs(config.FP32_MODEL_DIR, exist_ok=True)
    os.makedirs(config.QAT_MODEL_DIR, exist_ok=True)
    os.makedirs(config.QUANTIZED_MODEL_DIR, exist_ok=True)


    print("\n[STEP 1/7] Loading FP32 MobileNetV2...")
    fp32_model = models.mobilenet_v2(pretrained=True)
    fp32_model.eval()

    # Save FP32 model
    fp32_path = os.path.join(config.FP32_MODEL_DIR, 'mobilenet_v2_fp32.pth')
    torch.save(fp32_model.state_dict(), fp32_path)
    print(f"✓ FP32 model saved: {fp32_path}")