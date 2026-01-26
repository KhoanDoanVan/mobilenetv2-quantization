import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from config import Config
from preparation.model_prep import ModelPreparation
from fake_quantization.fake_quantization import FakeQuantize
from fake_quantization.quantize_conv2d import QuantConv2d
from qat_trainer import QATTrainer
from exporter import QuantizedModelExporter
from coreml_converter import CoreMLConverter
from evaluator import OnDeviceEvaluator



class DummyDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random image và label
        img = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
        label = torch.randint(0, 1000, (1,)).item()
        img = self.transform(img)
        return img, label



# ImageNet-1K mini subset dataset
class ImageNetMiniDataset(Dataset):
    """
    Tải ImageNet-1K validation set subset (~50MB, 100 ảnh)
    Sử dụng Hugging Face datasets để download tự động
    """
    def __init__(self, split='validation', num_samples=100):
        from datasets import load_dataset
        
        print(f"Downloading ImageNet-1K {split} subset...")
        print(f"Size: ~50MB for {num_samples} images")
        
        # Load từ Hugging Face (public dataset)
        # imagenet-1k có sẵn validation set (50,000 ảnh)
        # Ta chỉ lấy subset nhỏ
        try:
            dataset = load_dataset(
                "imagenet-1k",
                split=f"{split}[:{num_samples}]",
                trust_remote_code=True
            )
            self.dataset = dataset
            print(f"✓ Loaded {len(self.dataset)} images")
        except Exception as e:
            print(f"⚠ Không thể load ImageNet từ HuggingFace: {e}")
            print("  Falling back to ImageFolder local...")
            self.dataset = None
        
        # Transform pipeline (ImageNet standard)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        if self.dataset is not None:
            return len(self.dataset)
        return 0
    
    def __getitem__(self, idx):
        if self.dataset is not None:
            item = self.dataset[idx]
            image = item['image'].convert('RGB')
            label = item['label']
            
            image = self.transform(image)
            return image, label
        
        # Fallback nếu không load được
        return torch.zeros(3, 224, 224), 0


class LocalImageNetDataset(Dataset):
    """
    Fallback: Load từ thư mục local nếu có
    
    Structure:
    data/imagenet_sample/
    ├── train/
    │   ├── n01440764/  # tench
    │   ├── n01443537/  # goldfish
    │   └── ...
    └── val/
        ├── n01440764/
        ├── n01443537/
        └── ...
    """
    def __init__(self, root_dir, split='val'):
        from torchvision.datasets import ImageFolder
        
        data_dir = os.path.join(root_dir, split)
        
        if not os.path.exists(data_dir):
            raise ValueError(
                f"Local ImageNet not found at {data_dir}\n"
                "Please either:\n"
                "1. Set up HuggingFace token for imagenet-1k\n"
                "2. Download ImageNet sample manually\n"
                "3. Use CIFAR-10 instead (automatic download)"
            )
        
        self.dataset = ImageFolder(
            data_dir,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class CIFAR10Dataset(Dataset):
    """
    Alternative: CIFAR-10 dataset (~170MB total, tự động download)
    Dùng khi không có ImageNet
    
    Note: CIFAR-10 chỉ 32x32, sẽ resize lên 224x224
    Accuracy sẽ thấp hơn nhưng đủ để demo QAT pipeline
    """
    def __init__(self, root='./data', train=True):
        from torchvision.datasets import CIFAR10
        
        print(f"Downloading CIFAR-10 {'train' if train else 'test'} set...")
        print("Size: ~170MB (will be cached)")
        
        self.dataset = CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(224),  # Resize từ 32x32 lên 224x224
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        )
        
        print(f"✓ Loaded {len(self.dataset)} images")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Map CIFAR-10 labels (10 classes) to ImageNet space (1000 classes)
        # Chỉ để demo, production cần train lại classifier head
        return img, label


def prepare_datasets(config, use_cifar10_fallback=True):
    """
    Chuẩn bị datasets với fallback strategy:
    1. Try ImageNet từ HuggingFace (tốt nhất)
    2. Try local ImageNet nếu có
    3. Fallback sang CIFAR-10 (automatic download)
    """
    
    print("\n" + "="*60)
    print("Preparing Datasets")
    print("="*60)
    
    # Strategy 1: HuggingFace ImageNet (preferred)
    try:
        train_dataset = ImageNetMiniDataset(
            split='train',
            num_samples=1000  # ~50MB
        )
        val_dataset = ImageNetMiniDataset(
            split='validation',
            num_samples=200   # ~10MB
        )
        
        if len(train_dataset) > 0 and len(val_dataset) > 0:
            print("✓ Using ImageNet-1K from HuggingFace")
            return train_dataset, val_dataset
    except Exception as e:
        print(f"⚠ Cannot load from HuggingFace: {e}")
    
    # Strategy 2: Local ImageNet
    try:
        train_dataset = LocalImageNetDataset(
            config.DATA_DIR,
            split='train'
        )
        val_dataset = LocalImageNetDataset(
            config.DATA_DIR,
            split='val'
        )
        print("✓ Using local ImageNet")
        return train_dataset, val_dataset
    except Exception as e:
        print(f"⚠ Local ImageNet not available: {e}")
    
    # Strategy 3: CIFAR-10 fallback
    if use_cifar10_fallback:
        print("\n→ Falling back to CIFAR-10 (automatic download)")
        print("  Note: CIFAR-10 is 32x32, will be resized to 224x224")
        print("  Accuracy will be lower, but pipeline works the same\n")
        
        train_dataset = CIFAR10Dataset(
            root=config.DATA_DIR,
            train=True
        )
        val_dataset = CIFAR10Dataset(
            root=config.DATA_DIR,
            train=False
        )
        
        print("✓ Using CIFAR-10")
        return train_dataset, val_dataset
    
    raise RuntimeError(
        "No dataset available! Please:\n"
        "1. Set up HuggingFace token for ImageNet, OR\n"
        "2. Download ImageNet sample to data/imagenet_sample/, OR\n"
        "3. Enable CIFAR-10 fallback (set use_cifar10_fallback=True)"
    )


def insert_fake_quant_layers(model, bit_width=8):
    """
    Insert FakeQuantization in all of Conv Layers
    """

    for name, module in model.named_children():
        if isinstance(module, torch.nn.Conv2d):
            # Replace Conv2d to QuantConv2d
            setattr(model, name, QuantConv2d(module, bit_width))
        else:
            # Recursively apply
            insert_fake_quant_layers(module, bit_width)

    return model



def main():
    """
    7-STEP QAT PIPELINE
    ====================
    
    1. Load FP32 MobileNetV2
    2. Fuse Conv + BN
    3. Insert FakeQuant (QAT-style)
    4. Fine-tune 5-20 epochs (QAT)
    5. Export INT8-friendly graph
    6. Convert to CoreML (keep FP16 where needed)
    7. Evaluate on-device
    """

    print("="*70)
    print("QUANTIZATION-AWARE TRAINING (QAT) PIPELINE FOR iOS")
    print("="*70)

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

    print("\n[STEP 2/7] Fusing Conv + BatchNorm layers...")
    fused_model = ModelPreparation.fuse_conv_bn(fp32_model)

    # Count fused layers
    num_conv = sum(
        1 for m in fp32_model.modules() if isinstance(m, torch.nn.Conv2d)
    )
    num_bn = sum(
        1 for m in fp32_model.modules() if isinstance(m, torch.nn.BatchNorm2d)
    )
    num_fused_conv = sum(
        1 for m in fp32_model.modules() if isinstance(m, torch.nn.Conv2d)
    )

    print(f"✓ Original: {num_conv} Conv + {num_bn} BN")
    print(f"✓ After fusion: {num_fused_conv} Conv (with fused BN)")
    print(f"✓ Reduced {num_conv + num_bn - num_fused_conv} layers")


    print("\n[STEP 3/7] Inserting FakeQuantization modules...")
    qat_model = insert_fake_quant_layers(
        fused_model,
        bit_width=config.QUANT_BIT_WIDTH
    )

    num_quant_layers = sum(
        1 for m in qat_model.modules() if isinstance(m, QuantConv2d)
    )

    print(f"✓ Inserted FakeQuant into {num_quant_layers} layers")
    print(f"✓ Quantization: INT{config.QUANT_BIT_WIDTH}")

    print("\n[STEP 4/7] Quantization-Aware Training...")
    # Prepare dataloaders
    train_dataset = DummyDataset(num_samples=1000)
    val_dataset = DummyDataset(num_samples=200)

    train_dataset, val_dataset = prepare_datasets(config, use_cifar10_fallback=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # Train
    trainer = QATTrainer(qat_model, config)
    best_acc = trainer.train(train_loader, val_loader)

    print(f"\n✓ QAT Training completed!")
    print(f"✓ Best validation accuracy: {best_acc:.2f}%")

    # Load best checkpoint
    checkpoint = torch.load(f'{config.QAT_MODEL_DIR}/best_qat_model.pth')
    qat_model.load_state_dict(checkpoint['model_state_dict'])

    print("\n[STEP 5/7] Exporting INT8-friendly graph...")
    example_input = torch.randn(1, 3, 224, 224)
    torchscript_path = os.path.join(
        config.QUANTIZED_MODEL_DIR,
        'mobilenet_v2_int8.pt'
    )

    traced_model = QuantizedModelExporter.export_to_torchscript(
        qat_model,
        example_input,
        torchscript_path
    )

    # Extract quantization parameters
    quant_params = QuantizedModelExporter.extract_quantization_params(qat_model)
    print(f"✓ Extracted quantization params from {len(quant_params)} layers")


    print("\n[STEP 6/7] Converting to CoreML...")
    coreml_path = os.path.join(
        config.QUANTIZED_MODEL_DIR,
        config.COREML_MODEL_NAME
    )
    converter = CoreMLConverter(qat_model, config)

    mlmodel = converter.convert(coreml_path, example_input)

    if config.USE_FP16_FOR_SENSITIVE_LAYERS:
        print("✓ Mixed precision: FP16 for first/last, INT8 for middle layers")
    else:
        print("✓ Full INT8 quantization")


    print("\n[STEP 7/7] Evaluating on-device performance...")
    evaluator = OnDeviceEvaluator(config)

    # Compare FP32 vs QAT INT8
    results = evaluator.compare_models(
        fp32_model,
        qat_model,
        val_loader,
        example_input
    )

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "="*70)
    print("QAT PIPELINE COMPLETED!")
    print("="*70)
    
    print("\n📊 Results Summary:")
    print(f"  • Accuracy (FP32): {results['fp32_acc']:.2f}%")
    print(f"  • Accuracy (INT8): {results['int8_acc']:.2f}%")
    print(f"  • Accuracy Drop: {results['fp32_acc'] - results['int8_acc']:.2f}%")
    print(f"  • Speedup: {results['fp32_latency']['mean'] / results['int8_latency']['mean']:.2f}x")
    
    print("\n📦 Output Files:")
    print(f"  • FP32 Model: {fp32_path}")
    print(f"  • QAT Checkpoint: {config.QAT_MODEL_DIR}/best_qat_model.pth")
    print(f"  • TorchScript: {torchscript_path}")
    print(f"  • CoreML: {coreml_path}")
    
    print("\n📱 iOS Deployment:")
    print("  1. Kéo file .mlmodel vào Xcode project")
    print("  2. Xcode tự động generate Swift wrapper")
    print("  3. Sử dụng:")
    print("""
    import Vision
    import CoreML
    
    let model = try! MobileNetV2_QAT_INT8(configuration: .init())
    let vnModel = try! VNCoreMLModel(for: model.model)
    
    let request = VNCoreMLRequest(model: vnModel) { request, error in
        guard let results = request.results as? [VNClassificationObservation]
        else { return }
        
        // Top prediction
        if let topResult = results.first {
            print("Predicted: \\(topResult.identifier)")
            print("Confidence: \\(topResult.confidence)")
        }
    }
    
    // Run inference
    let handler = VNImageRequestHandler(ciImage: inputImage)
    try? handler.perform([request])
    """)
    
    print("\n" + "="*70)
    print("✓ Ready for production deployment!")
    print("="*70)


if __name__ == "__main__":

    config = Config()
    
    train_dataset, val_dataset = prepare_datasets(config, use_cifar10_fallback=True)
    # main()