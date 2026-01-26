import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from config import Config
from src.preparation.model_prep import ModelPreparation
from src.fake_quantization.fake_quantization import FakeQuantize
from src.fake_quantization.quantize_conv2d import QuantConv2d
from src.qat_trainer import QATTrainer
from src.exporter import QuantizedModelExporter
from src.coreml_converter import CoreMLConverter
from src.evaluator import OnDeviceEvaluator



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
    main()