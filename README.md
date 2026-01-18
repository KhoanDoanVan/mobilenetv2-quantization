#  Pipeline 7 Steps

### Using Quantization-Aware Training (QAT) for iOS

### STEP 1: Load FP32 MobileNetV2
```python
model = models.mobilenet_v2(pretrained=True)
```
- Using pretrained ImageNet weights
- MobileNetV2 optimized for mobile

### STEP 2: Fuse Conv + BN
```python
# Mathematics of fusion:
# BN: y = γ(x - μ)/√(σ² + ε) + β
# Fused: w' = γw/√(σ² + ε)
#        b' = γ(b - μ)/√(σ² + ε) + β

fused_model = ModelPreparation.fuse_model(model)
```
**Benefits:**
- Reduce operations
- Improvement numerical stability
- Prepare for quantization

### STEP 3: Insert FakeQuant
```python
# FakeQuantization simulates INT8:
# q = round(x / scale) + zero_point
# x' = (q - zero_point) * scale  # Dequantize

# Gradient flows through (STE)
```
**FakeQuant Module:**
- Forward: quantize → dequantize
- Backward: gradient like FP32
- Learn scale & zero_point from data

### STEP 4: Fine-tune (QAT)
```python
# Lower learning rate (1e-4)
# Train 5-20 epochs
# Warmup: disable quant for epoch begins

trainer = QATTrainer(model, config)
trainer.train(train_loader, val_loader)
```
**QAT Strategy:**
- Epoch 0-1: Warmup (FP32)
- Epoch 2+: Enable FakeQuant
- Model learning adapt with quantization noise

### STEP 5: Export INT8 Graph
```python
# Convert FakeQuant → Real INT8
traced = torch.jit.trace(model, example_input)
torch.jit.save(traced, 'model_int8.pt')
```
**Output:** TorchScript with INT8 ops

### STEP 6: Convert to CoreML
```python
mlmodel = ct.convert(
    traced_model,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16
)

# Mixed precision:
# - First/last layers: FP16
# - Middle layers: INT8
```
**Mixed Precision Strategy:**
- **FP16**: Input/output layers (sensitive)
- **INT8**: Hidden layers (majority computation)

### STEP 7: Evaluate On-Device
```python
# Metrics:
# - Accuracy: FP32 vs INT8
# - Latency: Inference time
# - Model size: Compression ratio
```