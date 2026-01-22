
# Fuse Conv Bn
- This is Fuse Technical Conv2d + BatchNorm2d

Conv2d $\rightarrow$ BatchNorm2d
$\downarrow$ Fuse
Conv2d (new, merge BN)

$\rightarrow$ Not change output, but:
- Few layers
- Faster
- More quantize INT8
- Cpacity for deploy (mobile/edge/CoreML/TFLite/TensorRT)

# Why does Conv+BN can fuse?
## Conv2d is linearity:
y = w * x + b

## BatchNorm (in inference) also linearity:
BN(y) = γ * (y − μ) / √(σ² + ε) + β


$\rightarrow$ Linear + Linear = Linear

### So we should merge BN with Conv by:
- adjust Weight
- adjust Bias


## Mathematics

### BN in inference:
y = γ * (x − running_mean) / sqrt(running_var + eps) + β

replace x = w*x + b from Conv to:
$\rightarrow$ Fused weight: w' = γ * w / sqrt(var + eps)

$\rightarrow$ Fused bias: b' = γ * (b − mean) / sqrt(var + eps) + β

#### No longer need BN


## Why we need to do this things?

### 1. Speed of Inference
- Few more op
- Don't need to load BN Params
- Cache friendly

### 2. Essensially for INT8 Quantization
- Quantization Conv + BN -> wrong scale
- Fuse Before -> qunatize 1 Conv Only

### 3. Deploy mobile / edge
- MobileNetV2
- CoreML
- TFLite
- NNAPI
$\rightarrow$ BN sometimes will be remove entire