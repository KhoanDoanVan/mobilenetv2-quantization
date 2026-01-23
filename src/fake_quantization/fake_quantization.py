import torch
import torch.nn as nn


class FakeQuantize(nn.Module):
    """
    STEP 3: Fake Quantization Module (QAT-style)

    Simulate quantization in training:
    1. Quantize: q = round(x/scale) + zero_point
    2. Dequantize: x' = (q - zero_point) * scale
    3. Backward pass still use FP32

    Straight-Through Estimator (STE): Gradient Flow Through Rounding
    """

    def __init__(
            self,
            bit_width=8,
            observer_type='minmax',
            symmetric=False
    ):
        
        super().__init__()

        self.bit_width = bit_width
        self.observer_type = observer_type
        self.symmetric = symmetric

        # Quantization range
        if symmetric:
            self.qmin = -(2 ** (bit_width - 1))
            self.qmax = 2 ** (bit_width - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bit_width - 1

        # Learnable scale and zero_point (or observe from data)
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))

        # Running statistics for observer
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))

        self.enabled = True