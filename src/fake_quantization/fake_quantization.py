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


    def update_statistics(self, x):
        """
        Observer: collect min/max from activation
        """
        if not self.training:
            return
        
        with torch.no_grad():
            min_val = x.min()
            max_val = x.max()

            # Update running min/max with momentum
            momentum = 0.1
            self.min_val = (1 - momentum) * self.min_val + momentum * min_val
            self.max_val = (1 - momentum) * self.max_val + momentum * max_val

    
    def calculate_qparams(self):
        """
        Calculate scale and zero_point from observed statistics
        """

        if self.symmetric:
            max_abs = max(abs(self.min_val), abs(self.max_val))
            self.scale = max_abs / (2 ** (self.bit_width - 1) - 1)
            self.zero_point = torch.tensor(0.0)
        else:
            self.scale = (self.max_val - self.min_val) / (self.qmax - self.qmin)
            self.zero_point = self.qmin - self.min_val / self.scale
            self.zero_point = torch.round(self.zero_point).clamp(self.qmin, self.qmax)

        # Prevent scale = 0
        self.scale = torch.maximum(self.scale, torch.tensor(1e-8))

    
    def forward(self, x):
        """
        FakeQuantize forward pass

        1. Update statistics (training mode)
        2. Calculate qparams
        3. Quantize + Dequantize
        """
        if not self.enabled:
            return x
        
        # Step 1: Update statistics
        if self.training:
            self.update_statistics(x)
            self.calculate_qparams()

        # Step 2: Quantize
        x_int = torch.round(x / self.scale) + self.zero_point

        # Step 3: Clip to valid range
        x_int = torch.clamp(x_int, self.qmin, self.qmax)

        # Step 4: Dequantize (simulate INT8 -> FP32)
        x_fake_quant = (x_int - self.zero_point) * self.scale

        # Straight-Through Estimator: forward uses quantized, backward uses original
        # Gradient flows through as if no quantization

        return x_fake_quant