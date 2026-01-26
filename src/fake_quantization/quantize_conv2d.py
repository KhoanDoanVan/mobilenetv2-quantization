import torch.nn as nn
from fake_quantization.fake_quantization import FakeQuantize


class QuantConv2d(nn.Module):
    """
    Conv2d with FakeQuantization for weights and activations
    """

    def __init__(
            self, 
            conv_layer, 
            bit_width=8
    ):
        super().__init__()
        self.conv = conv_layer
        self.weight_fake_quant = FakeQuantize(
            bit_width=bit_width,
            symmetric=True
        )
        self.activation_fake_quant = FakeQuantize(
            bit_width=bit_width,
            symmetric=False
        )

    
    def forward(self, x):
        # Quantize weights
        quantized_weight = self.weight_fake_quant(self.conv.weight)

        # Temporary replace weight
        original_weight = self.conv.weight.data
        self.conv.weight.data = quantized_weight

        # Forward pass
        out = self.conv(x)

        # Restore original weight
        self.conv.weight.data = original_weight

        # Quantize activation
        out = self.activation_fake_quant(out)

        return out