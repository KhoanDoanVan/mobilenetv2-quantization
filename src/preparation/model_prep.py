import torch
import torch.nn as nn
from copy import deepcopy



class ModelPreparation:
    """
    STEP 2: Fuse Conv + BatchNorm Layers

    Conv + BN fusion:
    - Reduce operations
    - Prepare for quantization
    - Math: y = gamma * (Conv(x) - mu) / sqrt(var + eps) + beta
    """

    @staticmethod
    def fuse_conv_bn(
        conv,
        bn
    ):
        """
        Merge Fuse Conv2d and BatchNorm2d into only 1 Conv2d
        
        Maths:
        BN: y = gamma * (x - mu) / sqrt(var + eps) + beta
        Fused weight: w' = gamma * w / sqrt(var + eps)
        Fused bias: b' = gamma * (b - mu) / sqrt(var + eps) + beta
        """

        # Get parameters
        conv_weight = conv.weight.clone()
        conv_bias = conv.bias.clone() if conv.bias is not None else torch.zeros(conv.out_channels)

        bn_weight = bn.weight.clone()
        bn_bias = bn.bias.clone()
        bn_running_mean = bn.running_mean.clone()
        bn_running_var = bn.running_var.clone()
        bn_eps = bn.eps


        # Calculate fused parameters
        bn_std = torch.sqrt(bn_running_var + bn_eps)
        scale = bn_weight / bn_std


        # Fuse weights: w' = gamma * w / sqrt(var + eps)
        fused_weight = conv_weight * scale.view(-1, 1, 1, 1)

        # Fuse bias: b' = gamma * (b - mu) / sqrt(var + eps) + beta
        fused_bias = (conv_bias - bn_running_mean) * scale + bn_bias

        # Create new fused conv layer
        fused_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            padding_mode=conv.padding_mode
        )

        fused_conv.weight.data = fused_weight
        fused_conv.bias.data = fused_bias

        return fused_conv
    


    @staticmethod
    def fuse_model(model):
        """
        Fuse entire Conv+BN pairs in model
        """

        fused_model = deepcopy(model)

        modules = list(fused_model.named_children())

        for name, module in modules:
            if isinstance(module, nn.Sequential):
                fused_layers = []
                i = 0
                layers = list(module.children())

                while i < len(layers):
                    # Check if Conv + BN pattern
                    if i < len(layers) - 1:
                        if isinstance(layers[i], nn.Conv2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
                            # Fuse Conv + BN
                            fused_conv = ModelPreparation.fuse_conv_bn(
                                layers[i],
                                layers[i + 1]
                            )
                            fused_layers.append(fused_conv)
                            i += 2
                            continue

                    # don't fuse, will remain layer
                    fused_layers.append(layers[i])
                    i += 1

                # Replace module with fused version
                setattr(fused_model, name, nn.Sequential(*fused_layers))

        
        return fused_model