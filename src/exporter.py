import torch
import torch.nn as nn


class QuantizedModelExporter:
    """
    STEP 5: Export INT8-friendly graph

    Convert FakeQuantization -> Real INT8 operations
    """

    @staticmethod
    def convert_fake_quant_to_int8(model):
        """
        Replace FakeQuant modules with actual INT8 operations

        Pytorch will automatic convert since export to TorchScript
        """

        # Prepare model for export
        model.eval()

        # Freeze BN statistics
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.track_running_stats = False

        return model
    

    @staticmethod
    def export_to_torchscript(model, example_input, save_path):
        """
        Export to TorchScript format
        """

        model = QuantizedModelExporter.convert_fake_quant_to_int8(model)

        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)


        # Save
        torch.jit.save(traced_model, save_path)
        print(f"✓ Exported TorchScript model: {save_path}")

        return traced_model
    

    @staticmethod
    def extract_quantization_params(model):
        """
        Extract scale and zero_point from all of FakeQuant modules
        """

        quant_params = {}

        for name, module in model.named_modules():
            if hasattr(module, 'scale'):
                quant_params[name] = {
                    'scale': module.scale.item(),
                    'zero_point': module.zero_point.item()
                }


        return quant_params