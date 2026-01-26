import torch
import coremltools as ct
from coremltools.models.neural_network.quantization_utils import quantize_weights



class CoreMLConverter:
    """
    STEP 6: Convert to CoreML (keep FP16 where needed)

    Strategy:
    - First/Last layers: FP16 (sensitive to quantization)
    - Middle layers: INT8 (most computation)
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config


    def convert(self, save_path, example_input):
        """
        Convert Pytorch -> CoreML with mixed precision
        """

        self.model.eval()

        # Trace model
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, example_input)

        # Convert to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input",
                    shape=example_input.shape,
                    dtype=float
                )
            ],
            convert_to="mlprogram", # newest format
            compute_precision=ct.precision.FLOAT16 # Default FP16
        )

        # Apply INT8 quantization (automatic mixed precision)
        if self.config.USE_FP16_FOR_SENSITIVE_LAYERS:
            # Quantize with skip first/last layers
            mlmodel = self._apply_mixed_precision_quantization(mlmodel)
        else:
            # Full INT8 quantization
            mlmodel = quantize_weights(mlmodel, nbits=8)


        # Save
        mlmodel.save(save_path)
        print(f"✓ CoreML model saved: {save_path}")

        # Print model info
        self._print_model_info(mlmodel, save_path)

        return mlmodel


    def _apply_mixed_precision_quantization(self, mlmodel):
        """
        Apply quantization but still hold FP16 for first/last layers
        """

        # CoreMLTools will automatic detect and skip sensitive layers
        # when using compute_precision=Float16 + quantize_weights
        
        config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel"
        )

        mlmodel = ct.optimize.coreml.linear_quantize_weights(
            mlmodel,
            config=config
        )
        
        return mlmodel


    
    def _print_model_info(self, mlmodel, save_path):
        """
        Print info of model
        """
        import os
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        
        print(f"\n{'='*60}")
        print("CoreML Model Information")
        print(f"{'='*60}")
        print(f"Size: {size_mb:.2f} MB")
        print(f"Input: {mlmodel.get_spec().description.input}")
        print(f"Output: {mlmodel.get_spec().description.output}")
        print(f"{'='*60}\n")