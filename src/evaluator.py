import torch
import numpy as np
from tqdm import tqdm


class OnDeviceEvaluator:
    """
    STEP 7: Evaluate on-device performance

    Metrics:
    - Accuracy
    - Latency
    - Memory usage
    - Comparison with FP32
    """

    def __init__(
            self,
            config
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def evaluate_accuracy(
            self,
            model,
            dataloader
    ):
        """
        Evaluate Accuracy
        """

        model.eval()
        model.to(self.device)

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc='Evaluating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        return accuracy
    

    def measure_latency(
            self,
            model,
            example_input,
            num_runs=100
    ):
        """
        Measure Inference Latency
        """

        model.eval()
        model.to(self.device)
        example_input = example_input.to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(example_input)

        # Measure
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        import time
        latencies = []

        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(example_input)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                latencies.append((time.time() - start) * 1000) # ms

        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies)
        }
    

    def compare_models(
            self,
            fp32_model,
            int8_model,
            dataloader,
            example_input
    ):
        """
        Comparison FP32 and INT8
        """

        print(f"\n{'='*60}")
        print("Model Comparison: FP32 vs INT8")
        print(f"{'='*60}\n")

        # Accuracy
        print("Evaluating FP32 accuracy...")
        fp32_acc = self.evaluate_accuracy(fp32_model, dataloader)

        print("Evaluating INT8 accuracy...")
        int8_acc = self.evaluate_accuracy(int8_model, dataloader)

        print(f"\nAccuracy:")
        print(f"  FP32: {fp32_acc:.2f}%")
        print(f"  INT8: {int8_acc:.2f}%")
        print(f"  Drop: {fp32_acc - int8_acc:.2f}%")

        # Latency
        print("\nMeasuring FP32 latency...")
        fp32_latency = self.measure_latency(fp32_model, example_input)

        print("Measuring INT8 latency...")
        int8_latency = self.measure_latency(int8_model, example_input)

        print(f"\nLatency (ms):")
        print(f"  FP32: {fp32_latency['mean']:.2f} ± {fp32_latency['std']:.2f}")
        print(f"  INT8: {int8_latency['mean']:.2f} ± {int8_latency['std']:.2f}")
        print(f"  Speedup: {fp32_latency['mean'] / int8_latency['mean']:.2f}x")

        print(f"\n{'='*60}\n")

        return {
            'fp32_acc': fp32_acc,
            'int8_acc': int8_acc,
            'fp32_latency': fp32_latency,
            'int8_latency': int8_latency
        }