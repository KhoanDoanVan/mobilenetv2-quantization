import os


class Config:

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    FP32_MODEL_DIR = os.path.join(MODELS_DIR, 'fp32')
    QAT_MODEL_DIR = os.path.join(MODELS_DIR, 'qat')
    QUANTIZED_MODEL_DIR = os.path.join(MODELS_DIR, 'quantized')
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'imagenet_sample')

    # Model settings
    MODEL_NAME = 'mobilenet_v2'
    INPUT_SIZE = (224, 224)
    NUM_CLASSES = 1000

    # QAT settings
    QAT_EPOCHS = 10 # 5-20 epochs recommend
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4 # Lower LR for fine-tuning
    WEIGHT_DECAY = 1e-5

    # Quantization settings
    QUANT_BIT_WIDTH = 8
    OBSERVER_TYPE = 'minmax' # 'minmax' or 'histogram'
    QUANT_DELAY = 1 # Start quantization after N epochs

    # COREML settings
    COREML_MODEL_NAME = 'mobilenetV2_quant_int8.mlmodel'
    USE_FP16_FOR_SENSITIVE_LAYERS = True # Keep FP16 for first/last layers

    # Device evaluation
    TEST_BATCH_SIZE = 1
    NUM_TEST_SAMPLE = 100
