"""Confirm TensorFlow can train on the GPU (run in WSL after `source scripts/wsl_env.sh`).

    python -m scripts.check_gpu
"""

import time

import keras
import tensorflow as tf


def main() -> None:
    print(f"TensorFlow {tf.__version__}, Keras {keras.__version__}")
    build = tf.sysconfig.get_build_info()
    print(f"Built for CUDA {build.get('cuda_version')}, cuDNN {build.get('cudnn_version')}")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("No GPU visible to TensorFlow. Did you `source scripts/wsl_env.sh`?")
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        print(f"GPU: {details.get('device_name')} (compute capability {details.get('compute_capability')})")

    x = tf.random.normal((4096, 4096))
    for dtype in (tf.float32, tf.float16):
        a = tf.cast(x, dtype)
        tf.linalg.matmul(a, a)  # warm-up
        start = time.perf_counter()
        for _ in range(50):
            y = tf.linalg.matmul(a, a)
        _ = y.numpy()
        seconds = time.perf_counter() - start
        tflops = 50 * 2 * 4096**3 / seconds / 1e12
        print(f"{dtype.name} matmul on {y.device}: {tflops:.1f} TFLOPS")


if __name__ == "__main__":
    main()
