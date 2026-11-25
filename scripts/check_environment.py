
"""
ACAL-AD environment verification.

Checks Python, required libraries, PyTorch and hardware.
"""

import importlib
import platform
import sys


REQUIRED_PACKAGES = {
    "torch": "PyTorch",
    "torchvision": "Torchvision",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "pandas": "Pandas",
    "sklearn": "Scikit-learn",
    "PIL": "Pillow",
    "cv2": "OpenCV",
    "yaml": "PyYAML",
    "dotenv": "Python-dotenv",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "tqdm": "TQDM",
    "tensorboard": "TensorBoard",
    "streamlit": "Streamlit",
}


def check_python():
    """Display Python version and executable path."""

    print("\n[1] PYTHON ENVIRONMENT")
    print("-" * 50)

    print("Python version:", platform.python_version())
    print("Python path:", sys.executable)

    if sys.version_info < (3, 10):
        print("WARNING: Python 3.10 or newer is recommended.")
    else:
        print("Python version is supported.")


def check_packages():
    """Check whether required packages can be imported."""

    print("\n[2] REQUIRED PACKAGES")
    print("-" * 50)

    missing_packages = []

    for module_name, display_name in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "Installed")

            print(f"[OK] {display_name}: {version}")

        except ImportError:
            print(f"[MISSING] {display_name}")
            missing_packages.append(display_name)

    return missing_packages


def check_pytorch():
    """Check PyTorch and CUDA availability."""

    print("\n[3] PYTORCH AND HARDWARE")
    print("-" * 50)

    try:
        import torch

        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())

        if torch.cuda.is_available():
            print("GPU count:", torch.cuda.device_count())

            for index in range(torch.cuda.device_count()):
                print(
                    f"GPU {index}: "
                    f"{torch.cuda.get_device_name(index)}"
                )
        else:
            print("No CUDA GPU detected on this machine.")
            print("This is acceptable for laptop development.")
            print("Kaggle GPU will be checked separately.")

        return True

    except ImportError:
        print("PyTorch is not installed.")
        return False


def main():
    """Execute all environment checks."""

    print("=" * 60)
    print("ACAL-AD ENVIRONMENT CHECK")
    print("=" * 60)

    check_python()

    missing_packages = check_packages()
    pytorch_available = check_pytorch()

    print("\n[4] FINAL STATUS")
    print("-" * 50)

    if not missing_packages and pytorch_available:
        print("All required packages are available.")
        print("Environment verification successful.")

    else:
        print("Some requirements need attention.")

        if missing_packages:
            print("Missing packages:", ", ".join(missing_packages))

        if not pytorch_available:
            print("Install or repair PyTorch.")

    print("=" * 60)


if __name__ == "__main__":
    main()
