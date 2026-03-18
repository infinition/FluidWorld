from setuptools import setup, find_packages

setup(
    name="fluidworld",
    version="0.2.0",
    description="PDE-based world model using reaction-diffusion dynamics",
    author="Fabien Polly",
    url="https://github.com/infinition/FluidWorld",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "matplotlib>=3.5",
        "tensorboard>=2.10",
        "opencv-python>=4.5",
        "Pillow>=9.0",
        "scikit-learn>=1.0",
    ],
)
