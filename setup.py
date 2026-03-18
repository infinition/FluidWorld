from setuptools import setup, find_packages

setup(
    name="fluidworld",
    version="0.1.0",
    description="PDE-based world model using reaction-diffusion dynamics",
    author="Fabien Polly",
    url="https://github.com/infinition/FluidWorld",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "numpy",
        "matplotlib",
        "tensorboard",
        "opencv-python",
        "Pillow",
    ],
)
