"""Setup file for Dignity package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme = Path("README.md").read_text() if Path("README.md").exists() else ""

setup(
    name="dignity",
    version="0.1.0",
    description="Streamlined sequence modeling for transactional behavior patterns",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Dignity Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dignity-train=dignity.train.cli:main",
            "dignity-export=dignity.export.to_onnx:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
