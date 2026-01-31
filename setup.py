"""
⚠️ DEPRECATED: This file is kept for backward compatibility only.

🔹 Use pyproject.toml for all package configuration (PEP 621 compliant).
🔹 Modern installations should use: pip install -e .
"""

from setuptools import setup

# All configuration is now in pyproject.toml (PEP 621)
# This file exists only for compatibility with older pip versions (<21.0)
setup()
    install_requires=[
        # Core ML frameworks
        "torch>=2.5.0",
        "transformers>=4.45.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        
        # Federated learning
        "flwr[simulation]>=1.11.0",
        
        # Blockchain integration
        "substrate-interface>=1.7.9",
        
        # Security & cryptography
        "cryptography>=42.0.4",
        
        # Data processing
        "numpy>=2.1.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.5.0",
        
        # Configuration & logging
        "pyyaml>=6.0.2",
        "python-dotenv>=1.0.1",
        "structlog>=24.4.0",
        "rich>=13.9.0",
        
        # Validation
        "pydantic>=2.9.0",
        "pydantic-settings>=2.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-asyncio>=0.24.0",
            "pytest-cov>=5.0.0",
            "pytest-mock>=3.14.0",
            "ruff>=0.7.0",
            "mypy>=1.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nawal-server=nawal.orchestrator:main",
            "nawal-client=nawal.client.trainer:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
