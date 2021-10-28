import sys

from setuptools import find_packages, setup
import argparse

import ai_ct_scans


if "--gpu_available" in sys.argv:
    g_index = sys.argv.index("--gpu_available")
    g_args = sys.argv[g_index : g_index + 2]

    parser = argparse.ArgumentParser(
        description="Command line utility for ai_ct_scans install"
    )
    parser.add_argument(
        "--gpu_available",
        type=str,
        default="yes",
        help="Whether you have a CUDA enabled GPU (yes/no)",
    )

    args = parser.parse_args(g_args)
    sys.argv.pop(g_index)
    sys.argv.pop(g_index)
else:

    class _args:
        def __init__(self):
            self.gpu_available = "no"

    args = _args()

dependency_links = []

if sys.platform == "linux":
    torch_versions = ["torch==1.9.0", "torchvision==0.10.0"]
    dependency_links.append(r"https://download.pytorch.org/whl/torch_stable.html")
elif sys.platform == "win32" and args.gpu_available == "no":
    torch_versions = ["torch==1.9.0", "torchvision==0.10.0"]
    dependency_links.append(r"https://download.pytorch.org/whl/torch_stable.html")
elif args.gpu_available == "yes":
    torch_versions = ["torch==1.9.0+cu102", "torchvision==0.10.0+cu102"]
    dependency_links.append(r"https://download.pytorch.org/whl/torch_stable.html")
else:
    torch_versions = ["torch==1.9.0", "torchvision==0.10.0"]
    dependency_links.append(r"https://download.pytorch.org/whl/torch_stable.html")

with open("README.md") as f:
    long_description = f.read()

with open("LICENCE", encoding="utf8") as f:
    licence = f.read()

tests_requires = [
    "flake8",
    "pytest==6.2.1",
    "pytest-cov",
    "pytest-assume",
    "pytest-qt",
    "mock",
]

experiments_requires = [
    "jupyter",
]

install_requires = (
    [
        "appdirs==1.4.4",
        "wheel",
        "scikit-build==0.11.1",
        "scikit-learn==0.24.2",
        "matplotlib",
        "scikit-learn",
        "opencv-contrib-python",
        "ruamel.yaml",
        "tqdm",
        "setuptools==51.1.0",
        "cython",
        "pydicom",
        "scipy",
        "scikit-image",
        "kneed",
        "PySide2",
        "pyqtgraph",
        "PyOpenGL",
        "requests",
        "timm",
        "pandas",
        "shiboken2",
    ]
    + torch_versions
    + tests_requires
)

setup(
    name="ai_ct_scans",
    description="Roke AI CT scan AI code",
    long_description=long_description,
    license=licence,
    version=ai_ct_scans.__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={"tests": tests_requires, "experiments": experiments_requires},
    dependency_links=dependency_links,
    python_requires=">=3.8.5",
    entry_points={
        "console_scripts": [
            "ai-ct-scan-tool=ai_ct_scans.scan_tool:main",
            "calculate-cpd-transform=ai_ct_scans.non_rigid_alignment:main",
        ],
    },
)
