from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).parent
version_namespace = {}
exec((ROOT / "src" / "_version.py").read_text(), version_namespace)


def discover_packages(source_root):
    packages = ["scCompass"]
    package_dir = {"scCompass": "src"}
    for init_file in sorted(source_root.rglob("__init__.py")):
        relative = init_file.parent.relative_to(source_root)
        if relative == Path("."):
            continue
        package = "scCompass." + ".".join(relative.parts)
        packages.append(package)
        package_dir[package] = str(Path("src") / relative)
    return packages, package_dir


packages, package_dir = discover_packages(ROOT / "src")


setup(
    name="scCompass",
    version=version_namespace["__version__"],
    description="AnnData-native single-cell metabolic flux analysis with Compass",
    packages=packages,
    package_dir=package_dir,
    include_package_data=True,
    package_data={"scCompass": ["core/Resources/**/*"]},
    entry_points={
        "console_scripts": [
            "sccompass = scCompass.cli:main",
            "sccompass-pipeline = scCompass.pipeline_cli:main",
        ]
    },
    python_requires=">=3.9",
    install_requires=[
        "anndata>=0.10",
        "numpy>=1.23",
        "pandas>=2.0",
        "scikit-learn>=1.2",
        "scipy>=1.9",
        "six>=1.10",
        "tqdm>=4.11",
    ],
    extras_require={
        "plot": [
            "matplotlib>=3.7",
            "networkx>=3.0",
            "scanpy>=1.9",
            "seaborn>=0.13",
        ],
        "solver": [
            "cplex>=22.1",
            "igraph>=0.10",
            "leidenalg>=0.8.2",
            "python-libsbml>=5.13",
        ],
        "test": ["pytest>=8"],
    },
    zip_safe=False,
)
