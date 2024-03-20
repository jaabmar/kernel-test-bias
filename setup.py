import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="kernel_test_bias",
    version="0.0.1",
    description="Python implementation of the methods introduced in the paper: Hidden among subgroups: Detecting critical treatment effect bias in observational studies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/jaabmar/private-pgd",
    author="Javier Abad & Piersilvio de Bartolomeis & Konstantin Donhauser",
    author_email="javier.abadmartinez@ai.ethz.ch",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords="machine learning, treatment effect, kernel test, confounding, observational study, randomized trial",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy==1.26.2",
        "optax==0.1.7",
        "jax==0.4.23",
        "jaxlib==0.4.23",
        "scipy==1.11.4",
        "flax==0.7.5",
        "pandas==2.1.4",
        "scikit-learn==1.1.3",
        "scikit-uplift==0.5.1",
    ],
    python_requires="==3.9.18",
)
