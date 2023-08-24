# -*- coding: utf-8 -*-

import setuptools

setuptools.setup(
    name="Deep-LC",
    version="0.1.0",
    author="Kaiming Cui",
    author_email="cuikaiming@sjtu.edu.cn",
    description="A general Light curve classification framework based on deep learning.",
    packages=setuptools.find_packages(where="src"),
    long_description="""
        # Deep-LC

        ``Deep-LC`` is open-source and intended for the classification of light curves in a gernaral purpose. 

        ## Installation
        ``Deep-LC`` is easy to install with pip:
        ```
        pip install deep-lc
        ```
        ## Quickstart

        Please visit the [quickstart page](https://deep-lc.readthedocs.io/en/latest/Quickstart.html) for details.
    """,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    include_package_data=True,
    url="https://github.com/ckm3/Deep-LC",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"],
    python_requires='>=3.6.1',
    install_requires=["torch", "torchvision", "lightkurve>=2.0", "gatspy"],
)