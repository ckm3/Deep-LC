# Deep-LC

![PyPI](https://img.shields.io/pypi/v/deep-lightcurve?style=flat) [![Image](https://img.shields.io/badge/arXiv-2311.08080-blue)](https://arxiv.org/abs/2311.08080)

``Deep-LC``  is open-source and intended for the classification of light curves (LCs) in a gernaral purpose. It utilizes a weakly supervised object detection algorithm to automatically zoom in on the LC and power spectrum (PS) to extract local features. This eliminates the need for manual feature extraction and allows it to be applied to both space- and ground-based observations, as well as multiband LCs with large gaps and nonuniform sampling.

The implenmentaion and performace detail can be found in our paper.

LC component processing animation of TIC 470109695, which is a rotating variable star. Our model can automatically zoom in on the LC and PS to extract local features.

![LC component processing](docs/source/lc.gif)

PS component processing animation of KIC 12268220, which is an eclipsing bianry with a $\delta$ Scuti pulsating primary star.

![PS component processing](docs/source/ps.gif)

## Introduction

``Deep-LC`` is easy to install with pip:
```
pip install deep-lightcurve
```

Or install the development version from source:
```
git clone https://github.com/ckm3/Deep-LC.git
cd Deep-LC
pip install -e .
```

## Quickstart

```python
from deep_lc import DeepLC

# Load model
dl_combined = DeepLC(combined_model="model_path")

# Load light curve
lc = np.load("your_light_curve_path")

# Predict as show intermediate results
prediction, figs = dl_combined.predict(lc, show_intermediate_results=True)

```

Please visit the [quickstart page](https://deep-lightcurve.readthedocs.io/en/latest/notebooks/Quickstart.html) for details. All models are avilable on [Zenodo](https://zenodo.org/records/10081600).

## Citing

If you are using Deep-LC in your research, please cite our paper and add a footnote of this Github project.
```
@misc{cui2023identifying,
      title={Identifying Light-curve Signals with a Deep Learning Based Object Detection Algorithm. II. A General Light Curve Classification Framework}, 
      author={Kaiming Cui and D. J. Armstrong and Fabo Feng},
      year={2023},
      eprint={2311.08080},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM}
}
```