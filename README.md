# Deep-LC

``Deep-LC``  is open-source and intended for the classification of light curves (LCs) in a gernaral purpose. It utilizes a weakly supervised object detection algorithm to automatically zoom in on the LC and power spectrum (PS) to extract local features. This eliminates the need for manual feature extraction and allows it to be applied to both space- and ground-based observations, as well as multiband LCs with large gaps and nonuniform sampling.

The implenmentaion and performace detail can be found in our paper.

LC component processing animation of TIC 470109695, which is a rotating variable star.

![LC component processing](docs/source/lc.gif)

PS component processing animation of KIC 12268220, which is an eclipsing bianry with a $\delta$ Scuti pulsating primary star.

![PS component processing](docs/source/ps.gif)

## Introduction

``Deep-LC`` is easy to install with pip:
```
pip install deep-lc
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

Please visit the document for details.

## Citing

If you find Deep-LC helpful in your research, please cite our paper and add a footnote of this Github project.