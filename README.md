# Fast Machine Learning in Science
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Keras implementations of the FML-Science models, including a standard Keras (float) and  QKeras implementations.


# keras-jet-classify

## Requirements:
Python 3.7

```
conda env create -f environment.yml
```

## Training:

```
python3 train.py -c <config.yml>>
```

Upon training completion, graphs for the ROC for each tagger, are saved to the output directory, along with a .h5 saved model file. 

The benchmark includes a float/unquantized 3 layer model as well as a uniformally quantized 6b model

## Sample Runs

### Training Float Baseline:

```
python3 train.py -c float_baseline.yml
```
![Alt text](keras-jet-classify/model/float_baseline/keras_roc_curve.png?raw=true "Float Baseline ROC Curve")

### Training Quantized Baseline:

```
python3 train.py -c float_baseline.yml
```
![Alt text](keras-jet-classify/model/quantized_baseline/keras_roc_curve.png?raw=true "Quantized Baseline ROC Curve")