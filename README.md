# Fast Machine Learning in Science
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
