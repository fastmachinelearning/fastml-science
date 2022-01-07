# Fast Machine Learning in Science
Pytorch implementations of the FML-Science models, including a standard Pytorch (float) and  Quantized (via Xilinx's Brevitas library) implementations.

# keras-jet-classify

## Requirements:
Python 3.7

```
pip install -r requirements.txt
```

## Training:

```
python3 train.py -c <config.yml>>
```

Upon training completion, graphs for the ROC for each tagger, are saved to the output directory, along with a .h5 saved model file. 

The benchmark includes a float/unquantized 3 layer model as well as a uniformally quantized 6b model for each of pytorch and keras implementations