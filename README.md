# XAI-TREPAN-for-Regression
TREPAN algorithm modified to extract rules from trained ANNs using regression trees

TREPAN is an algorithm proposed by Craven and Shavlik [1] for constructing decision trees to explain the predictions of trained Artificial Neural Networks. The algorithm is modified here to extract rules in a regression setting by using regression trees.

To run:
python3 xai.py

The other source files are decomposed into two parts: The decision tree code which implements the main TREPAN algorithm and the neural network code, which consists of training code for a 4 layer ANN with 100 units per layer.
