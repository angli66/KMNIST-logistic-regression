# Logistic/Softmax Regression on KMNIST Dataset
This repository contains three experiment that perform logistic/softmax regression on KMNIST dataset. The first experiment perform a logistic regression on class 0 (hiragana お(o)) and class 2 (hiragana ま(ma)) of KMNIST dataset. The second experiment perform a logistic regression on class 2 (hiragana ま(ma)) and class 6 (hiragana す(su)) of KMNIST dataset. The third experiment perform a softmax regression on all 10 classes of the KMNIST dataset. The corresponding source code is `experiment1.py`, `experiment2.py` and `experiment3.py`.

For every experiment, the hyperparameters includes:
- `--batch-size`: batch size of the stochastic gradient descent, default is 64
- `--epochs`: number of epochs to train, default is 100 (or less if early stop happens in cross validation)
- `--learning-rate`: learning rate of the stochastic gradient descent, default is 0.001
- `--z-score`: use z-score normalization on the dataset, default is min-max normalization
- `--k-folds`: number of folds for cross-validation, default is 10
- `--validation`: enable cross-validation mode, default is False

To run the code, first run `get_data.sh` to download the data. An example run of the experiment would be like:

    python3 experiment1.py --batch-size=32 --learning-rate=0.01 --validation=True

This will perform a cross validation on the set of hyperparameters (batch_size = 32, epochs = 100, learning_rate = 0.01, normalization=max_min_normalization, k_folds=10).

Run experiment 3 with cross-validation turned off will also create 10 visualization of the trained weights, where `weights_i.png` refers to weights for class i.

Run `random_sample.py` will generate 10 random sampled images from each of the 10 classes of the train set.