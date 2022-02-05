import numpy as np
from data import load_data
from data import shuffle
from image import export_image

trainset = load_data(train = True)
X, y = shuffle(trainset)

preview = []
for i in range(10):
    for j, x in enumerate(X):
        if y[j] == i:
            preview.append(x.reshape(28, 28))
            break

for i, img in enumerate(preview):
    export_image(img, name = f'class_{i}.png')