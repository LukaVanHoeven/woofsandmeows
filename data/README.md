# Data directory

Download the data yourself and place it in this directory.

Data can be downloaded [here](https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs?resource=download).

The directory should look like this after downloading:
```
data/
  |- cats_dogs/
  |    |- test/
  |    |    |- ...
  |    |- train/
  |    |    |- ...
  |    |- cat_1.wav
  |    |- ...
  |    |- dog_barking_112
  |- README.md
  |- train_test_split.csv
  |- utils.py
```

Note that we make our own train and test split, so the `test/` and `train/` folders can be disregarded.