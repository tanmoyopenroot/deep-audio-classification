# Deep Audio Classification
A pipeline to build a dataset from your own music library and use it to fill the missing genres

Required install:

```
numpy
open cv
tensorflow
```

- path = "data/"

- mp3_data_path = path + "mp3/"
- wav_data_path = path + "wav/"
- spec_data_path = path + "specgram/"
- spec_slice_path = path + "slices/"
- pickle_data_path = path + "pickle/"

To create the song slices :
```
python main.py create
```

To train the classifier
```
python main.py train
```

To test the classifier
```
python main.py test
```