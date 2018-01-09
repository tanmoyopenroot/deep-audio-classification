# Image Shape
img_height = 480
img_width = 128 

# Define paths for files
path = "data/"

mp3_data_path = path + "mp3/"
wav_data_path = path + "wav/"
spec_data_path = path + "specgram/"
spec_slice_path = path + "slices/"
pickle_data_path = path + "pickle/"

# specgram Time
spec_time = 15

# Desired Width
desired_width = 128

# Dataset parameters
per_genre = 100
validation_ratio = 0.1
test_ratio = 0.1

# Train Parameter
num_channel = 1
pixel_depth = 255
num_labels = 7
batch_size = 70
num_epoch = 20
eval_frequency = 5