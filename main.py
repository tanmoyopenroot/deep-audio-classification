import argparse
import sys

from config import mp3_data_path
from config import wav_data_path
from config import spec_data_path
from config import spec_slice_path
from config import pickle_data_path 

from config import per_genre
from config import validation_ratio
from config import test_ratio

from config import spec_time

from config import 	desired_width
from config import img_height, img_width

from audioToSpecgram import createSpecgramFromAudio, createSlicesFromSpecgram
from util import createDatasetFromSlices

from model import trainModel

parser = argparse.ArgumentParser()
parser.add_argument(
	"mode", 
	help = "Create Dataset - create, Train CNN - train, Test CNN - test", 
	nargs = "+",
	choices = ["create", "train", "test"]
)

args = parser.parse_args()

print("CNN Config")
print("Validation Ratio : {0}".format(validation_ratio))
print("Test Ratio : {0}".format(test_ratio))
print("Specgram Time : {0}".format(spec_time))
print("Slices Per Genre : {0}".format(per_genre))

if "create" in args.mode:
	createDatasetFromSlices()

if "train" in args.mode:
	trainModel()

if "test" in args.mode:
	trainModel()
