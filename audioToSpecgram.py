import os

import cv2
import matplotlib.pyplot as plt

import pydub
import scipy.io.wavfile

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

def plotAudio(genre, file_name, channel_1, channel_2, audio_data, rate):
	# spec_data_path = path + "specgram/"
	fig,ax = plt.subplots(1)
	fig.subplots_adjust(left=-0.1, right=1, bottom=0, top=1)
	ax.axis('off')
	pxx, freqs, bins, im = ax.specgram(channel_1, Fs=rate)
	ax.axis('off')
	fig.savefig(spec_data_path + genre + "/" + file_name + "_channel_1.png", format='png', frameon='false')
	pxx, freqs, bins, im = ax.specgram(channel_2, Fs=rate)
	ax.axis('off')
	fig.savefig(spec_data_path + genre + "/" + file_name + "_channel_2.png", format='png', frameon='false')


def processAudio(genre, file_name, ext, sec):
	# mp3_data_path = path + "mp3/"
	# spec_data_path = path + "specgram/"
	# wav_data_path = path + "wav/"

	mp3 = pydub.AudioSegment.from_mp3(mp3_data_path + genre + "/" + file_name + ext)
	mp3.export(wav_data_path + genre + "/" + file_name + ".wav", format = "wav")
	rate, audio_data = scipy.io.wavfile.read(wav_data_path + genre + "/" + file_name + ".wav")

	wav_len = audio_data.shape[0] / rate
	num_channels = audio_data.shape[1]

	# print("Rate : {0}".format(rate))
	# print("Wav Length : {0}".format(wav_len))
	# print("Number Of Channel : {0}".format(num_channels))
	# print(audio_data[1000:2000,:])

	req_len = rate * sec

	channel_1 = audio_data[: req_len, 0]
	channel_2 = audio_data[: req_len, 1]

	plotAudio(genre, file_name, channel_1, channel_2, audio_data, rate)

def createSpecgramFromAudio():
	# mp3_data_path = path + "mp3/"
	# spec_data_path = path + "specgram/"
	# wav_data_path = path + "wav/"
	# spec_time = 15
	genres = os.listdir(mp3_data_path)
	for genre in genres:
		mp3_files = [file for file in os.listdir(mp3_data_path + genre + "/") if file.endswith(".mp3")]
		if not os.path.exists(spec_data_path + genre + "/"):
			try:
				os.makedirs(spec_data_path + genre + "/")
			except Exception as e:
				raise e

		if not os.path.exists(wav_data_path + genre + "/"):
			try:
				os.makedirs(wav_data_path + genre + "/")
			except Exception as e:
				raise e

		print("Processing Genre : {0}".format(genre))
		for file in mp3_files:
			processAudio(genre ,file.split(".mp3")[0], ".mp3", spec_time)

		print("Genre : {0}, Number Of Files : {1}".format(genre, len(mp3_files)))

def sliceSpecgram(genre, file_name, ext):
	# spec_data_path = path + "specgram/"
	# spec_slice_path = path + "slices/"
	# desired_width = 128

	print("Slice Specgram : {0}".format(spec_data_path + genre + "/" + file_name + ext))
	img = cv2.imread(spec_data_path + genre + "/" + file_name + ext)
	height, width, channel = img.shape

	num_slices = int(width / desired_width)

	for i in range(num_slices):
		print("Creating Slice : {0} / {1} for : {2}".format((i + 1), num_slices, file_name + ext))
		start_pixel = i * desired_width
		img_temp = img[0 : height, start_pixel : start_pixel + desired_width ]
		cv2.imwrite(spec_slice_path + genre + "/" + file_name + "_" + str(i) + "_" + ext, img_temp)


def createSlicesFromSpecgram():
	# spec_data_path = path + "specgram/"
	# spec_slice_path = path + "slices/"

	genres = os.listdir(spec_data_path)
	for genre in genres:
		genre_spec_files = [file for file in os.listdir(spec_data_path + genre + "/") if file.endswith(".png")]
		
		if not os.path.exists(spec_slice_path + genre + "/"):
			try:
				os.makedirs(spec_slice_path + genre + "/")
			except Exception as e:
				raise e

		for file in genre_spec_files:
			sliceSpecgram(genre, file.split(".png")[0], ".png")


# createSpecgramFromAudio()
# createSlicesFromSpecgram()
