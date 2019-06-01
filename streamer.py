from time import sleep
from tcp_pickle_stream import streamer
import picamera
import picamera.array
from PIL import Image
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

s = streamer("192.168.93.26")

with picamera.PiCamera() as camera:
	with picamera.array.PiRGBArray(camera) as stream:
		camera.resolution = (320, 240)		
		for frame in camera.capture_continuous(stream, format="bgr", use_video_port=True):
			image = frame.array

			//get prediction of image
			img = image.load_img(image, target_size=(224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)

			preds = model.predict(x)
			//get prediction of image

			s.send_frame(preds)
			stream.truncate(0)

	


