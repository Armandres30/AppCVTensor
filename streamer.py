from time import sleep
from tcp_pickle_stream import streamer
import picamera
import picamera.array
from PIL import Image

s = streamer("192.168.93.26")

with picamera.PiCamera() as camera:
	with picamera.array.PiRGBArray(camera) as stream:
		camera.resolution = (320, 240)		
		for frame in camera.capture_continuous(stream, format="bgr", use_video_port=True):
			image = frame.array
			s.send_frame(image)
			stream.truncate(0)

	


