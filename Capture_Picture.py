from tcp_pickle_stream import listener
from PIL import Image
import cv2

l = listener()

while(1):
    frame = l.get_frame()
    img = Image.fromarray(frame)
    cv2.imshow('test', frame)
    cv2.waitKey(1)
     





