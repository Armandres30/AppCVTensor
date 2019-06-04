from imutils import paths
import argparse
import requests
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="urls.txt")
args = vars(ap.parse_args())
 
# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
total = 0

url="https://picsum.photos/200/300"
#for url in rows:
for i in range(1000):

	try:
		# try to download the image
		r = requests.get(url, timeout=60)
 
		# save the image to disk
		p = os.path.sep.join([args["output"], "{}.jpg".format(
			str(total).zfill(8))])
		f = open(p, "wb")
		f.write(r.content)
		f.close()
 
		# update the counter
		print("[INFO] downloaded: {}".format(p))
		total += 1
 
	# handle if any exceptions are thrown during the download process
	except:
		print("[INFO] error downloading {}...skipping".format(p))