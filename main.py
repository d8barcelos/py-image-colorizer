import numpy as np
import cv2

prototxt_path = 'models\colorization_deploy_v1.prototxt'
model_path = 'models\dummy.caffemodel'
kernel_path = 'models/pts_in_hull.npy'
image_path = 'Lion.jpg'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rn")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

bw_image = cv2.imread(image_path)
normalized = bw_image.astype("float32") / 255.0

