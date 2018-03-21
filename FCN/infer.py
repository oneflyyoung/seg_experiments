import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('3_15.png')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((0,0,0))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('deploy.prototxt', 'train_iter_4000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)
#plt.imshow(out);
#plt.axis('off')
plt.savefig('test.png',out)

np.save("out.npy", out);
