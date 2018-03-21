from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as k
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
#import get_result

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

input_ = "/home/abc/Unet/convertModel/"
weight_file = "Unet.hdf5"
num_output = 1
ascii_flag = True
node_name = 'output_node'
output_graph_name = 'constant_graph_weights.pb'

smooth = 1
def dice_coef(y_true, y_pred):
	y_true_f = k.flatten(y_true)
	y_pred_f = k.flatten(y_pred)
	intersection = k.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

output_ = input_ + 'tensorflow_model/'
if not os.path.isdir(output_):
	os.mkdir(output_)
weight_file_path = osp.join(input_, weight_file)

print (weight_file_path)

k.set_learning_phase(0)
#net_model = load_model(weight_file)
net_model = load_model(weight_file_path, custom_objects = {"dice_coef_loss":dice_coef_loss, "dice_coef":dice_coef})

pred = [None] * num_output
pred_node_name = [None] * num_output

for i in range(num_output):
	pred_node_name[i] = node_name + str(i)
	pred[i] = tf.identity(net_model.output[i], name = pred_node_name[i])

print("output: ", pred_node_name)

sess = k.get_session()

#if ascii_flag:
#	f = 'only_the_graph_def.pb.ascii'
#	tf.train.write_graph(sess.graph.as_graph_def(), output_, as_text = True)
#	print('saved in ascii format at: ', osp.join(output_, f))

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_name)
graph_io.write_graph(constant_graph, output_, output_graph_name, as_text = False)
print('saved graph as: ', osp.join(output_, output_graph_name))
