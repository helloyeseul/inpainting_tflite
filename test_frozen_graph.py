import argparse

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_util


class Inpaint(object) :

    def __init__(self,model_filepath):
        self.model_filepath = model_filepath

        self.load_graph(model_filepath = self.model_filepath)


    def load_graph(self, model_filepath):

        print("loading model...")
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            self.input_image_ph = tf.placeholder(                                                                                                                                                                                                         
                tf.float32, name='input', shape=(1, args.image_height, args.image_width*2, 3))

            tf.import_graph_def(graph_def, {'input': self.input_image_ph})
            
        self.graph.finalize()

        self.sess = tf.Session(graph = self.graph)

        print("load model done.")


        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print(layer)


        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            print("Value - " )
            print(tensor_util.MakeNdarray(n.attr['value'].tensor))


    def test(self, input_image):

        output_node = self.graph.get_tensor_by_name("import/saturate_cast/Minimum:0")
        output = self.sess.run(output_node, feed_dict = {self.input_image_ph: input_image})

        return output



def test_inpaint_model(model_filepath, output_filepath, image, mask, width, height):

    tf.reset_default_graph()

    model = Inpaint(model_filepath = model_filepath)

    image = cv2.imread(image)
    mask = cv2.imread(mask)
    image = cv2.resize(image, (width, height))                                                                                                                                                                      
    mask = cv2.resize(mask, (width, height)) 
    assert image.shape == mask.shape
    
    h, w, _ = image.shape                                                                                                                                                                                                                 
    grid = 8                                                                                                                                                                                                                              
    image = image[:h//grid*grid, :w//grid*grid, :]                                                                                                                                                                                        
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    image = np.expand_dims(image, 0)                                                                                                                                                                                                      
    mask = np.expand_dims(mask, 0)

    input_image = np.concatenate([image, mask], axis=2)

    output = model.test(input_image)

    cv2.imwrite(output_filepath, output[0][:, :, ::-1])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_filepath', default='', type=str)
    parser.add_argument('--test_dir', default='', type=str)
    parser.add_argument('--image_height', default='', type=int)
    parser.add_argument('--image_width', default='', type=int)
    args = parser.parse_args()

    
    # prepare folder
    input_folder = args.test_dir + "/input"
    mask_folder = args.test_dir + "/mask"
    output_folder = args.test_dir + "/output_1"

    # check output dir 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # sort dir
    dir_files = os.listdir(input_folder)
    dir_files.sort()
            
    file_inter = dir_files[0]
    
    base = os.path.basename(file_inter)

    image = input_folder + "/" + base
    mask = mask_folder + "/" + base
    out = output_folder + "/" + base

    test_inpaint_model(args.model_filepath, out, image, mask, args.image_width, args.image_height)
    
