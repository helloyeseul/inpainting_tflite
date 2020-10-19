import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import optimize_for_inference_lib


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]


def freeze_graph(model_dir, output_dir):
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # check output dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # freezed graph file name
    output_graph = output_dir + "/frozen_model.pb"

    input_node_names = "input"
    output_node_names = "inpaint_net/Tanh_1"

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)

    print("init saver done")

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    print("init graph def done")

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(","))

        # optimize graph
        optimize_for_inference_lib.optimize_for_inference(
            output_graph_def,
            input_node_names.split(","),
            output_node_names.split(","),
            tf.float32.as_datatype_enum)

        display_nodes(output_graph_def.node)
        
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        print("{} operations in the graph.".format(len(output_graph_def.node)))

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        help="Model directory to export")
    parser.add_argument("--output_dir", type=str,
                        help="Output graph directory to export")
    
    
    args = parser.parse_args()
        
    freeze_graph(args.model_dir, args.output_dir)
