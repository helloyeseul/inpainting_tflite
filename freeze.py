import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import optimize_for_inference_lib


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print("%d %s %s" % (i, node.name, node.op))
        [print(u"└─── %d ─ %s" % (i, n)) for i, n in enumerate(node.input)]


def get_output_file(output_dir):
    # check output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # freezed graph file name
    output_file = output_dir + "/frozen_model.pb"

    return output_file


def freeze_graph(ckpt_dir, output_file, input_nodes, output_nodes):
    # load checkpoint
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    ckpt_path = ckpt.model_checkpoint_path

    saver = tf.train.import_meta_graph(ckpt_path + ".meta", clear_devices=True)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)

        # freeze graph
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_nodes.split(",")
        )

        # print nodes info
        # display_nodes(output_graph_def.node)

        # optimize graph - only take input~output graph
        optimize_for_inference_lib.optimize_for_inference(
            output_graph_def,
            input_nodes.split(","),
            output_nodes.split(","),
            tf.uint8.as_datatype_enum,
        )

        # save graph to file
        with tf.gfile.GFile(output_file, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        print("Saving graph done.")
        print("{} operations in the graph.".format(len(output_graph_def.node)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="model_logs/test", type=str)
    parser.add_argument("--output_dir", default="tflite", type=str)
    args = parser.parse_args()

    output_file = get_output_file(args.output_dir)

    input_node_names = "input"
    output_node_names = "saturate_cast"
    # output_node_names = "ReverseV2"

    freeze_graph(
        ckpt_dir=args.ckpt_dir,
        output_file=output_file,
        input_nodes=input_node_names,
        output_nodes=output_node_names,
    )