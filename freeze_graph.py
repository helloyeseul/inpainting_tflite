import os, argparse
import tensorflow as tf
from tensorflow.python.tools import freeze_graph


def get_output_path(output_dir):
    # check output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # freezed graph file name
    output_file = output_dir + "/frozen_graph.pb"

    return output_file


def freeze(input_graph, input_ckpt, output_path, output_nodes):

    print("Freezing graph start ...")

    freeze_graph.freeze_graph(
        input_graph=input_graph,
        input_saver="",
        input_binary=False,
        input_checkpoint=input_ckpt,
        output_node_names=output_nodes,
        restore_op_name="save/restore_all",
        filename_tensor_name="save/Const",
        output_graph=output_path,
        clear_devices=True,
        initializer_nodes="",
    )

    print("Freezing graph done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_graph", default="model_logs/test/graph.pbtxt", type=str)
    parser.add_argument("--output_dir", default="tflite", type=str)
    parser.add_argument("--input_ckpt", default="model_logs/places2/snap-0", type=str)
    args = parser.parse_args()

    output_path = get_output_path(args.output_dir)

    output_nodes = "saturate_cast"

    freeze(args.input_graph, args.input_ckpt, output_path, output_nodes)