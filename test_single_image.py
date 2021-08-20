import os, argparse
import cv2
import numpy as np
import neuralgym as ng
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import optimize_for_inference_lib

from inpaint_model import InpaintCAModel


def get_dirs(data_dir, index):
    # prepare directories
    input_dir = data_dir + "/input"
    mask_dir = data_dir + "/mask"
    output_dir = data_dir + "/output"

    # check output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dir_files = os.listdir(input_dir)
    dir_files.sort()

    base = os.path.basename(dir_files[index])

    image = input_dir + "/" + base
    mask = mask_dir + "/" + base
    out = output_dir + "/" + base

    return image, mask, out


def get_input_image(image_path, mask_path, image_height, image_width):
    # load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # resize image
    image = cv2.resize(image, (image_width, image_height))
    mask = cv2.resize(mask, (image_width, image_height))

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[: h // grid * grid, : w // grid * grid, :]
    mask = mask[: h // grid * grid, : w // grid * grid, :]

    print("image shape: {}".format(image.shape))

    # reshape image and mask : [h, w, 3] to [1, h, w, 3]
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)

    # concat [image]+[mask] to make input image
    # input_image shape : [1, h, w * 2, 3]
    input_image = np.concatenate([image, mask], axis=2)

    print("input image shape: {}".format(input_image.shape))

    return input_image


def test_single_image(image_path, mask_path, output_path, image_height, image_width, ckpt_dir):
    # generate input image
    input_image = get_input_image(image_path, mask_path, image_height, image_width)

    # start sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    # inpaint model
    model = InpaintCAModel()

    # input node placeholder
    input_image_ph = tf.placeholder(
        tf.float32, name="input", shape=(1, image_height, image_width * 2, 3)
    )

    output = model.build_server_graph(input_image_ph)
    output = (output + 1.0) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)

    # load variables from checkpoint
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(ckpt_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))

    sess.run(assign_ops)

    # input, output
    print("input = {}".format(input_image_ph))
    print("output = {}".format(output))

    # run model
    print("Running model ...")
    result = sess.run(output, feed_dict={input_image_ph: input_image})
    print("Running model done.")

    # save output image to file
    print("Saving output to {}".format(output_path))
    cv2.imwrite(output_path, result[0][:, :, ::-1])
    print("Saving output done.")

    save_tensorboard_log(sess, "trensorboard")

    return sess


def save_checkpoint(sess, ckpt_path):
    # save checkpoint
    print("Saving checkpoint to {}".format(ckpt_path))
    saver = tf.train.Saver().save(sess, ckpt_path)
    print("Saving checkpoint done.")


def save_tensorboard_log(sess, output_dir):
    # save graph for Tensorboard
    print("Saving Tensorboard log file to {}".format(output_dir))
    tf.summary.FileWriter(output_dir, sess.graph)
    print("Saving Tensorboard log file done.")


def save_graph(sess, output_dir):
    # save graph to file
    print("Saving graph file to {}".format(output_dir))
    tf.train.write_graph(sess.graph_def, output_dir + "/", "graph.pbtxt")
    tf.train.write_graph(sess.graph_def, output_dir + "/", "graph.pb")
    print("Saving graph file done.")


def save_frozen_graph(sess):
    output_file = "tflite"

    # check output directory exists
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    output_file = output_file + "/frozen_graph.pb"

    input_node_names = "input"
    output_node_names = "saturate_cast"
    # output_node_names = "ReverseV2"

    input_graph_def = tf.get_default_graph().as_graph_def()

    # freeze graph
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(",")
    )

    # print nodes info
    # display_nodes(output_graph_def.node)

    # optimize graph - only take input~output graph
    optimize_for_inference_lib.optimize_for_inference(
        output_graph_def,
        input_node_names.split(","),
        output_node_names.split(","),
        tf.uint8.as_datatype_enum,
    )

    # save graph to file
    with tf.gfile.GFile(output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Saving graph done.")
    print("{} operations in the graph.".format(len(output_graph_def.node)))


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print("%d %s %s" % (i, node.name, node.op))
        [print(u"└─── %d ─ %s" % (i, n)) for i, n in enumerate(node.input)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="test_data", type=str)
    parser.add_argument("--image_height", default="512", type=int)
    parser.add_argument("--image_width", default="680", type=int)
    parser.add_argument("--ckpt_dir", default="model_logs/places2", type=str)
    parser.add_argument("--output_ckpt_dir", default="model_logs/test", type=str)
    parser.add_argument("--tensorboard_dir", default="tensorboard", type=str)
    args = parser.parse_args()

    image, mask, out = get_dirs(args.data_dir, 0)

    sess = test_single_image(
        image_path=image,
        mask_path=mask,
        output_path=out,
        image_height=args.image_height,
        image_width=args.image_width,
        ckpt_dir=args.ckpt_dir,
    )

    save_frozen_graph(sess)
    save_checkpoint(sess, args.output_ckpt_dir + "/model.ckpt")
    save_tensorboard_log(sess, args.tensorboard_dir)
    # save_graph(sess, args.output_ckpt_dir)

    sess.close()
