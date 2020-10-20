import argparse

import os
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

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


def get_input(image_path, mask_path, image_width, image_height):
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

    return input_image, h, w


def test_single_image(input_image, output_path, image_height, image_width, ckpt_dir):

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

    # run model
    print("Running model ...")
    result = sess.run(output, feed_dict={input_image_ph: input_image})
    print("Running model done.")

    # save output image to file
    print("Saving output to {}".format(output_path))
    cv2.imwrite(output_path, result[0][:, :, ::-1])
    print("Saving output done.")

    # save checkpoint
    saver = tf.train.Saver().save(sess, "./model_logs/test/model.ckpt")

    # save graph for Tensorboard
    tf.summary.FileWriter("./tbgraph", sess.graph)

    # save graph to file
    # tf.train.write_graph(sess.graph_def, './graph/', 'graph.pbtxt')
    # tf.train.write_graph(sess.graph_def, './graph/', 'graph.pb')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--image_height", default="", type=int)
    parser.add_argument("--image_width", default="", type=int)
    parser.add_argument("--ckpt_dir", default="", type=str)

    args = parser.parse_args()

    image, mask, out = get_dirs(args.data_dir, 0)
    input_image, h, w = get_input(image, mask, args.image_width, args.image_height)

    test_single_image(input_image, out, h, w, args.ckpt_dir)