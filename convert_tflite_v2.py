import os, argparse
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1


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


def get_input_images():
    input_folder = "test_data/input"
    mask_folder = "test_data/mask"

    dir_files = os.listdir(input_folder)
    dir_files.sort()

    input_images = []

    for file_inter in dir_files:
        base = os.path.basename(file_inter)

        image = input_folder + "/" + base
        mask = mask_folder + "/" + base

        input_images.append(get_input_image(image, mask, 512, 680))

    return input_images


def representative_data_gen():
    for input_value in get_input_images():
        # Model has only one input so each data point has one element.
        yield [input_value]


def get_output_file(output_dir):
    # check output dir is exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output = output_dir + "/model.tflite"

    return output


def convert_to_ftlite(graph_path, output_file, input_shapes, input_nodes, output_nodes):
    # init TFLite converter
    converter = tf1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=graph_path,
        input_shapes=input_shapes,
        input_arrays=input_nodes.split(","),
        output_arrays=output_nodes.split(","),
    )

    # converter.allow_custom_ops = True
    # converter.experimental_new_converter = True
    # converter.post_training_quantize = True
    converter.target_spec.supported_ops = [
        # tf.lite.OpsSet.TFLITE_BUILTINS_INT8,  # enable TensorFlow Lite ops
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops
    ]
    converter.target_spec.support_types = [
        tf.float32,
        # tf.uint8
    ]
    # converter.optimizations = [
    # tf.lite.Optimize.OPTIMIZE_FOR_SIZE
    # ]  # DEFAULT, OPTIMIZE_FOR_SIZE, OPTIMIZE_FOR_LATENCY
    # converter.representative_dataset = representative_data_gen
    # converter.inference_type = tf.uint8
    # converter.inference_input_type = tf.uint8  # or tf.uint8
    # converter.inference_output_type = tf.uint8  # or tf.uint8

    # input_arrays = converter.get_input_arrays()
    # converter.quantized_input_stats = {input_arrays[0]: (0.0, 1.0)}

    # convert model to TFLite model
    print("Converting model start ...")
    tflite_model = converter.convert()
    print("Converting model done.")

    # save model to file
    print("Saving model to {}".format(output_file))
    open(output_file, "wb").write(tflite_model)
    print("Saving model done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_graph", default="tflite/frozen_model.pb", type=str)
    parser.add_argument("--output_dir", default="tflite", type=str)
    args = parser.parse_args()

    output_file = get_output_file(args.output_dir)

    input_shapes = {"input": [1, 512, 1360, 3]}
    input_node_names = "input"
    output_node_names = "saturate_cast"
    # output_node_names = "ReverseV2"

    convert_to_ftlite(
        graph_path=args.input_graph,
        output_file=output_file,
        input_shapes=input_shapes,
        input_nodes=input_node_names,
        output_nodes=output_node_names,
    )