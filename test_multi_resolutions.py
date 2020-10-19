import argparse

import os
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel

def deepfill_model(checkpoint_dir):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, None, None, 3))
    output = model.build_server_graph(input_image_ph, dynamic=True)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    return input_image_ph, output, sess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagedir', default='', type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--outdir', default='', type=str,
                        help='Where to write output.')
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='The directory of tensorflow checkpoint.')

    args = parser.parse_args()

    # check_folder(args.outdir)
    output_image_dir = os.path.join(args.outdir, "images")
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    # ng.get_gpus(1)
    counts = 0
    input_image_ph, output, sess = deepfill_model(args.checkpoint_dir)
    for img in os.listdir(args.imagedir):
        print("img: ", img)

        fre_img, ex = os.path.splitext(img)
        img_outname_deepfill = fre_img + "_output_deepfill.jpg"

        img_path = os.path.join(args.imagedir, img)
        img_in_path = os.path.join(output_image_dir, img)
        img_out_deepfill_path = os.path.join(output_image_dir, img_outname_deepfill)
        print("img_out_deepfill_path: ", img_out_deepfill_path)
        image = cv2.imread(img_path)
        _, mask = getMask(image)

        counts += 1
        assert image.shape == mask.shape
        h, w, _ = image.shape
        grid = 8
        print('before shape of image: {}'.format(image.shape))
        image_deep = image[:h // grid * grid, :w // grid * grid, :]
        mask_deep = mask[:h // grid * grid, :w // grid * grid, :]
        print('Shape of image: {}'.format(image_deep.shape))

        image_deep = np.expand_dims(image_deep, 0)
        mask_deep = np.expand_dims(mask_deep, 0)
        input_image = np.concatenate([image_deep, mask_deep], axis=2)
        print("input_image shape: ", input_image.shape)
        # load pretrained model
        result = sess.run(output, feed_dict={input_image_ph: input_image})
        # logger.info('Processed: {}'.format(img_out_deepfill_path))
        cv2.imwrite(img_out_deepfill_path, result[0][:, :, ::-1])
