import argparse

import os
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


def test_single_image(image, mask, out,
                      image_height, image_width,
                      checkpoint_dir):

    # start sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()

    # input node placeholder
    input_image_ph = tf.placeholder(                                                                                                                                                                                                          
        tf.float32, name='input', shape=(1, image_height, image_width*2, 3)) 

    output = model.build_server_graph(input_image_ph)                                                                                                                                                                                         
    output = (output + 1.) * 127.5                                                                                                                                                                                                            
    output = tf.reverse(output, [-1])                                                                                                                                                                                                         
    output = tf.saturate_cast(output, tf.uint8)
    
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)                                                                                                                                                                              
    assign_ops = []
    for var in vars_list:                                                                                                                                                                                                                     
        vname = var.name                                                                                                                                                                                                                      
        from_name = vname                                                                                                                                                                                                                     
        var_value = tf.contrib.framework.load_variable(                                                                                                                                                                                       
            args.checkpoint_dir, from_name)                                                                                                                                                                                                   
        assign_ops.append(tf.assign(var, var_value))
    
    sess.run(assign_ops)     
    
    # load image
    image = cv2.imread(image)
    mask = cv2.imread(mask)

    # resize image
    image = cv2.resize(image, (args.image_width, args.image_height))                                                                                                                                                                      
    mask = cv2.resize(mask, (args.image_width, args.image_height)) 
    
    assert image.shape == mask.shape

    h, w, _ = image.shape                                                                                                                                                                                                                 
    grid = 8                                                                                                                                                                                                                              
    image = image[:h//grid*grid, :w//grid*grid, :]                                                                                                                                                                                        
    mask = mask[:h//grid*grid, :w//grid*grid, :]                                                                                                                                                                                          
    print('image shape: {}'.format(image.shape))

    # [512, 680, 3] to [1, 512, 680, 3]
    image = np.expand_dims(image, 0)                                                                                                                                                                                                      
    mask = np.expand_dims(mask, 0)

    # [image]+[mask] to [1, 512, 1360, 3]
    input_image = np.concatenate([image, mask], axis=2)
    print('input_image [image + mask] shape: {}'.format(input_image.shape))

    # run pretrained model                                                                                                                                                                                                               
    result = sess.run(output, feed_dict={input_image_ph: input_image})                                                                                                                                                                    
    print('Processed: {}'.format(out))                                                                                                                                                             
    cv2.imwrite(out, result[0][:, :, ::-1])

    print('Processing {} done.'.format(out))

    # 체크포인트 저장
    saver = tf.train.Saver().save(sess, './model_logs/test/model.ckpt')

    # 그래프 저장
    # tf.train.write_graph(sess.graph_def, './graph/', 'graph.pbtxt')
    # tf.train.write_graph(sess.graph_def, './graph/', 'graph.pb')
    
    # tensorboard 출력을 위한 그래프 저장
    # tf.summary.FileWriter('./tbgraph', sess.graph)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default='', type=str,
                        help='The directory of test images to be completed.')
    parser.add_argument('--image_height', default='', type=int,
                        help='The height of test images')
    parser.add_argument('--image_width', default='', type=int,
                        help='The width of test images')
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='The directory of tensorflow checkpoint.')
    
    args = parser.parse_args()

    # prepare folder
    input_folder = args.test_dir + "/input"
    mask_folder = args.test_dir + "/mask"
    output_folder = args.test_dir + "/output"

    # check output dir 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    dir_files = os.listdir(input_folder)
    dir_files.sort()
    
    file_inter = dir_files[0]
    
    base = os.path.basename(file_inter)

    image = input_folder + "/" + base
    mask = mask_folder + "/" + base
    out = output_folder + "/" + base

    test_single_image(image, mask, out,
                      args.image_height,
                      args.image_width,
                      args.checkpoint_dir)

    
        
