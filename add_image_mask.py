import argparse

import os
import cv2
import numpy as np

def add_mask_to_image(image, mask, image_width, image_height, output_dir):
    
    # check output dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base = os.path.basename(image)

    output = '{0}/input_{1}'.format(output_dir, base)
    
    # load image
    image = cv2.imread(image)
    mask = cv2.imread(mask)

    # resize image
    image = cv2.resize(image, (image_width, image_height))                                                                                                                                                                      
    mask = cv2.resize(mask, (image_width, image_height)) 
    
    assert image.shape == mask.shape
    # shape image
    h, w, _ = image.shape                                                                                                                                                                                                                 
    grid = 4                                                                                                                                                                                                                              
    image = image[:h//grid*grid, :w//grid*grid, :]                                                                                                                                                                                        
    mask = mask[:h//grid*grid, :w//grid*grid, :]      

    # prepare input image                                                                                                                                                                                                     
    output_image = np.concatenate([image, mask], axis=1)

    # write image
    cv2.imwrite(output, output_image)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='', type=str)
    parser.add_argument('--mask', default='', type=str)
    parser.add_argument('--image_width', default='', type=int)
    parser.add_argument('--image_height', default='', type=int)
    parser.add_argument('--output_dir', default='', type=str)
    args = parser.parse_args()

    add_mask_to_image(args.image,
                      args.mask,
                      args.image_width,
                      args.image_height,
                      args.output_dir)
