#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:44:17 2018

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cifar10

BATCH_SIZE = 1

def evaluate_images():
    """
    Test one image against the saved models and parameters
    """
    data_dir = '/home/caozhang/spyder_projects/cifar10_code/my_test_data/'
    log_dir = '/home/caozhang/spyder_projects/cifar10_code/logs_no_one_hot/'
    images_list = []
    for file in os.listdir(data_dir):
        images_list.append(data_dir + file)
        
    with tf.Graph().as_default() as g:
        for i in np.arange(0, len(images_list)):
            image_dir = images_list[i]
            image = Image.open(image_dir)
            plt.imshow(image)
            plt.show()
            
            image_arrary = np.array(image)
            
            image = tf.cast(image_arrary, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image, shape=[1, 32, 32, 3])
            
            logit = cifar10.inference(image, batch_size=BATCH_SIZE)
            logit = tf.nn.softmax(logit)
            saver = tf.train.Saver()
            
            with tf.Session() as sess:
                 # 测试多张图片，我们模型的参数需要重复使用，所以我们需要告诉TF允许复用参数，加上下行代码
                 tf.get_variable_scope().reuse_variables()
                 print ('Reading checkpoints...')
                 ckpt = tf.train.get_checkpoint_state(log_dir)
                 
                 if ckpt and ckpt.model_checkpoint_path:
                     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                     print ('Loading success! global step is %s' % global_step)  #global_step 是字符型
                     saver.restore(sess, ckpt.model_checkpoint_path)
                     
                 else:
                     print ('Not find checkpoint file!')
                     return 
                 
                 prediction = sess.run(logit)
                 max_index = np.argmax(prediction)
                 
                 if max_index == 0:
                     print ('This is a airplane with possibility %.6f' % prediction[:, 0])
                     
                 if max_index == 1:
                     print ('This is a automobile with possibility %.6f' % prediction[:, 1])
                 
                 if max_index == 2:
                     print ('This is a bird with possibility %.6f' % prediction[:, 2])
                     
                 if max_index == 3:
                     print ('This is a cat with possibility %.6f' % prediction[:, 3])
                    
                 if max_index == 4:
                     print ('This is a deer with possibility %.6f' % prediction[:, 4])
                    
                 if max_index == 5:
                     print ('This is a dog with possibility %.6f' % prediction[:, 5])
                     
                 if max_index == 6:
                     print ('This is a frog with possibility %.6f' % prediction[:, 6])
                     
                 if max_index == 7:
                     print ('This is a horse with possibility %.6f' % prediction[:, 7])
                    
                 if max_index == 8:
                     print ('This is a ship with possibility %.6f' % prediction[:, 8])
                    
                 if max_index == 9:
                         print ('This is a truck with possibility %.6f' % prediction[:, 9])
                     
if __name__ == '__main__':
    evaluate_images()