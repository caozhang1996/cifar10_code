#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:17:39 2018

@author: caozhang
"""

import tensorflow as tf 
import os
import numpy as np
import matplotlib.pyplot as plt


def read_cifar10(data_dir, is_train, batch_size, is_shuffle):
    """
    read cifar10 binary data
    
    Args:
        data_dir: the directory of cifar10
        is_train: 采用训练数据还是测试数据
    Return:
        images: 4D tensor of [batch_size, image_height, image_width, 3] size.
        labels: 1D tensor of [batch_size] size.
    """
    image_height = 32
    image_width = 32
    image_depth = 3
    label_bytes = 1
    image_bytes = image_height*image_width*image_depth
    
    with tf.name_scope("input"):
        if is_train:
           filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in np.arange(1, 6)]
        else:
           filenames = [os.path.join(data_dir, "test_batch.bin")]
           
        filenames_queue = tf.train.string_input_producer(filenames)
        reader = tf.FixedLengthRecordReader(record_bytes=label_bytes+image_bytes)
        key, value = reader.read(filenames_queue)
        
        # 定义Decoder(解码器),将二进制变为十进制
        record_bytes = tf.decode_raw(value, tf.uint8)
        
        label = tf.slice(record_bytes, [0], [label_bytes])
        label = tf.cast(label, tf.int32)
        
        image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
        image_raw = tf.reshape(image_raw, [image_depth, image_height, image_width])
        image = tf.transpose(image_raw, [1, 2, 0])  # Convert from [depth, height, width] to [height, width, depth]
        image = tf.cast(image, tf.float32)  # 将image变为tf.float32型是因为后一步进行标准化操作

         # data augmentation
#         distorted_image = tf.random_crop(image, [24, 24, 3])  # 随机剪裁图像的高度， 宽度部分
#         distorted_image = tf.image.random_flip_left_right(distorted_image)    # 随机地水平翻转图像
#         distorted_image = tf.image.random_brightness(distorted_image, max_delta=65)  # 随机改变亮度
#         distorted_image = tf.image.random_contrast(distorted_image, lower=0.1, upper=2.0)
        
        image = tf.image.per_image_standardization(image)  # 对图像进行标准化
        if is_shuffle:
            images, label_batch = tf.train.shuffle_batch([image, label],
                                                         batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=2000,
                                                         min_after_dequeue=1500)
        else:
            images, label_batch = tf.train.batch([image, label],
                                                 batch_size=batch_size,
                                                 num_threads=16,
                                                 capacity=2000)
        # one hot coding
        # tf.one_hot()函数的作用是将一个值化为一个概率分布的向量，一般用于分类问题
#        n_classes = 10
#        label_batch = tf.one_hot(label_batch, depth= n_classes)
#        return images, tf.reshape(label_batch, [batch_size, n_classes])
    
        return images, tf.reshape(label_batch, [batch_size])
    

if __name__ == "__main__":
    data_dir = "/home/caozhang/spyder_projects/cifar10_code/cifar-10-batches-bin/"
    BATCH_SIZE = 10
    image_batch, label_batch = read_cifar10(data_dir, is_train=True, 
                                            batch_size=BATCH_SIZE, is_shuffle=True)
    
    with tf.Session() as sess:
        i = 0
        # 添加线程管理器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        try:
            while not coord.should_stop() and i < 1:   # 其中有一个不 满足时，停止从队列中取数
                img, label = sess.run([image_batch, label_batch])
                for j in np.arange(BATCH_SIZE):
                    print("label: %d" % label[j])
                    plt.imshow(img[j, :, :, :])   # 4D,所以后三维用冒号
                    plt.show()
                i += 1
        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
        coord.join(threads)
