# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io
import tensorflow as tf
import tf_slim as slim
import numpy as np
import rawpy
import glob
from PIL import Image
import cv2
import math

input_dir = './Sony/short/'
gt_dir = './Sony/long/'
checkpoint_dir = './checkpoint_danet/Sony/'
result_dir = './result_danet_test/'

# get test IDs
test_fns = glob.glob(gt_dir + '/0*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)



def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.compat.v1.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def DANet_block(inputs, out_dim, kernel_size=3, stride=1, dilation=1, padding='SAME', scope='DANet_block'):
    with tf.compat.v1.variable_scope(scope):
        # Convolution branch
        conv = slim.conv2d(inputs, out_dim, [kernel_size, kernel_size], stride=stride, rate=dilation, padding=padding, scope='conv')

        # Channel Attention
        avg_pool = tf.reduce_mean(conv, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(conv, axis=[1, 2], keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        channel_wise_attention = slim.conv2d(concat, 1, [1, 1], activation_fn=tf.nn.sigmoid, scope='channel_attention')
        scaled_conv = tf.multiply(conv, channel_wise_attention)

        # Spatial Attention
        squeeze = slim.conv2d(scaled_conv, 1, [1, 1], activation_fn=tf.nn.sigmoid, scope='squeeze')
        expand = slim.conv2d(squeeze, out_dim, [1, 1], activation_fn=None, scope='expand')
        spatial_wise_attention = tf.multiply(scaled_conv, expand)

        # Output
        output = tf.add(conv, spatial_wise_attention, name='output')
        return output


def network(input):
    block1 = DANet_block(input, 32, scope='g_block1')
    pool1 = slim.max_pool2d(block1, [2, 2], padding='SAME')

    block2 = DANet_block(pool1, 64, scope='g_block2')
    pool2 = slim.max_pool2d(block2, [2, 2], padding='SAME')

    block3 = DANet_block(pool2, 128, scope='g_block3')
    pool3 = slim.max_pool2d(block3, [2, 2], padding='SAME')

    block4 = DANet_block(pool3, 256, scope='g_block4')
    pool4 = slim.max_pool2d(block4, [2, 2], padding='SAME')

    block5 = DANet_block(pool4, 512, scope='g_block5')

    up6 = upsample_and_concat(block5, block4, 256, 512)
    block6 = DANet_block(up6, 256, scope='g_block6')

    up7 = upsample_and_concat(block6, block3, 128, 256)
    block7 = DANet_block(up7, 128, scope='g_block7')

    up8 = upsample_and_concat(block7, block2, 64, 128)
    block8 = DANet_block(up8, 64, scope='g_block8')

    up9 = upsample_and_concat(block8, block1, 32, 64)
    block9 = DANet_block(up9, 32, scope='g_block9')

    block10 = slim.conv2d(block9, 12, [1, 1], rate=1, activation_fn=None, scope='g_block10')
    out = tf.compat.v1.depth_to_space(block10, 2)

    return out


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


sess = tf.compat.v1.Session()
tf.compat.v1.disable_eager_execution()
in_image = tf.compat.v1.placeholder(tf.float32, [1, 512, 512, 4])
gt_image = tf.compat.v1.placeholder(tf.float32, [1, 1024,1024, 3])
out_image = network(in_image)

saver = tf.compat.v1.train.Saver()
sess.run(tf.compat.v1.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(f"{input_dir}{test_id:05d}_00*.ARW")
    print(in_files)
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_files = glob.glob(f"{gt_dir}{test_id:05d}_00*.ARW")
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
        gt_full = cv2.resize(gt_full[0], dsize=(1024,1024), interpolation=cv2.INTER_CUBIC)
        gt_full= np.expand_dims(gt_full,0)
        #gt_full= np.resize(gt_full,(1,1024,1024,3))
        input_full = np.minimum(input_full, 1.0)
        #input_full = np.resize(input_full,(1,512,512,4))
        input_full = cv2.resize(input_full[0], dsize=(512,512), interpolation=cv2.INTER_CUBIC)
        input_full= np.expand_dims(input_full,0)
        #input_full= tf.image.resize(input_full,size=[512,512])


        output = sess.run(out_image, feed_dict={in_image: input_full})
        output = np.minimum(np.maximum(output, 0), 1)

        output = output[0, :, :, :]
        gt_full = gt_full[0, :, :, :]
        scale_full = scale_full[0, :, :, :]
        scale_full = scale_full * np.mean(gt_full) / np.mean(
            scale_full)  # scale the low-light image to the same mean of the groundtruth


        Image.fromarray((output*255).astype(np.uint8)).save(
            f"{result_dir}final/{test_id:05d}_00_{ratio}_out.png")
        Image.fromarray((scale_full*255).astype(np.uint8)).save(
            f"{result_dir}final/{test_id:05d}_00_{ratio}_scale.png")
        Image.fromarray((gt_full * 255).astype(np.uint8)).save(
            f"{result_dir}final/{test_id:05d}_00_{ratio}_gt.png")
