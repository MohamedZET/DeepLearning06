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

input_dir = './image_map/short/'
gt_dir = './image_map/long/'
checkpoint_dir = 'checkpoint_segnet'
result_dir = './result_segnet/'

# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
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


def network(input):
    # Encoding layers
    global pool1
    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
        conv1 = slim.conv2d(input, 64, [3, 3], rate=1, scope='g_conv1_1')
        conv1 = slim.conv2d(conv1, 64, [3, 3], rate=1, scope='g_conv1_2')
        pool1 = slim.max_pool2d(conv1, [2, 2], stride=2, padding='SAME')

        conv2 = slim.conv2d(pool1, 128, [3, 3], rate=1, scope='g_conv2_1')
        conv2 = slim.conv2d(conv2, 128, [3, 3], rate=1, scope='g_conv2_2')
        pool2 = slim.max_pool2d(conv2, [2, 2], stride=2, padding='SAME')

        conv3 = slim.conv2d(pool2, 256, [3, 3], rate=1, scope='g_conv3_1')
        conv3 = slim.conv2d(conv3, 256, [3, 3], rate=1, scope='g_conv3_2')
        conv3 = slim.conv2d(conv3, 256, [3, 3], rate=1, scope='g_conv3_3')
        pool3 = slim.max_pool2d(conv3, [2, 2], stride=2, padding='SAME')

        conv4 = slim.conv2d(pool3, 512, [3, 3], rate=1, scope='g_conv4_1')
        conv4 = slim.conv2d(conv4, 512, [3, 3], rate=1, scope='g_conv4_2')
        conv4 = slim.conv2d(conv4, 512, [3, 3], rate=1, scope='g_conv4_3')
        pool4 = slim.max_pool2d(conv4, [2, 2], stride=2, padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, scope='g_conv5_1')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, scope='g_conv5_2')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, scope='g_conv5_3')
        pool5 = slim.max_pool2d(conv5, [2, 2], stride=2, padding='SAME')
        
    # Decoding layers
    with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
        up6 = slim.conv2d_transpose(pool5, 512, [3, 3], stride=2, scope='g_up6')
        up6 = tf.concat([up6, conv5], 3)
    
        conv6 = slim.conv2d(up6, 512, [3, 3], rate=1, scope='g_conv6_1')
        conv6 = slim.conv2d(conv6, 512, [3, 3], rate=1, scope='g_conv6_2')
        conv6 = slim.conv2d(conv6, 512, [3, 3], rate=1, scope='g_conv6_3')
    
        up7 = slim.conv2d_transpose(conv6, 512, [3, 3], stride=2, scope='g_up7')
        up7 = tf.concat([up7, conv4], 3)
    
        conv7 = slim.conv2d(up7, 512, [3, 3], rate=1, scope='g_conv7_1')
        conv7 = slim.conv2d(conv7, 512, [3, 3], rate=1, scope='g_conv7_2')
        conv7 = slim.conv2d(conv7, 256, [3, 3], rate=1, scope='g_conv7_3')
    
        up8 = slim.conv2d_transpose(conv7, 256, [3, 3], stride=2, scope='g_up8')
        up8 = tf.concat([up8, conv3], 3)
    
        conv8 = slim.conv2d(up8, 256, [3, 3], rate=1, scope='g_conv8_1')
        conv8 = slim.conv2d(conv8, 256, [3, 3], rate=1, scope='g_conv8_2')
        conv8 = slim.conv2d(conv8, 128, [3, 3], rate=1, scope='g_conv8_3')
    
        up9 = slim.conv2d_transpose(conv8, 128, [3, 3], stride=2, scope='g_up9')
        up9 = tf.concat([up9, conv2], 3)
    
        conv9 = slim.conv2d(up9, 128, [3, 3], rate=1, scope='g_conv9_1')
        conv9 = slim.conv2d(conv9, 64, [3, 3], rate=1, scope='g_conv9_2')
    
        up10 = slim.conv2d_transpose(conv9, 64, [3, 3], stride=2, scope='g_up10')
        up10 = tf.concat([up10, conv1], 3)
    
        conv10 = slim.conv2d(up10, 64, [3, 3], rate=1, scope='g_conv10_1')
        conv10 = slim.conv2d(conv10, 64, [3, 3], rate=1, scope='g_conv10_2')
    
        # Output
        output = slim.conv2d(conv10, 3, [1, 1], rate=1, activation_fn=tf.nn.softmax, scope='g_conv10_3')

    return output



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
in_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3])
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
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
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

        input_full = np.minimum(input_full, 1.0)

        output = sess.run(out_image, feed_dict={in_image: input_full})
        output = np.minimum(np.maximum(output, 0), 1)
        print(output.shape)

        output = output[0, :, :, :]
        gt_full = gt_full[0, :, :, :]
        scale_full = scale_full[0, :, :, :]
        scale_full = scale_full * np.mean(gt_full) / np.mean(
            scale_full)  # scale the low-light image to the same mean of the groundtruth

        Image.fromarray((output*255).astype(np.uint8)).save(
            result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio))
        Image.fromarray((scale_full*255).astype(np.uint8)).save(
            result_dir + 'final/%5d_00_%d_scale.png' % (test_id, ratio))
        Image.fromarray((gt_full * 255).astype(np.uint8)).save(
            result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))
