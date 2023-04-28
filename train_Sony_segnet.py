# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tf_slim as slim
import numpy as np
import rawpy
import glob
import math
from PIL import Image
import logging
import numpy as np
import time

# Configure logging
logging.basicConfig(filename='SegnetEpoch.log',
                    level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Set up a logger
logger = logging.getLogger('debug')




input_dir = './images_test/short/'
gt_dir = './images_test/long/'
checkpoint_dir = './checkpoint_segnet/Sony/'
result_dir = './result_segnet/'

# get train IDs
train_fns = glob.glob(gt_dir + '1*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

ps = 512  # patch size for training
save_freq = 50

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]


def network(input):
    # Encoding layers
    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
        conv1 = slim.conv2d(input, 64, [3, 3], rate=1, scope='g_conv1_1')  
        conv1 = slim.conv2d(conv1, 64, [3, 3], rate=1, scope='g_conv1_2')
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')        

        
              
        conv2 = slim.conv2d(pool1, 128, [3, 3], rate=1, scope='g_conv2_1')
        conv2 = slim.conv2d(conv2, 128, [3, 3], rate=1, scope='g_conv2_2')
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')


        
        conv3 = slim.conv2d(pool2, 256, [3, 3], rate=1, scope='g_conv3_1')
        conv3 = slim.conv2d(conv3, 256, [3, 3], rate=1, scope='g_conv3_2')
        conv3 = slim.conv2d(conv3, 256, [3, 3], rate=1, scope='g_conv3_3')
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        
        conv4 = slim.conv2d(pool3, 512, [3, 3], rate=1, scope='g_conv4_1')
        conv4 = slim.conv2d(conv4, 512, [3, 3], rate=1, scope='g_conv4_2')
        conv4 = slim.conv2d(conv4, 512, [3, 3], rate=1, scope='g_conv4_3')
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, scope='g_conv5_1')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, scope='g_conv5_2')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, scope='g_conv5_3')
        pool5 = slim.max_pool2d(conv5, [2, 2], padding='SAME')
    # Decoding layers
    with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):    
        up6 = tf.image.resize(pool5, conv5.get_shape()[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv6 = slim.conv2d(up6, 512, [3, 3], rate=1, scope='g_conv6_1')
        conv6 = slim.conv2d(conv6, 512, [3, 3], rate=1, scope='g_conv6_2')
        conv6 = slim.conv2d(conv6, 512, [3, 3], rate=1, scope='g_conv6_3')


        #up7 = upsample(pool4, conv6,conv6.shape[-1],pool4.shape[-1])
        up7 = tf.image.resize(conv6, conv4.get_shape()[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        conv7 = slim.conv2d(up7, 512, [3, 3], rate=1, scope='g_conv7_1')
        conv7 = slim.conv2d(conv7, 512, [3, 3], rate=1, scope='g_conv7_2')
        conv7 = slim.conv2d(conv7, 256, [3, 3], rate=1, scope='g_conv7_3')


        
        up8 = tf.image.resize(conv7,conv3.get_shape()[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        conv8 = slim.conv2d(up8, 256, [3, 3], rate=1, scope='g_conv8_1')
        conv8 = slim.conv2d(conv8, 256, [3, 3], rate=1, scope='g_conv8_2')
        conv8 = slim.conv2d(conv8, 128, [3, 3], rate=1, scope='g_conv8_3')

        up9 = tf.image.resize(conv8, conv2.get_shape()[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


        conv9 = slim.conv2d(up9, 128, [3, 3], rate=1, scope='g_conv9_1')
        conv9 = slim.conv2d(conv9, 64, [3, 3], rate=1, scope='g_conv9_2')


        up10 = tf.image.resize(conv9,conv1.get_shape()[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


        conv10 = slim.conv2d(up10, 64, [3, 3], rate=1, scope='g_conv10_1')
        conv10 = slim.conv2d(conv10, 64, [3, 3], rate=1, scope='g_conv10_2')

        # Output
       # 
        output = slim.conv2d(conv10, 64, [1, 1], rate=1, activation_fn=tf.nn.softmax, scope='g_conv10_3')

        output = slim.conv2d(output, 3, [3, 3], rate=1, activation_fn=None, scope='g_conv11')
        output = tf.image.resize(output, size=[1024, 1024], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

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
in_image = tf.compat.v1.placeholder(tf.float32, [1, 512, 512, 4])
gt_image = tf.compat.v1.placeholder(tf.float32, [1, 1024,1024, 3])

out_image = network(in_image)

G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))

t_vars = tf.compat.v1.trainable_variables()
lr = tf.compat.v1.placeholder(tf.float32)
G_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver = tf.compat.v1.train.Saver()
sess.run(tf.compat.v1.global_variables_initializer())
ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# Raw data takes long time to load. Keep them in memory after loaded.
gt_images = [None] * 6000
input_images = {}
input_images['300'] = [None] * len(train_ids)
input_images['250'] = [None] * len(train_ids)
input_images['100'] = [None] * len(train_ids)

g_loss = np.zeros((5000, 1))

allfolders = glob.glob(result_dir + '*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
for epoch in range(lastepoch, 1001):
    if os.path.isdir(str(result_dir) + f'{epoch:04d}'):
        continue
    cnt = 0
    if epoch > 500:
        learning_rate = 1e-5
    
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(f"{input_dir}*_00*.ARW")
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(f"{gt_dir}{train_id:05d}_00*.ARW")
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]
        
        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        

        _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                        feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
        output = np.minimum(np.maximum(output, 0), 1)
        g_loss[ind] = G_current

        logger.debug(f"{epoch} {cnt} Loss={np.mean(g_loss[np.where(g_loss)]):.3f} Time={time.time() - st:.3f}")

        if epoch % save_freq == 0:
            if not os.path.isdir(f"{result_dir}{epoch:04d}"):
                os.makedirs(f"{result_dir}{epoch:04d}")

            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            if not os.path.isdir(f"{result_dir}{epoch:04d}"):
                os.makedirs(f"{result_dir}{epoch:04d}")
            
            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            temp = (temp * 255).clip(0, 255).astype('uint8')
            img = Image.fromarray(temp)
            img.save(f"{result_dir}{epoch:04d}/{train_id:05d}_00_train_{ratio:.2f}.jpg")
            saver.save(sess, checkpoint_dir + 'model.ckpt')
            


    