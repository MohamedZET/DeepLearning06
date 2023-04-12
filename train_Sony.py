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

input_dir = './image_map/short/'
gt_dir = './image_map/long/'
checkpoint_dir = './checkpoint_segnet/Sony/'
result_dir = './result_segnet/'

# get train IDs
train_fns = glob.glob(gt_dir + '1*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

ps = 512  # patch size for training
save_freq = 500

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.compat.v1.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    # deconv_output = tf.concat([deconv, x2], 3)
    deconv.set_shape([None, None, None, output_channels])

    return deconv

def upsample(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.compat.v1.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
    return deconv

# def up_sampling(pool, ind, output_shape, batch_size, name=None):
#     """
#         Unpooling layer after max_pool_with_argmax.
#         Args:
#             pool:   max pooled output tensor
#             ind:      argmax indices
#             ksize:     ksize is the same as for the pool
#         Return:
#             unpool:    unpooling tensor
#             :param batch_size:
#     """

#     pool_ = tf.reshape(pool, [-1])
    
#     batch_range = tf.reshape(tf.range(batch_size, dtype=ind.dtype), [tf.shape(pool)[0], 1, 1, 1])
#     b = tf.ones_like(ind) * batch_range
#     b = tf.reshape(b, [-1, 1])
#     ind_ = tf.reshape(ind, [-1, 1])
#     ind_ = tf.concat([b, ind_], 1)
    
#     ret = tf.scatter_nd(ind_, pool_, shape=[batch_size, output_shape[1] * output_shape[2] * output_shape[3]])
#     # the reason that we use tf.scatter_nd: if we use tf.sparse_tensor_to_dense, then the gradient is None, which will cut off the network.
#     # But if we use tf.scatter_nd, the gradients for all the trainable variables will be tensors, instead of None.
#     # The usage for tf.scatter_nd is that: create a new tensor by applying sparse UPDATES(which is the pooling value) to individual values of slices within a
#     # zero tensor of given shape (FLAT_OUTPUT_SHAPE) according to the indices (ind_). If we ues the orignal code, the only thing we need to change is: changeing
#     # from tf.sparse_tensor_to_dense(sparse_tensor) to tf.sparse_add(tf.zeros((output_sahpe)),sparse_tensor) which will give us the gradients!!!
#     ret = tf.reshape(ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])
#     return ret

# def up_sampling(pool, ind, output_shape, __, name = None):
#     """Upsamples the input tensor using max pooling indices.
    
#     Args:
#         pool: The input tensor to upsample.
#         ind: The max pooling indices returned by `tf.nn.max_pool_with_argmax`.
#         output_shape: The desired output shape.
        
#     Returns:
#         The upsampled tensor.
#     """
#     ind = tf.cast(ind,tf.int32)
#     input_shape = tf.shape(pool)
#     batch_size = input_shape[0]
#     height = input_shape[1]
#     width = input_shape[2]
#     channels = input_shape[3]

#     # Flatten the pooled indices
#     ind = tf.reshape(ind, [batch_size, -1, channels])

#     # Calculate the linear indices of the pooled values
#     pool_size = height * width
#     linear_indices = ind + tf.expand_dims(tf.range(batch_size) * pool_size * channels, axis=-1)

#     # Gather the pooled values using the linear indices
#     pool_ = tf.reshape(pool, [batch_size, -1, channels])
#     pooled = tf.gather(tf.reshape(pool_, [-1]), linear_indices)

#     # Reshape the pooled tensor
#     upsampled = tf.reshape(pooled, [batch_size, output_shape[1], output_shape[2], channels])

#     return upsampled

# def up_sampling(input_tensor, pool_indices, output_shape, name=None):
  
#     # Unpooling using the max pooling indices
#     print("yes")

    
#     ind = tf.reshape(pool_indices, [input_tensor[0], -1, input_tensor[-1]])
#     unpool = tf.scatter_nd(
#         ind, input_tensor, shape = [output_shape[0], output_shape[1]*output_shape[2]*output_shape[3]]
#     )
#     print("No")
#     # 2D transposed convolution layer for upsampling
#     upsample = tf.layers.conv2d_transpose(
#         unpool,
#         filters=output_shape[-1],
#         kernel_size=(3, 3),
#         strides=(2, 2),
#         padding="same",
#         activation=tf.nn.relu,
#         name="conv_transpose",
#     )
#     return upsample

def up_sampling(input, indices, output_shape, batchsoze,scale=2):
    """
    Performs upsampling of the input tensor based on the max pooling indices and output shape.

    Arguments:
    input -- tensor of shape (batch_size, height, width, channels)
    indices -- tensor of shape (batch_size, new_height, new_width, channels)
    output_shape -- a tuple specifying the output shape of the upsampled tensor
    scale -- an integer specifying the upsampling scale factor

    Returns:
    upsampled -- tensor of shape output_shape
    """

    # Calculate the batch size
    batch_size = tf.shape(input)[0]

    # Calculate the new height and width
    new_height = tf.shape(indices)[1] * scale
    new_width = tf.shape(indices)[2] * scale
    
    print(new_height,new_width)

    # Create the indices tensor
    indices = tf.reshape(indices, [batch_size, -1, 1])
    indices = tf.tile(indices, [1, 1, scale*scale])
    indices = tf.reshape(indices, [batch_size, -1])

    # Create the updates tensor
    updates = tf.reshape(input, [batch_size, -1])
    updates = tf.tile(updates, [1, scale*scale])
    updates = tf.reshape(updates, [batch_size, -1, tf.shape(input)[-1]])
    updates = tf.transpose(updates, [0, 2, 1])
    updates = tf.reshape(updates, [batch_size, -1])

    # Perform the scatter operation to create the upsampled tensor
    upsampled = tf.scatter_nd(indices[:, tf.newaxis], updates, shape=[batch_size, new_height*new_width, tf.reduce_prod(tf.shape(input)[3:])])

    # Reshape the upsampled tensor to the output shape
    upsampled = tf.reshape(upsampled, [batch_size, new_height, new_width, -1])
    upsampled.set_shape(output_shape)

    return upsampled




# def conv__shape(input_shape, kernel_size, stride=1, padding='same'):
#     padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)    
#     height = math.ceil(float(input_shape[1] - kernel_size[0] + 2 * padding[0]) / stride + 1)
#     width = math.ceil(float(input_shape[2] - kernel_size[1] + 2 * padding[1]) / stride + 1)
#     output_shape = (input_shape[0], height, width, input_shape[3])
#     return output_shape

def pool__size(input_size, pool_size, stride):
    output_size = math.floor((input_size - pool_size) / stride) + 1
    return output_size
def network(input):
    # Encoding layers
    global pool5, ind5,conv5
    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
        conv1 = slim.conv2d(input, 64, [3, 3], rate=1, scope='g_conv1_1')  
        conv1 = slim.conv2d(conv1, 64, [3, 3], rate=1, scope='g_conv1_2')
        pool1, ind1 = tf.nn.max_pool_with_argmax(conv1, [2, 2], strides=2, padding='VALID')
        print(conv1.shape)

        
       # shape1 = tf.shape(pool1)
              
        conv2 = slim.conv2d(pool1, 128, [3, 3], rate=1, scope='g_conv2_1')
        conv2 = slim.conv2d(conv2, 128, [3, 3], rate=1, scope='g_conv2_2')
        pool2, ind2 = tf.nn.max_pool_with_argmax(conv2, [2, 2], strides=2, padding='VALID')
        print(conv2.shape)

        
        conv3 = slim.conv2d(pool2, 256, [3, 3], rate=1, scope='g_conv3_1')
        conv3 = slim.conv2d(conv3, 256, [3, 3], rate=1, scope='g_conv3_2')
        conv3 = slim.conv2d(conv3, 256, [3, 3], rate=1, scope='g_conv3_3')
        pool3, ind3 = tf.nn.max_pool_with_argmax(conv3, [2, 2], strides=2, padding='VALID')
        print(conv3.shape)

        
        conv4 = slim.conv2d(pool3, 512, [3, 3], rate=1, scope='g_conv4_1')
        conv4 = slim.conv2d(conv4, 512, [3, 3], rate=1, scope='g_conv4_2')
        conv4 = slim.conv2d(conv4, 512, [3, 3], rate=1, scope='g_conv4_3')
        pool4, ind4 = tf.nn.max_pool_with_argmax(conv4, [2, 2], strides=2, padding='VALID')
        print(conv4.shape)

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, scope='g_conv5_1')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, scope='g_conv5_2')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, scope='g_conv5_3')
        pool5, ind5 = tf.nn.max_pool_with_argmax(conv5, [2, 2], strides=2, padding='VALID')
        print(conv5.shape)
    # Decoding layers
    with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):    
        
        print("---======---")
        #up6 = upsample(pool5, conv5,conv5.shape[-1],pool5.shape[-1])
        up6 = tf.image.resize(pool5, conv5.get_shape()[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv6 = slim.conv2d(up6, 512, [3, 3], rate=1, scope='g_conv6_1')
        conv6 = slim.conv2d(conv6, 512, [3, 3], rate=1, scope='g_conv6_2')
        conv6 = slim.conv2d(conv6, 512, [3, 3], rate=1, scope='g_conv6_3')
        print(up6.shape,conv6.shape)
    

        #up7 = upsample(pool4, conv6,conv6.shape[-1],pool4.shape[-1])
        up7 = tf.image.resize(conv6, conv4.get_shape()[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        conv7 = slim.conv2d(up7, 512, [3, 3], rate=1, scope='g_conv7_1')
        conv7 = slim.conv2d(conv7, 512, [3, 3], rate=1, scope='g_conv7_2')
        conv7 = slim.conv2d(conv7, 256, [3, 3], rate=1, scope='g_conv7_3')
        print(up7.shape,conv7.shape)

        
        #up8 = upsample(pool3, conv7,conv7.shape[-1],pool3.shape[-1])
        up8 = tf.image.resize(conv7,conv3.get_shape()[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        conv8 = slim.conv2d(up8, 256, [3, 3], rate=1, scope='g_conv8_1')
        conv8 = slim.conv2d(conv8, 256, [3, 3], rate=1, scope='g_conv8_2')
        conv8 = slim.conv2d(conv8, 128, [3, 3], rate=1, scope='g_conv8_3')
        print(up8.shape,conv8.shape)

        #up9 = upsample(pool2, conv8,conv8.shape[-1],pool2.shape[-1])
        up9 = tf.image.resize(conv8, conv2.get_shape()[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


        conv9 = slim.conv2d(up9, 128, [3, 3], rate=1, scope='g_conv9_1')
        conv9 = slim.conv2d(conv9, 64, [3, 3], rate=1, scope='g_conv9_2')
        print(up9.shape,conv9.shape)

    
        #up10 = upsample(pool1, conv9,conv9.shape[-1],pool1.shape[-1])
        up10 = tf.image.resize(conv9,conv1.get_shape()[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


        conv10 = slim.conv2d(up10, 64, [3, 3], rate=1, scope='g_conv10_1')
        conv10 = slim.conv2d(conv10, 64, [3, 3], rate=1, scope='g_conv10_2')
        print(up10.shape,conv10.shape)
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
print("start")
for epoch in range(lastepoch, 4001):
    if os.path.isdir(str(result_dir) + f'{epoch:04d}'):
        continue

    
    cnt = 0
    if epoch > 2000:
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

        print(f"{epoch} {cnt} Loss={np.mean(g_loss[np.where(g_loss)]):.3f} Time={time.time() - st:.3f}")

        if epoch % save_freq == 0:
            if not os.path.isdir(f"{result_dir}{epoch:04d}"):
                os.makedirs(f"{result_dir}{epoch:04d}")

            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            if not os.path.isdir(f"{result_dir}{epoch:04d}"):
                os.makedirs(f"{result_dir}{epoch:04d}")
            
            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            temp = (temp * 255).clip(0, 255).astype('uint8')
            img = Image.fromarray(temp)
            img.save(f"{result_dir}{epoch:04d}/{train_id:05d}_00_train_{ratio:d}.jpg")
            saver.save(sess, checkpoint_dir + 'model.ckpt')
            


    