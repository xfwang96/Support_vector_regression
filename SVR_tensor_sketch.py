import tensorflow as tf
import numpy as np
import os
import sklearn.model_selection as ms
# import sklearn.datasets as ds
import matplotlib.pyplot as plt
import pandas as pd
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

iris = ds.load_iris()

x_ = iris.data[50:, 0:3]
y_ = iris.data[50:, 3].reshape(-1, 1)



x_train, x_test, y_train, y_test = ms.train_test_split(x_, y_, train_size=0.8, test_size=0.2)


n_epoch = 80000
lr = 0.005
eps = 0.01
lam = 0.001
batch_size = 2000


p = 5
pro_dim = 128
input_dim = 3

x = tf.placeholder(tf.float32, [None, input_dim])
y = tf.placeholder(tf.float32, [None, 1])

a = tf.Variable(tf.random_uniform([p, 1]))
P = tf.complex(real=1.0, imag=0.0)

x_0 = tf.ones([tf.shape(x)[0], 1])
Feature = tf.concat([tf.multiply(a[0], x_0), tf.multiply(a[1], x)], axis=1)

for i in range(p):

    h = np.random.randint(low=0, high=pro_dim, size=[input_dim, 1])
    s = np.random.randint(low=0, high=2, size=[input_dim, 1]) * 2 - 1
    M_ = np.zeros(shape=[pro_dim, input_dim], dtype=np.float32)

    for j in range(input_dim):
        M_[h[j, 0], j] = s[j, 0]

    M = tf.transpose(M_)

    CountSketch = tf.to_complex64(tf.matmul(x, M))

    P = tf.multiply(P, tf.fft2d(CountSketch))

    Feature_ = tf.multiply(a[i], tf.real(tf.ifft2d(P)))

    if i > 1:
        Feature = tf.concat([Feature, Feature_], axis=1)

    print("Feature shape:  ")
    print(Feature.shape)

W = tf.Variable(tf.random_normal([1+input_dim+(p-2)*pro_dim, 1], stddev=0.35))

b = tf.Variable(tf.zeros([1]))
z = tf.matmul(Feature, W) + b

hinge_loss = tf.reduce_mean(tf.maximum(tf.abs(z - y) - eps, 0))
L2_loss = lam * tf.nn.l2_loss(W)
loss = hinge_loss + L2_loss

tf.summary.scalar('test_loss', loss)

op = tf.train.AdamOptimizer(lr).minimize(loss)

y_mean = tf.reduce_mean(y)
r_sqr = 1 - tf.reduce_sum((y - z) ** 2) / tf.reduce_sum((y - y_mean) ** 2)

merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=1)

epoch_losses = []
r_sqrs = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    data_size = len(x_train)
    index = np.arange(data_size)
    batch_number = np.int(np.ceil(data_size / batch_size))

    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, '../SVR/models/model.ckpt')
    # print("Model restored.")

    summary_writer = tf.summary.FileWriter('../SVR/logs', sess.graph)

    for e in range(n_epoch):
        epoch_loss_ = 0
        batch_losses = []
        np.random.shuffle(index)

        for j in range(batch_number):
            low = 0 + j * batch_size
            high = np.min([(j + 1) * batch_size, data_size])
            batch_index = index[low:high]

            x_batch = x_train[batch_index, :]
            y_batch = y_train[batch_index, :]
            _, batch_loss_ = sess.run([op, loss], feed_dict={x: x_batch, y: y_batch})

            batch_losses.append(batch_loss_)

        for i in range(batch_number):
            if i < batch_number-1:
                epoch_loss_ += batch_losses[i]
            elif data_size % batch_size == 0:
                epoch_loss_ += batch_losses[i]
            else:
                epoch_loss_ += (data_size % batch_size)/batch_size * batch_losses[i]

        epoch_loss_ /= batch_number
        epoch_losses.append(epoch_loss_)

        # r_sqr_, summary_str = sess.run([r_sqr, merged_summary_op], feed_dict={x: x_test, y: y_test})
        # r_sqrs.append(r_sqr_)

        if e % 1 == 0:
            # print('epoch: %d, loss: %f, r_sqr: %f' % (e, epoch_loss_, r_sqr_))
            print('batch_losses: ', end='')
            print(batch_losses)

            # summary_writer.add_summary(summary_str, e+1)
            # saver.save(sess, '../SVR/models/model.ckpt')

# open tensorboard in cmd
# tensorboard --logdir=/home/wxf/PycharmProjects/SVR/logs/

plt.figure()
plt.plot(epoch_losses)
plt.title('Loss on Training Set')
plt.xlabel('#epoch')
plt.ylabel('MSE')
plt.show()

plt.figure()
plt.plot(r_sqrs)
plt.title('$R^2$ on Testing Set')
plt.xlabel('#epoch')
plt.ylabel('$R^2$')
plt.show()
