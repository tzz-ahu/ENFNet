import cv2
import numpy as np
import NLDF
import vgg16
import tensorflow as tf
import os
import pdb


def load_training_list():

    with open('./dataset/MSRA-B/train_cvpr2013.txt') as f:
        lines = f.read().splitlines()

    files = []
    files_edge = []
    labels = []

    for line in lines:
        labels.append('./dataset/MSRA-B/annotation/%s' % line)
        files.append('./dataset/MSRA-B/image/%s' % line.replace('.png', '.jpg'))
        files_edge.append('./dataset/MSRA-B/image_edge/%s' % line.replace('.png', '.jpg'))

    return files, files_edge, labels


def load_train_val_list():

    files = []
    files_edge = []
    labels = []

    with open('./dataset/MSRA-B/train_cvpr2013.txt') as f:
        lines = f.read().splitlines()

    for line in lines:
        labels.append('./dataset/MSRA-B/annotation/%s' % line)
        files.append('./dataset/MSRA-B/image/%s' % line.replace('.png', '.jpg'))
        files_edge.append('./dataset/MSRA-B/image_edge/%s' % line.replace('.png', '.jpg'))

    with open('./dataset/MSRA-B/valid_cvpr2013.txt') as f:
        lines = f.read().splitlines()

    for line in lines:
        labels.append('./dataset/MSRA-B/annotation/%s' % line)
        files.append('./dataset/MSRA-B/image/%s' % line.replace('.png', '.jpg'))
        files_edge.append('./dataset/MSRA-B/image_edge/%s' % line.replace('.png', '.jpg'))

    return files, files_edge, labels


if __name__ == "__main__":

    model = NLDF.Model()
    model.build_model()

    sess = tf.Session()

    max_grad_norm =1
    tvars= tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.Loss_Mean, tvars), max_grad_norm)
    opt = tf.train.AdamOptimizer(1e-5)
    train_op = opt.apply_gradients(zip(grads, tvars))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    train_list, train_edge_list, label_list = load_train_val_list()

    n_epochs =4
    img_size = NLDF.img_size
    label_size = NLDF.label_size

    for i in xrange(n_epochs):
        whole_loss = 0.0
        whole_acc = 0.0
        count = 0
        for f_img, f_edge_img, f_label in zip(train_list, train_edge_list, label_list):

            img = cv2.imread(f_img).astype(np.float32)
            img_edge = cv2.imread(f_edge_img).astype(np.float32)

            img_flip = cv2.flip(img, 1)
            img_edge_flip = cv2.flip(img_edge, 1)

            label = cv2.imread(f_label)[:, :, 0].astype(np.float32)
            label_flip = cv2.flip(label, 1)

            img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
            img_edge = cv2.resize(img_edge, (img_size, img_size)) - vgg16.VGG_MEAN

            label = cv2.resize(label, (label_size, label_size))
            label = label.astype(np.float32) / 255.

            img = img.reshape((1, img_size, img_size, 3))
            img_edge = img_edge.reshape((1, img_size, img_size, 3))

            label = np.stack((label, 1-label), axis=2)
            label = np.reshape(label, [-1, 2])

            _, loss, acc = sess.run([train_op, model.Loss_Mean, model.accuracy],
                                    feed_dict={model.input_holder: img,
                                               model.input_edge_holder: img_edge,
                                               model.label_holder: label})

            whole_loss += loss
            whole_acc += acc
            count = count + 1

          #add horizon flip image for training
            img_flip = cv2.resize(img_flip, (img_size, img_size)) - vgg16.VGG_MEAN
            img_edge_flip = cv2.resize(img_edge_flip, (img_size, img_size)) - vgg16.VGG_MEAN

            label_flip = cv2.resize(label_flip, (label_size, label_size))
            label_flip = label_flip.astype(np.float32) / 255.

            img_flip = img_flip.reshape((1, img_size, img_size, 3))
            img_edge_flip=img_edge_flip.reshape((1,img_size,img_size,3))

            label_flip = np.stack((label_flip, 1 - label_flip), axis=2)
            label_flip = np.reshape(label_flip, [-1, 2])

            _, loss, acc = sess.run([train_op, model.Loss_Mean, model.accuracy],
                                      feed_dict={model.input_holder: img_flip,
                                                 model.input_edge_holder: img_edge_flip,
                                                 model.label_holder: label_flip})

            whole_loss += loss
            whole_acc += acc
            count = count + 1

           # print "Loss of %d images: %f, Accuracy: %f" % (count, (whole_loss / count), (whole_acc / count))
            if count % 200 == 0:
                print "Loss of %d images: %f, Accuracy: %f" % (count, (whole_loss/count), (whole_acc/count))

        print "Epoch %d: %f" % (i, (whole_loss/len(train_list)))

    os.mkdir('Model')
    saver.save(sess, 'Model/model.ckpt', global_step=n_epochs)
