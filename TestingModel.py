import cv2
import numpy as np
import NLDF
import os
import sys
import tensorflow as tf
import time
import vgg16


def load_img_list(dataset):

    if dataset == 'MSRA-B':
        path = 'dataset/MSRA-B/image/'
        edge_path = 'dataset/MSRA-B/image_edge/'
    elif dataset == 'HKU-IS':
        path = 'dataset/HKU-IS/imgs/'
        edge_path = 'dataset/HKU-IS/imgs_edge/'
    if dataset == 'DUT-OMRON':
        path = 'dataset/DUT-OMRON/DUT-OMRON-image/'
        edge_path = 'dataset/DUT-OMRON/DUT-OMRON-image_edge/'
    elif dataset == 'PASCAL-S':
        path = 'dataset/PASCAL-S/pascal/'
        edge_path = 'dataset/PASCAL-S/pascal_edge/'
    elif dataset == 'SOD':
        path = 'dataset/BSDS300/imgs/'
        edge_path = 'dataset/BSDS300/imgs_edge/'
    elif dataset == 'ECSSD':
        path = 'dataset/ECSSD/images/'
        edge_path = 'dataset/ECSSD/images_edge/'

    imgs = os.listdir(path)
    edge_imgs = os.listdir(edge_path)
    return path, imgs, edge_path, edge_imgs


if __name__ == "__main__":

    model = NLDF.Model()
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    img_size = NLDF.img_size
    label_size = NLDF.label_size

    ckpt = tf.train.get_checkpoint_state('Model')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    datasets = ['HKU-IS', 'DUT-OMRON',
                'PASCAL-S', 'ECSSD', 'SOD']
    #datasets = ['PASCAL-S', 'ECSSD', 'SOD']

    if not os.path.exists('Result'):
        os.mkdir('Result')

    for dataset in datasets:
        path, imgs, edge_path, edge_imgs = load_img_list(dataset)

        save_dir = 'Result/' + dataset
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dir = 'Result/' + dataset + '/NLDF'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for f_img in imgs:

            img = cv2.imread(os.path.join(path, f_img))
            img_name, ext = os.path.splitext(f_img)

            img_edge = cv2.imread(os.path.join(edge_path, f_img))

            if not os.path.exists(os.path.join(edge_path, f_img)) :
                if ext is '.jpg':
                    edge_ext = '.png'
                else:
                    edge_ext = '.jpg'
                img_edge_path = os.path.join(edge_path, img_name+edge_ext)
                img_edge = cv2.imread(img_edge_path)
                img_edge_name, edge_ext = os.path.splitext(str.join(img_name,edge_ext))
            else:
                img_edge_name, edge_ext = os.path.splitext(f_img)

            if img is not None:
                ori_img = img.copy()
                img_shape = img.shape
                img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
                img = img.reshape((1, img_size, img_size, 3))

                ori_edge_img = img_edge.copy()
                img_edge_shape = img_edge.shape
                img_edge = cv2.resize(img_edge, (img_size, img_size)) - vgg16.VGG_MEAN
                img_edge = img_edge.reshape((1, img_size, img_size, 3))

                start_time = time.time()
                """
                result = sess.run(model.Prob,
                                  feed_dict={model.input_holder: img,
                                             model.keep_prob: 1})
                """
                result = sess.run(model.Prob,
                                  feed_dict={model.input_holder: img,
                                             model.input_edge_holder: img_edge})
                print("--- %s seconds ---" % (time.time() - start_time))

                result = np.reshape(result, (label_size, label_size, 2))
                result = result[:, :, 0]

                result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))

                save_name = os.path.join(save_dir, img_name+'_NLDF.png')
                cv2.imwrite(save_name, (result*255).astype(np.uint8))

    sess.close()
