import tensorflow as tf
import os
import random

def _parse_function(data, label):
    data_imgs = []
    for i in range(data.shape[0]):
        img_st = tf.read_file(data[i])
        img = tf.image.decode_image(img_st)
        data_imgs.append(img)
    data_imgs = tf.concat(data_imgs, 2)
    label_img = tf.image.decode_image(tf.read_file(label))
    merged = tf.concat((label_img, data_imgs), 2)
    merged = tf.random_crop(merged, [224, 224, data.shape[0] * 3 + 3])
    if (random.randint(0, 1) == 0):
        merged = tf.image.flip_left_right(merged)
    if (random.randint(0, 1) == 0):
        merged = tf.image.flip_up_down(merged)
    label_img = merged[:, :, 0:3]
    data_imgs = merged[:, :, 3:]
    return data_imgs, label_img

def load_dataset(lists, batch_size):
    labels = []
    data = []
    cnt = 0
    for l in lists:
        src_dir = l['src_dir']
        filelist = l['filelist']
        filelist = os.path.join(src_dir, filelist)
        for line in open(filelist).readlines():
            label_filename = line[:-1]
            labels.append(os.path.join(src_dir, label_filename))
            base_filename = label_filename[:-4]
            while base_filename[-1] >= '0' and base_filename[-1] <= '9':
                base_filename = base_filename[:-1]
            base_filename = base_filename[:-1]
            tmp_data = []
            for i in xrange(1, 11):
                tmp_data.append(os.path.join(src_dir, base_filename + '_' + str(i) + '.png'))
            data.append(tmp_data)
            cnt += 1
    labels = tf.constant(labels)
    data = tf.constant(data)

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(buffer_size = cnt)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5 * batch_size)

    return cnt, dataset

if __name__ == '__main__':
    lists = [
            {'src_dir': '/data3/lzh/10000x672x672_torus2_diff', 'filelist': 'test_filelist.txt'},
            {'src_dir': '/data3/lzh/10000x672x672_box_diff', 'filelist': 'test_filelist.txt'},
            {'src_dir': '/data3/lzh/10000x672x672_box3_diff', 'filelist': 'test_filelist.txt'},
            {'src_dir': '/data3/lzh/10000x672x672_Diamond_diff', 'filelist': 'test_filelist.txt'}
            ]
    cnt, dataset = load_dataset(lists, 10)
    print cnt

