import tensorflow.contrib.slim as slim
import tensorflow as tf
import os

class EDSR(object):
    def resBlock(self, x):
        conv1 = slim.conv2d(x, self.feature_size, self.kernel_size)
        conv2 = slim.conv2d(conv1, self.feature_size, self.kernel_size, activation_fn = None)
        #conv2 *= scaling
        return x + conv2

    def __init__(
            self,
            pics = 10,
            num_layers = 4,
            kernel_size = [3,3],
            feature_size = 64,
            scaling = 1):
        self.input = rough_pics = tf.placeholder(tf.float32, [None, None, None, pics * 3])
        self.target = target = tf.placeholder(tf.float32, [None, None, None, 3])

        self.kernel_size = kernel_size
        self.feature_size = feature_size
        #self.scaling = scaling
        
        # First conv
        conv1 = slim.conv2d(rough_pics, feature_size, kernel_size)
        conv = conv1

        for i in xrange(num_layers):
            conv = self.resBlock(conv)

        conv = conv + conv1

        self.output = output = slim.conv2d(conv, 3, kernel_size, activation_fn = None)

        self.loss = tf.reduce_mean(tf.losses.absolute_difference(target, output))

        self.MSE = MSE = tf.reduce_mean(tf.squared_difference(target, output))
        PSNR = tf.constant(255 ** 2, dtype = tf.float32) / MSE
        PSNR = tf.log(PSNR) / tf.log(tf.constant(10, dtype = tf.float32))
        self.PSNR = PSNR = tf.constant(10, dtype = tf.float32) * PSNR


        #Scalar to keep track for loss
        sl = tf.summary.scalar("loss",self.loss)
        sm = tf.summary.scalar("MSE",self.MSE)
        sp = tf.summary.scalar("PSNR",self.PSNR)
        #Image summaries for input, target, and output
        for i in xrange(pics):
            tf.summary.image("input_image_" + str(i),tf.cast(self.input[:,:,:,i * 3:i * 3 + 3],tf.uint8))
        tf.summary.image("target_image",tf.cast(self.target,tf.uint8))
        tf.summary.image("output_image",tf.cast(tf.clip_by_value(self.output, 0.0, 255.0),tf.uint8))
        self.merged_summary = tf.summary.merge_all()
        self.loss_summary = tf.summary.merge([sl, sm, sp])
        
        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.saver = tf.train.Saver()
        print 'Build Network Done!'

    # Save the current state of the network to file
    def save(self, save_dir = 'model/', save_file = 'model.ckpt', step = 0):
        dst = os.path.join(save_dir, save_file)
        print("Saving to:")
        print '    ', self.saver.save(self.sess, dst, global_step = step)
        print("Saved!")

    # Resume network from previously saved weights
    def resume(self, src = 'model/'):
        print("Restoring from " + src)
        self.saver.restore(self.sess,tf.train.latest_checkpoint(src))
        print("Restored!")

    def test(self, dataset_size, dataset, dst_dir):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        with self.sess as sess:
            cnt = 0
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            sess.run(iterator.initializer)
            while True:
                try:
                    base_filenames, x, y = sess.run(next_element)
                except tf.errors.OutOfRangeError:
                    break
                res = sess.run(self.output, {self.input: x})
                res = tf.cast(tf.clip_by_value(res, 0.0, 255.0), tf.uint8)
                for i in xrange(res.shape[0]):
                    enc = tf.image.encode_jpeg(res[i,:,:,:])
                    filename = os.path.join(dst_dir, 'test' + base_filenames[i] + '.jpg')
                    print filename
                    fwrite = tf.write_file(tf.constant(filename), enc)
                    sess.run(fwrite)

    def train(self, epoch, dataset_size, dataset, model_name):
        train_op = tf.train.AdamOptimizer().minimize(self.loss)
        init = tf.global_variables_initializer()
        print("Begin training...")
        with self.sess as sess:
            sess.run(init)
            #create summary writer for train
            train_writer = tf.summary.FileWriter(os.path.join('logs/', model_name),sess.graph)

            #This is our training loop
            cnt = 0
            for i in range(epoch):
                iterator = dataset.make_initializable_iterator()
                next_element = iterator.get_next()
                sess.run(iterator.initializer)
                print 'Epoch', i, 'Iter', cnt
                while True:
                    try:
                        base_filename, x, y = sess.run(next_element)
                    except tf.errors.OutOfRangeError:
                        break
                    cnt += 1
                    #Create feed dictionary for the batch
                    feed = {
                            self.input:x,
                            self.target:y
                    }
                    if (cnt % 100 == 0) or (cnt <= 100):
                        summary = self.merged_summary
                    else:
                        summary = self.loss_summary
                    #Run the train op and calculate the train summary
                    summary, MSE, _ = sess.run([summary, self.MSE, train_op],feed)

                    print 'Epoch', i, 'Iter', cnt, "MSE:", MSE
                    train_writer.add_summary(summary, cnt)
                #Save our trained model
                self.save(save_dir = model_name + '/', save_file = model_name + '.ckpt', step = i)
