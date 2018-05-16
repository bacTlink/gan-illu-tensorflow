import tensorflow.contrib.slim as slim
import tensorflow as tf
import os

class EDSR(object):
    def resBlock(self, x):
        conv1 = slim.conv2d(x, self.feature_size, self.kernel_size, activation_fn = tf.nn.relu)
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
        conv1 = slim.conv2d(rough_pics, feature_size, kernel_size, activation_fn = None)
        conv = conv1

        for i in xrange(num_layers):
            conv = self.resBlock(conv)
        conv = slim.conv2d(conv, feature_size, kernel_size, activation_fn = None)

        conv = conv + conv1


        self.output = output = slim.conv2d(conv, 3, kernel_size, activation_fn = None)

        self.loss = tf.reduce_mean(tf.losses.absolute_difference(target, output))

        self.MSE = MSE = tf.reduce_mean(tf.squared_difference(target, output))
        PSNR = tf.constant(255 ** 2, dtype = tf.float32) / MSE
        PSNR = tf.log(PSNR) / tf.log(tf.constant(10, dtype = tf.float32))
        self.PSNR = PSNR = tf.constant(10, dtype = tf.float32) * PSNR


        #Scalar to keep track for loss
        tf.summary.scalar("loss",self.loss)
        tf.summary.scalar("MSE",self.MSE)
        tf.summary.scalar("PSNR",self.PSNR)
        #Image summaries for input, target, and output
        for i in xrange(pics):
            tf.summary.image("input_image_" + str(i),tf.cast(self.input[:,:,:,i * 3:i * 3 + 3],tf.uint8))
        tf.summary.image("target_image",tf.cast(self.target,tf.uint8))
        tf.summary.image("output_image",tf.cast(self.output,tf.uint8))
        
        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.saver = tf.train.Saver()
        print 'Build Network Done!'

    # Save the current state of the network to file
    def save(self, savedir = 'saved_models', savefile = 'model'):
        dst = os.path.join(savedir, savefile)
        print("Saving to " + dst)
        self.saver.save(self.sess, dst)
        print("Saved!")

    # Resume network from previously saved weights
    def resume(self,savedir='saved_models', savefile = 'model'):
        src = os.path.join(savedir, savefile)
        print("Restoring from " + src)
        self.saver.restore(self.sess,tf.train.latest_checkpoint(src))
        print("Restored!")  

    def train(self, epoch, dataset_size, dataset, save_dir = 'snapshots/', savefile = 'model', logfile = 'train'):
        train_op = tf.train.AdamOptimizer().minimize(self.loss)
        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()
        print("Begin training...")
        with self.sess as sess:
            #Initialize all variables
            sess.run(init)
            #create summary writer for train
            train_writer = tf.summary.FileWriter(os.path.join(save_dir, logfile),sess.graph)

            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            #This is our training loop
            for i in range(epoch):
                sess.run(iterator.initializer)
                cnt = 0
                print 'Epoch', i, 'Iter', cnt
                while True:
                    cnt += 1
                    try:
                        x, y = sess.run(next_element)
                        #Create feed dictionary for the batch
                        feed = {
                                self.input:x,
                                self.target:y
                        }
                        #Run the train op and calculate the train summary
                        summary, _ = sess.run([merged,train_op],feed)
                        #Write train summary for this step
                        if ((i * dataset_size + cnt) % 10 == 0):
                            print 'Epoch', i, 'Iter', cnt
                            train_writer.add_summary(summary, i * dataset_size + cnt)
                    except tf.errors.OutOfRangeError:
                        break
            #Save our trained model
            self.save()
