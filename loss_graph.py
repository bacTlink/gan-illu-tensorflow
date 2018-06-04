import tensorflow as tf
import matplotlib.pyplot as plt

def get_loss(event_file, label):
    step = []
    MSE = []
    PSNR = []
    loss = []
    for e in tf.train.summary_iterator(event_file):
        has_MSE = False
        for v in e.summary.value:
            if v.tag == 'MSE':
                has_MSE = True
        if has_MSE:
            for v in e.summary.value:
                if v.tag == 'MSE':
                    MSE.append(v.simple_value)
                if v.tag == 'PSNR':
                    PSNR.append(v.simple_value)
                if v.tag == 'loss':
                    loss.append(v.simple_value)
            step.append(e.step)
    minus_step = step[0]
    div_step = (step[-1]-minus_step) / 100.0
    for i in xrange(len(step)):
        step[i] = (step[i] - minus_step) / div_step
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(step, loss, label=label)
    plt.legend(loc='best')
    plt.savefig('/home/linzehui/test.jpg')
    return [step, MSE, PSNR, loss]

if __name__ == '__main__':
    #get_loss('logs/edsr-10pic-8layer-3x3-64-Diamond/events.out.tfevents.1526651093.i23d-GPU', 'Diamond')
    #get_loss('logs/edsr-10pic-8layer-3x3-64-torus2/events.out.tfevents.1527003017.i23d-GPU', 'torus')
    get_loss('logs/edsr-10pic-8layer-3x3-64-box/events.out.tfevents.1526797302.i23d-GPU', 'box-3x3')
    get_loss('logs/edsr-10pic-8layer-5x5-64-box/events.out.tfevents.1527060744.i23d-GPU', 'box-5x5')

