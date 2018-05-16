import data
from edsr import EDSR

batch_size = 20
model_name = 'edsr-10pic-8layer-3x3-64'
lists = [
        {'src_dir': '/data3/lzh/10000x672x672_torus2_diff', 'filelist': 'train_filelist.txt'},
        {'src_dir': '/data3/lzh/10000x672x672_box_diff', 'filelist': 'train_filelist.txt'},
        {'src_dir': '/data3/lzh/10000x672x672_box3_diff', 'filelist': 'train_filelist.txt'},
        {'src_dir': '/data3/lzh/10000x672x672_Diamond_diff', 'filelist': 'train_filelist.txt'}
        ]
cnt, dataset = data.load_dataset(lists, batch_size)
print 'Load', cnt , 'pics'

edsr = EDSR(num_layers = 8)
edsr.train(100, cnt, dataset, savefile = model_name, logfile = 'log-' + model_name)
