import data
from edsr import EDSR

batch_size = 1
#model_name = 'edsr-10pic-4layer-3x3-64-Diamond'
#model_name = 'edsr-10pic-8layer-3x3-64-Diamond'
model_name = 'edsr-10pic-8layer-3x3-64-torus2-box3-CornellBox'
#model_name = 'test'
lists = [
        #{'src_dir': '/data3/lzh/10000x672x672_torus2_diff', 'filelist': 'train_filelist.txt'},
        #{'src_dir': '/data3/lzh/10000x672x672_box_diff', 'filelist': 'train_filelist.txt'},
        #{'src_dir': '/data3/lzh/10000x672x672_box3_diff', 'filelist': 'train_filelist.txt'},
        {'src_dir': '/data3/lzh/10000x672x672_Diamond_diff', 'filelist': 'test_filelist.txt'},
        #{'src_dir': '/data3/lzh/10000x672x672_CornellBox_diff', 'filelist': 'train_filelist.txt'},
        ]
cnt, dataset = data.load_dataset(lists, batch_size)
print 'Load', cnt , 'pics'

edsr = EDSR(num_layers = 8)
edsr.resume(src = model_name + '/')
edsr.test(cnt, dataset, '/data3/lzh/deploy/'+model_name)
