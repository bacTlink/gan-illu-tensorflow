import data
from edsr import EDSR

batch_size = 20
#model_name = 'edsr-10pic-4layer-3x3-64-Diamond'
#model_name = 'edsr-10pic-8layer-3x3-64-Diamond'
#model_name = 'edsr-10pic-8layer-3x3-64-box3'
#model_name = 'edsr-10pic-8layer-3x3-64-CornellBox'
model_name = 'edsr-10pic-8layer-3x3-64-torus2'
#model_name = 'edsr-10pic-8layer-3x3-64-conference'
#model_name = 'edsr-10pic-8layer-3x3-64-box'
#model_name = 'edsr-10pic-4layer-3x3-64-torus2-box3-CornellBox'
#model_name = 'edsr-10pic-8layer-3x3-64-torus2-box3-CornellBox'
#model_name = 'test'
lists = [
        {'src_dir': '/data3/lzh/10000x672x672_torus2_diff', 'filelist': 'train_filelist.txt'},
        #{'src_dir': '/data3/lzh/10000x672x672_box_diff', 'filelist': 'train_filelist.txt'},
        #{'src_dir': '/data3/lzh/10000x672x672_box3_diff', 'filelist': 'train_filelist.txt'},
        #{'src_dir': '/data3/lzh/10000x672x672_Diamond_diff', 'filelist': 'train_filelist.txt'},
        #{'src_dir': '/data3/lzh/10000x672x672_CornellBox_diff', 'filelist': 'train_filelist.txt'},
        #{'src_dir': '/data3/lzh/10000x672x672_CornellBox_diff', 'filelist': 'train_filelist.txt'},
        #{'src_dir': '/data3/lzh/10000x672x672_conference_diff', 'filelist': 'train_filelist.txt'},
        ]
cnt, dataset = data.load_dataset(lists, batch_size, 224, shuffle = True, flip = True)
print 'Load', cnt , 'pics'

edsr = EDSR(num_layers = 8)
edsr.train(100, cnt, dataset, model_name = model_name)
