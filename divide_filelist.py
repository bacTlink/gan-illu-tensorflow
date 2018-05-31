import os
import random

def divide_filelists(filelists):
    for l in filelists:
        src_dir = l['src_dir']
        filelist = l['filelist']
        train_filelist = os.path.join(src_dir, 'train_' + filelist)
        test_filelist = os.path.join(src_dir, 'test_' + filelist)
        train_file = open(train_filelist, 'w')
        test_file = open(test_filelist, 'w')
        filelist = os.path.join(src_dir, filelist)
        ri = 100;
        for line in open(filelist).readlines():
            if ri <= 90:
                # phase = 'train'
                train_file.write(line)
            else:
                # phase = 'test'
                test_file.write(line)
            ri = random.randint(1, 100)
        train_file.close()
        test_file.close()

if __name__ == "__main__":
    lists = [
            #{'src_dir': '/data3/lzh/10000x672x672_torus2_diff', 'filelist': 'filelist.txt'},
            #{'src_dir': '/data3/lzh/10000x672x672_box_diff', 'filelist': 'filelist.txt'},
            #{'src_dir': '/data3/lzh/10000x672x672_box3_diff', 'filelist': 'filelist.txt'},
            #{'src_dir': '/data3/lzh/10000x672x672_Diamond_diff', 'filelist': 'filelist.txt'}
            {'src_dir': '/data3/lzh/10000x672x672_conference_diff', 'filelist': 'filelist.txt'}
            ]
    divide_filelists(lists)
