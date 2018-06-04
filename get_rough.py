import os, sys
import numpy as np
import cv2

model = 'box'
version = ''
src_dir = '/data3/lzh/10000x672x672_'+model+version+'_diff/'
deploy_dir = '/data3/lzh/deploy/edsr-10pic-8layer-3x3-64-'+model+version
dst_dir = '/home/linzehui/'
img_count = 10
background = '/home/linzehui/'+model+'-background.png'
background_img = cv2.imread(background)
base_filename = model+version+'--c0-l0-r0'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

x = 120
y = 260
# process data
imgs = []
for i in xrange(1, img_count + 1):
    filename = base_filename + '_' + str(i) + '.png'
    img = cv2.imread(os.path.join(src_dir, filename))
    img = cv2.add(img, background_img)
    #img = img[x:x+100,y:y+100,:]
    #cv2.rectangle(img, (y,x), (y+100,x+100),(55,55,255),5)
    imgs.append(img)

res1 = np.concatenate(imgs[0:5], axis = 1)
res2 = np.concatenate(imgs[5:10], axis = 1)
res = np.concatenate((res1, res2), axis = 0)
cv2.imwrite(os.path.join(dst_dir, model+version+"-input.jpg"), res)

ref = cv2.imread(os.path.join(src_dir, base_filename + '_1000.png'))
out = cv2.imread(os.path.join(deploy_dir, 'test'+base_filename + '.jpg'))
ref = cv2.add(ref, background_img)
out = cv2.add(out, background_img)
#out = out[x:x+100,y:y+100,:]
#ref = ref[x:x+100,y:y+100,:]
#cv2.rectangle(ref, (y,x), (y+100,x+100),(55,55,255),5)
#cv2.rectangle(out, (y,x), (y+100,x+100),(55,55,255),5)
res = np.concatenate((out, ref), axis = 1)
cv2.imwrite(os.path.join(dst_dir, model+version+"-cmp.jpg"), res)

res = cv2.absdiff(ref, out)
res = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
res[:,:,2] = [[min(int(pixel) * int(pixel), 255) for pixel in row] for row in res[:,:,2]]
res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
cv2.imwrite(os.path.join(dst_dir, model+version+"-absdiff.jpg"), res)
