
import csv
import random
from cv2 import cv2
import sys
import os
import numpy as np


def _print_debug_yes(s):
  print(s)

def _print_debug_no(s): 
  pass

_print_debug=_print_debug_no

def enclosing_square(rect):
    def _to_wh(s,l,ss,ll, width_is_long):
        if width_is_long:
            return l,s,ll,ss
        else:
            return s,l,ss,ll
    def _to_long_short(rect):
        x,y,w,h = rect
        if w>h:
            l,s,ll,ss = x,y,w,h
            width_is_long = True
        else:
            s,l,ss,ll = x,y,w,h
            width_is_long = False
        return s,l,ss,ll,width_is_long

    s,l,ss,ll,width_is_long = _to_long_short(rect)

    hdiff = (ll - ss)//2
    s-=hdiff
    ss = ll

    return _to_wh(s,l,ss,ll,width_is_long)

def add_margin(roi, qty):
    return (
     roi[0]-qty,
     roi[1]-qty,
     roi[2]+2*qty,
     roi[3]+2*qty )

def cut(frame, roi):
    pA = ( int(roi[0]) , int(roi[1]) )
    pB = ( int(roi[0]+roi[2]-1), int(roi[1]+roi[3]-1) ) #pB will be an internal point
    W,H = frame.shape[1], frame.shape[0]
    A0 = pA[0] if pA[0]>=0 else 0
    A1 = pA[1] if pA[1]>=0 else 0
    data = frame[ A1:pB[1], A0:pB[0] ]
    if pB[0] < W and pB[1] < H and pA[0]>=0 and pA[1]>=0:
        return data
    w,h = int(roi[2]), int(roi[3])
    img = np.zeros((h,w,frame.shape[2]), dtype=np.uint8)
    offX = int(-roi[0]) if roi[0]<0 else 0
    offY = int(-roi[1]) if roi[1]<0 else 0
    np.copyto( img[ offY:offY+data.shape[0], offX:offX+data.shape[1] ], data )
    return img

def pad(img):
    w,h,c = img.shape
    if w==h:
        return img
    size = max(w,h)
    out = np.zeros((size,size,c))
    np.copyto(out[0:w, 0:h], img)
    return out

def equalize_hist(img):
    if len(img.shape)>2 and img.shape[2] > 1:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(img)



############ FIT PLANE ##########
tmp_A = []
FIT_PLANE_SIZ=16
for y in np.linspace(0,1,FIT_PLANE_SIZ):
    for x in np.linspace(0,1,FIT_PLANE_SIZ):
        tmp_A.append([y, x, 1])
Amatrix = np.matrix(tmp_A)

def _fit_plane(im):
    original_shape=im.shape
    if len(im.shape)>2 and im.shape[2]>1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (FIT_PLANE_SIZ,FIT_PLANE_SIZ))
    if im.dtype==np.uint8:
        im = im.astype(float)
    # do fit
    A = Amatrix
    tmp_b = []
    for y in range(FIT_PLANE_SIZ):
        for x in range(FIT_PLANE_SIZ):
            tmp_b.append(im[y,x])
    b = np.matrix(tmp_b).T
    fit = (A.T * A).I * A.T * b

    fit[0]/=original_shape[0]
    fit[1]/=original_shape[1]

    def LR(x,y):
        return np.repeat(fit[0]*x,len(y),axis=0).T + np.repeat(fit[1]*y,len(x),axis=0) + fit[2]
    xaxis = np.array(range(original_shape[1]))
    yaxis = np.array(range(original_shape[0]))
    imest = LR(yaxis, xaxis)
    return np.array(imest)

    
def linear_balance_illumination(im):
    if im.dtype==np.uint8:
        im = im.astype(float)
    if len(im.shape)==2:
        im = np.expand_dims(im,2)
    if im.shape[2] > 1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    imout = im.copy()
    imest = _fit_plane(im[:,:,0])
    imout[:,:,0] = im[:,:,0] - imest + np.mean(imest)
    if im.shape[2] > 1:
        imout = cv2.cvtColor(imout, cv2.COLOR_YUV2BGR)
    return imout.reshape(im.shape)

############ END FIT PLANE ##########

def mean_std_normalize(inp, means=None, stds=None):
    assert(len(inp.shape)>=3)
    d = inp.shape[2]
    if means is None and stds is None:
        means = []
        stds = []
        for i in range(d):
            stds.append( np.std(inp[:,:,i]) )
            means.append( np.mean(inp[:,:,i]) )
            if stds[i] < 0.001:
                stds[i] = 0.001
    outim = np.zeros(inp.shape)
    for i in range(d):
        if stds is not None:
            outim[:,:,i] = (inp[:,:,i] - means[i]) / stds[i]
        else:
            outim[:,:,i] = (inp[:,:,i] - means[i])
    return outim

def _random_normal_crop(n, maxval, positive=False, mean=0):
    gauss = np.random.normal(mean,maxval/2,(n,1)).reshape((n,))
    gauss = np.clip(gauss, mean-maxval, mean+maxval)
    if positive:
        return np.abs(gauss)
    else:
      return gauss


def random_brightness_contrast(img):
    #brightness and contrast
    a = _random_normal_crop(1, 0.5, mean=1)[0]
    b = _random_normal_crop(1, 48)[0]
    _print_debug((a,b))
    img=(img-128.0)*a + 128.0 + b
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img
def random_flip(img):
    # flip
    if random.randint(0,1):
        img=np.fliplr(img)
    return img

def random_monochrome(x, random_fraction_yes=0.2):
    if random.random() < random_fraction_yes:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        if len(x.shape)==2:
            x = x[:,:,np.newaxis]
        x = np.repeat(x, 3, axis=2)
    return x

def random_fixed_size_roi(roi, original_size=(256,256), dst_size=(224,224)):
    dst_size_np = np.array(list(dst_size))
    original_size_np = np.array(list(original_size))
    diff = original_size_np - dst_size_np
    r = np.array([ random.randint(0, diff[0]),
                random.randint(0, diff[1])
            ])
    true_size_np = np.array([roi[2], roi[3]])
    new_size = dst_size_np*true_size_np/original_size_np
    r = r*true_size_np/original_size_np

    roi2 = (roi[0]+r[0], roi[1]+r[1], new_size[0], new_size[1])
    return roi2

def random_change_roi(roi, max_change_fraction=0.045, only_narrow=False):
    #random crop con prob + alta su 0 (gaussiana)
    sigma = roi[3]*max_change_fraction
    xy = _random_normal_crop(2, sigma, mean=-sigma/5).astype(int)
    wh = _random_normal_crop(2, sigma*2, mean=sigma/2, positive=only_narrow).astype(int)
    _print_debug( "orig roi: %s" % str(roi) )
    _print_debug( "rand changes -> xy:%s, wh:%s" % (str(xy), str(wh)))
    roi2 = (roi[0]+xy[0], roi[1]+xy[1], roi[2]-wh[0], roi[3]-wh[1])
    return roi2

def roi_center(roi):
    return (roi[0]+roi[2]//2, roi[1]+roi[3]//2)

def random_image_rotate(img, rotation_center):
    angle_deg = _random_normal_crop(1, 10)[0]
    M = cv2.getRotationMatrix2D(rotation_center, angle_deg, 1.0)
    nimg = cv2.warpAffine(img, M, dsize=img.shape[0:2])
    if len(nimg.shape)<3:
        nimg = nimg[:,:,np.newaxis]
    return nimg #.reshape(img.shape)

def random_image_skew(img, rotation_center):
    s = _random_normal_crop(2, 0.1, positive=True)
    M=np.array( [ [1,s[0],1], [s[1],1,1]] )
    nimg = cv2.warpAffine(img, M, dsize=img.shape[0:2])
    if len(nimg.shape)<3:
        nimg = nimg[:,:,np.newaxis]
    return nimg #.reshape(img.shape)

############### CUTOUT ################################
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.15, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):

    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

############################################


def _readcsv(csvpath, debug_max_num_samples=None):
  data = []
  with open(csvpath, newline='', encoding="utf8") as csvfile:
      reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      i = 0
      for row in reader:
          if debug_max_num_samples is not None and i>=debug_max_num_samples:
              break
          i=i+1
          data.append(row)
  return np.array(data)
 

# Assumes that every row sums "rowtotale"
def cntk_filtering(data, rowtotal=10, num_classes=8):
    # remove outlier votes
    data = np.array([float(x) for x in data])*(10/rowtotal)
    outliers = data<=1
    data[outliers] = 0
    
    totalvotes = np.sum(data)
    
    # remove examples from class 9 or 10
    hardlabel = np.argmax(data)
    if hardlabel == 8 or hardlabel==9:
        return True, None

    # remove examples with more than two winners
    maxvotes = np.max(data)
    winners = data==maxvotes
    nwinners = np.sum(winners)
    if nwinners > 2:
        return True, None

    # remove examples where the winners have <=50% of all votes
    numwinnervotes = nwinners*maxvotes
    if numwinnervotes <= 0.5*totalvotes:
        return True, None

    # return normalized
    data = data.astype(float)/totalvotes
    return False, data[0:num_classes]


def draw_emotion(y, w,h, emotion_labels=None):
    EMOTIONS = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt','unknown','NF']
    if emotion_labels is None: emotion_labels=EMOTIONS
    COLORS = [(120,120,120), (50,50,255), (0,255,255), (255,0,0), (0,0,140), (0,200,0), (42,42,165), (100,100,200), (170,170,170), (80,80,80)]
    emotionim = np.zeros((w,h,3), dtype=np.uint8)
    barh = h//len(EMOTIONS)
    MAXEMO = np.sum(y)
    for i,yi in enumerate(y):
        #print((EMOTIONS[i], yi))
        emoindex = EMOTIONS.index(emotion_labels[i])
        p1,p2 = (0,i*barh), (int(yi*w//MAXEMO), (i+1)*20)
        cv2.rectangle(emotionim, p1,p2, COLORS[emoindex], cv2.FILLED)
        cv2.putText(emotionim, "%s: %.1f" % (EMOTIONS[emoindex], yi), (0,i*20+14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
    return emotionim

            

def findRelevantFace(objs, W,H):
    mindistcenter = None
    minobj = None
    for o in objs:
        cx = o['roi'][0] + (o['roi'][2]/2)
        cy = o['roi'][1] + (o['roi'][3]/2)
        distcenter = (cx-(W/2))**2 + (cy-(H/2))**2
        if mindistcenter is None or distcenter < mindistcenter:
            mindistcenter = distcenter
            minobj = o
    return minobj
def top_left(f):
    return (f['roi'][0], f['roi'][1])
def bottom_right(f):
    return (f['roi'][0]+f['roi'][2], f['roi'][1]+f['roi'][3])


class VGGFace2Augmentation():
    def before_cut(self, frame, roi):
        frame = random_monochrome(frame, random_fraction_yes=0.2)
        return frame
    def augment_roi(self, roi):
        roi= add_margin(roi, 0.3)
        roi = random_fixed_size_roi(roi, original_size=(256,256), dst_size=(224,224))
        return roi
    def after_cut(self, img):
        img = random_flip(img)
        return img

class DefaultAugmentation():
    def before_cut(self, frame, roi):
        frame = random_image_rotate(frame, roi_center(roi))
        frame = random_image_skew(frame, roi_center(roi))
        return frame
    def augment_roi(self, roi):
        roi = random_change_roi(roi)
        roi = enclosing_square(roi)
        return roi
        
    def after_cut(self, img):
        img = random_brightness_contrast(img)
        img = random_flip(img)
        return img

# VGGFACE2_MEANS = np.array([131.0912, 103.8827, 91.4953]) # RGB
VGGFACE2_MEANS = np.array([91.4953, 103.8827, 131.0912]) # BGR

import keras
from math import ceil
from threading import Lock
import tensorflow
class DataGenerator(tensorflow.keras.utils.Sequence): # TODO VIGILANTE

    'Generates data for Keras'
    def __init__(self, data, target_shape, with_augmentation=True, batch_size=64, custom_augmentation=None, num_classes=None, preprocessing='full_normalization', fullinfo=False):
        if preprocessing not in ['full_normalization', 'z_normalization', 'vggface2', 'no_normalization']:
            raise Exception('unknown preprocessing: %s' % preprocessing)
        self.mutex = Lock()
        self.data = data
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.on_epoch_end()
        self.num_classes=num_classes
        self.preprocessing = preprocessing
        self.fullinfo = fullinfo
        if preprocessing == 'vggface2':
            self.ds_means = VGGFACE2_MEANS
            self.ds_stds = None
        elif preprocessing == "no_normalization":
            self.ds_means = None
            self.ds_stds = None
        else:    
            self.ds_means = np.array([0.485, 0.456, 0.406])*255
            self.ds_stds = np.array([0.229, 0.224, 0.225])*255
            
        if with_augmentation and custom_augmentation is None:
            self.augmentation = DefaultAugmentation()
        else:
            self.augmentation = custom_augmentation
        



    def __len__(self):
        nitems = len(self.data)
        return ceil(nitems/self.batch_size)

    def __getitem__(self, index):
        self.mutex.acquire()
        if self.cur_index >= len(self.data):
            print("Reset->unexpected!")
            #raise StopIteration
            self.cur_index = 0
        i = self.cur_index
        self.cur_index += self.batch_size
        self.mutex.release()
        data = self._load_batch(i)
        return tuple(data) # TODO VIGILANTE
        # return data

    def on_epoch_end(self):
        self.mutex.acquire()
        self.cur_index = 0
        print('Shuffle set')
        np.random.shuffle(self.data)
        self.mutex.release()

    def _load_item(self, d):
        roi = [int(x) for x in d['roi'] ]
        label = d['label']
        if self.num_classes is not None and isinstance(label,int):
            label = np.array(keras.utils.to_categorical(label, num_classes=self.num_classes))
        frame = d['img']
        if isinstance(frame, str):
            frame = cv2.imread(frame)
            if frame is None:
                print('ERROR: Unable to read image %s' % d['img'])
                return None
        if self.augmentation is not None:
            frame = self.augmentation.before_cut(frame, roi)
            roi = self.augmentation.augment_roi(roi)
        img = cut(frame, roi)
        
        if self.augmentation is not None:
            img = self.augmentation.after_cut(img)
        # Preprocess the image for the network
        img = cv2.resize(img, self.target_shape[0:2])
        if self.preprocessing=='full_normalization':
            img = equalize_hist(img)
            img = img.astype(np.float32)
            img = linear_balance_illumination(img)
            if np.abs(np.min(img)-np.max(img)) < 1:
                print("WARNING: Image is =%d" % np.min(img))
            else:
                img = mean_std_normalize(img)
        elif self.preprocessing=='z_normalization':
            img = mean_std_normalize(img, self.ds_means, self.ds_stds)
        elif self.preprocessing=='vggface2':
            img = mean_std_normalize(img, self.ds_means, self.ds_stds)
        if self.target_shape[2]==3 and (len(img.shape)<3 or img.shape[2]<3):
            img = np.repeat(np.squeeze(img)[:,:,None], 3, axis=2)
        
        if self.fullinfo:
            return (img, label, d['img'], roi)
        return (img, label)
    
    def _load(self, index):
        return self._load_item(self.data[index])
            
            
    def _load_batch(self, start_index, load_pairs=False):
        def get_empty_stuff(item):
            if item is None:
                return None
            stuff = []
            #stuff = [len(item)*[]]
            for j in range(len(item)):
                # np.empty( [0]+list(item[j].shape)[1:], item[j].dtype)
                stuff.append( list() )
            return stuff
        item = self._load(start_index)
        stuff = get_empty_stuff(item)
        size_of_this_batch = min(self.batch_size, len(self.data) - start_index)
        for index in range(start_index, start_index+size_of_this_batch):
            if item is None:
                item = self._load(index)
            for j in range(len(item)):
                stuff[j].append(item[j])
            item = None
        for j in range(len(stuff)):
            stuff[j]=np.array(stuff[j])
            if len(stuff[j].shape)==2 and stuff[j].shape[1]==1:
                stuff[j] = np.reshape(stuff[j], (stuff[j].shape[0],))
        return stuff

    # @property
    # def shape(self):
    #     return self.target_shape


