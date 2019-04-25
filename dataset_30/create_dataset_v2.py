# -*- coding: utf-8 -*-
"""
    This code is for generative a New Dataset for different scale font images
Data Format is :
    dataset
    | ----  HYXXXX_train
    | ----  HYXXXX__test
    | ----  HYXXXX_train
    | ----  HYXXXX_test
    | ----  HYXXXX_train
    | --------- |  S3_C0_64.jpg
    | --------- |  S3_C1_64/128/256/512.jpg
    | --------- |  S3_C2_64/128/256/512.jpg
    | ----  HYXXXX_test
    | --------- |  S3_C10_64.jpg
    | --------- |  S3_C11_64/128/256/512.jpg
    | --------- |  S3_C12_64/128/256/512.jpg

The " *_train " folder is target image for NN generate when train phase
The " *_test " folder is will be generated when test phase.
Sx it meaning Font Style label is x , for now x shoudl be belong {0, 124}
Cy it meaning Character label is y , and y should be belong {0,6762}
"""
import cv2,math
from freetype import *
import numpy as np
import os,sys, random
import time
from multiprocessing import Pool

device = 'local' # 'local' or 'server'
if device == 'local':
    config = {
    # char_path is path of 6763 characters text file 
    'char_path':"/home/yue/DataSets/font_dataset/chars_6733_removed.txt",  # chars_6763_test or chars_6733_removed.txt or hand_824.txt
    # save_path is path of where image will be saved
    'save_path':"/media/yue/Data/Zi2Zi_30_256/",#"/media/yue/Backup_Data/Zi2Zi_New_Data/",  Zi2Zi_810 / Zi2Zi_Test / Zi2Zi_167
    # font_path is path of where typeface file (*.ttf)
    'font_path':"/home/yue/DataSets/font_dataset/Zi2Zi_Data/Hands_30/",#"/home/yue/DataSets/font_dataset/Zi2Zi_Data/ALL_Hand/",  [810]   Hands_167 / Hands_30
    
    # char_size is Image Size 
    'char_size':[256],    #[64,128,256,512]
    }

    label_file = "/home/yue/DataSets/font_dataset/chars_6733_removed.txt"
elif device == 'server':
    config = {
    # char_path is path of 6763 characters text file 
    'char_path':"/home/ubuntu/DataSets/font_dataset/chars_6733_removed.txt",  # chars_6763_test
    # save_path is path of where image will be saved
    'save_path':"/home/ubuntu/media_data/Zi2Zi_810_DataSets/",#"/media/yue/Backup_Data/Zi2Zi_New_Data/",  Zi2Zi_New_Data
    # font_path is path of where typeface file (*.ttf)
    'font_path':"/home/ubuntu/DataSets/Zi2Zi_Data/ALL_Hand/",#"/home/yue/DataSets/font_dataset/Zi2Zi_Data/ALL_Hand/", 
    # char_size is Image Size 
    'char_size':[64],    #[64,128,256,512]
    }

    label_file = "/home/ubuntu/DataSets/font_dataset/chars_6733_removed.txt"


with open(label_file, 'r') as lf:
    files = lf.readlines()
    char_6733 = [ c.strip("\n") for c in files]
    
class Font_Generator():

    def __init__(self, config):
        self.config = config
        assert os.path.isfile(self.config['char_path']) == True, print("char_path is not exists!")
        assert os.path.isdir(self.config['font_path']) == True, print("font_path is not exists!")
        if os.path.isdir(self.config['save_path']) == False:
            os.mkdir(self.config['save_path'])
            print("Make *Save Dir* at ",self.config['save_path'])
        else:
            print("image will be save in ",self.config['save_path'])

        self.get_all_fonts()
        self.get_all_chars()

        self.char_size = self.config['char_size']

    def get_all_fonts(self):
        self.font_files = os.listdir(self.config['font_path'])
        self.font_files = [self.config['font_path']+x for x in self.font_files if "ttf" in x]

        with open(self.config['save_path']+"font_labelmap.txt", 'w') as f:
            files = [c+" "+str(self.font_files.index(c))+"\n" for c in self.font_files]
            f.writelines(files)
    
    def get_all_chars(self):
        with open(self.config['char_path'],'r') as f:
            tmp_list = f.readlines()
        self.char_list = [c.strip("\n") for c in tmp_list]


    def writeJPG(self, image, save_path):
        """ Write __image__ into __save_path__"""
        #rows, width = image.shape
        #rbgImage = np.zeros((rows, width,3), dtype=np.uint8)
        #rbgImage[:,:,0] = image       # b
        #rbgImage[:,:,1] = image       # g
        #rbgImage[:,:,2] = image       # r
        cv2.imwrite(save_path, image)

    def get_plain(self, ttf, ttf_label ,char_list):
        
        char_size = self.char_size

        self.images = {}

        face = Face(ttf)
        for c_sz in char_size:
            font_size = c_sz
            suffix_image = "%d.png"%c_sz
            # write train dataset
            for  idxx, c in enumerate( char_list):
                #print(" Processing : {} in size {}...".format(c, c_sz))
                try:
                    mid_image = "C%d"%char_6733.index(c) #"C%d"%self.char_list.index(c)
                except:
                    print(" %s not in 6733 dataset. and skip it"%c)
                    continue
                face.set_char_size(font_size * 64)
                flags = FT_LOAD_RENDER
                face.load_char(c, flags)
                bitmap = face.glyph.bitmap

                maxValue = bitmap.width if bitmap.width >= bitmap.rows else bitmap.rows
                if maxValue == 0:
                    print(" This char :train_char {} not in font".format(c))
                    continue
                
                while c_sz < maxValue or c_sz - 10 > maxValue:
                    if maxValue > c_sz:
                        font_size -= 1
                    elif maxValue < c_sz -10:
                        font_size += 1
                    else:
                        break
                    face.set_char_size(font_size * 64)
                    face.load_char(c, flags)
                    bitmap = face.glyph.bitmap
                    maxValue = bitmap.width if bitmap.width >= bitmap.rows else bitmap.rows
                
                width, rows, pitch = bitmap.width, bitmap.rows, bitmap.pitch
                top, left = face.glyph.bitmap_top, face.glyph.bitmap_left
                Z2 = np.zeros((rows, width))
                dx ,dy = 0, 0
                data = np.array(bitmap.buffer[:rows*pitch]).reshape(rows, pitch)
                Z2[dx:dx+rows, dy:dy+width] = data[:, :width]

                image = np.zeros((c_sz, c_sz),dtype=np.uint8)
                dx = (c_sz - rows) / 2
                dy = (c_sz - width) /2
                dx, dy = int(dx), int(dy)
                image[dx:dx+rows, dy:dy+width] = Z2

                # save data into dict
                #image_name = prefix_image + "_" + mid_image + "_" + suffix_image
                image_name = mid_image  + "_" + suffix_image
                self.images[image_name] = image
                print("[ %4d / %4d ] ...cache Done."%(idxx, len(char_list)))
    
    def draw_all_chars(self, ttf):


        ttf_label = self.font_files.index(ttf)
        ttf_name = ttf.split("/")[-1].split(".")[0]
        root_path = self.config['save_path']+ttf_name+"/"
        if os.path.isdir(root_path) == False:
            os.mkdir(root_path)

        self.get_plain(ttf, ttf_label, self.char_list)

        
        
        i = 0
        num_chars = len(self.char_list) 
        # write image into file
        for f in self.images.keys():
            self.writeJPG(self.images[f], root_path+f)
            i +=1
            if i%100 ==0:print("S%d : processing....[%4d / %4d]" %(ttf_label, i, num_chars))
 
        print(" {} process Done!".format(ttf))

if __name__ == '__main__':
    g = Font_Generator(config)
    
    if False :
        # method 1 
        for ttf in g.font_files:
            t_time = time.time()
            g.draw_all_chars(ttf)
            cost_time = time.time() - t_time
            print("time is :", cost_time)
            pass
    else:
        # method 2
        p = Pool(40)
        p.map(g.draw_all_chars , g.font_files )
        p.close()
        p.join()

    print("Done ")
    







        
            
