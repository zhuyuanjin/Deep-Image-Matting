##Copyright 2017 Adobe Systems Inc.
##
##Licensed under the Apache License, Version 2.0 (the "License");
##you may not use this file except in compliance with the License.
##You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##Unless required by applicable law or agreed to in writing, software
##distributed under the License is distributed on an "AS IS" BASIS,
##WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##See the License for the specific language governing permissions and
##limitations under the License.


##############################################################
# Set your paths here

# path to provided foreground images
fg_path = 'fg/'

# path to provided alpha mattes
a_path = 'mask/'

# Path to background images (MSCOCO)
bg_path = 'bg/'

# Path to folder where you want the composited images to go
out_path = 'merged/'

##############################################################

from PIL import Image
import os
import math
import time



file = open('./img_name.txt', 'w')

num_bgs = 100

fg_files = os.listdir(fg_path)
a_files = os.listdir(a_path)
bg_files = os.listdir(bg_path)

bg_iter = iter(bg_files)
for im_name in fg_files:


    bcount = 0
    for i in range(num_bgs):
        bg_name = next(bg_iter)
        comp_name = im_name[:len(im_name) - 4] + '_' + str(bcount) + '.png'
        file.write(im_name + ' ')
        file.write(bg_name + " ")
        file.write(comp_name + " ")
        bcount += 1



