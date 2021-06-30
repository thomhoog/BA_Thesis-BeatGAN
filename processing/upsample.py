import os
from PIL import Image
import numpy as np

inputDir = "" #specify path to images that need upsampling
netList = os.listdir(inputDir)

def inverse_repeat(a, repeats, axis):
    if isinstance(repeats, int):
        indices = np.arange(a.shape[axis] / repeats, dtype=np.int) * repeats
    else:  # assume array_like of int
        indices = np.cumsum(repeats) - 1
    return a.take(indices, axis)


for netDir in netList:
    imgDirList = os.listdir(inputDir+"\\"+netDir)
    for imgDir in imgDirList:
        imgList = os.listdir(inputDir+"\\"+netDir+"\\"+imgDir)
        i = 0
        os.chdir(inputDir+"\\"+netDir+"\\"+imgDir)
        #os.mkdir(f'upscaled_images_for_epoch_{i}')
        for img in imgList:
            if os.path.isfile(img):
                image = Image.open(img)
                imgarray = np.asarray(image)
                imgarray = inverse_repeat(imgarray, repeats=20, axis=0)
                imgarray= inverse_repeat(imgarray, repeats=20, axis=1)
                upscaled = Image.fromarray(imgarray)
                upscaled = upscaled.convert("L")
                upscaled.save(f'upscaled_images_for_epoch_{i}\\' + img)
        i = i + 1