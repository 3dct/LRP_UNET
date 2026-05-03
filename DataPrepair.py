
import numpy as np


import SimpleITK as sitk

inputImageFileName =".\FibreBundle\IP532_Sample4-2umVS-16bit_1670x400x1670-Orig.mhd"
outdir = ".\out2\\"

image = sitk.ReadImage(inputImageFileName)

imageArray = sitk.GetArrayFromImage(image)

shape = imageArray.shape

for i in range(shape[0]):
    outImage = sitk.GetImageFromArray(imageArray[i])
    sitk.WriteImage(outImage, outdir +f"img{i}.png")