import os

#I USED THIS FILE TO CONVERT NIFTI IMAGES INTO NUMPY ARRAYS
from medpy.io import load, save
import numpy as np

def writeFiles(imageFiles, labelFiles, inputImageDir, inputLabelDir, outputImageDir, outputLabelDir, train=False):
    for i in range(len(imageFiles)):
        imageFile = imageFiles[i]
        print('Reading File {}'.format(imageFile), end='\r')
        labelFile = labelFiles[i]
        image, _ = load(inputImageDir + imageFile)
        label, _ = load(inputLabelDir + labelFile)
        if train:
            for i in range(image.shape[2]):
                imageSlice = image[:, :, i]
                labelSlice = label[:, :, i]
                np.save(outputImageDir + imageFile + 'slice_{}'.format(i), imageSlice)
                np.save(outputLabelDir + labelFile + 'slice_{}'.format(i), labelSlice)
        else:
            np.save(outputImageDir + imageFile, image)
            np.save(outputLabelDir + labelFile, label)
def main():
    trainImagesDir = 'niftiImages/raw/'
    trainLabelsDir = 'niftiImages/hand_mask/'

    testImagesDir = 'test_mask/'
    testLabelsDir = 'test_raw/'

    trainImageFiles = [file for file in os.listdir(trainImagesDir) if file.endswith('.nii')]
    trainLabelFiles = ['mask_' + file for file in trainImageFiles]

    testImageFiles = [file for file in os.listdir(testImagesDir) if file.endswith('.npy')]
    testLabelFiles = ['mask_' + file for file in testImageFiles]

    valdImageFiles = trainImageFiles[:100]
    valdLabelFiles = trainLabelFiles[:100]
    trainImageFiles = trainImageFiles[100:]
    trainLabelFiles = trainLabelFiles[100:]

    writeFilesNumpy(testImageFiles, testLabelFiles, testImagesDir, testLabelsDir, 'testSliceImages/', 'testSliceMasks/', train=True)
    writeFiles(trainImageFiles, trainLabelFiles, trainImagesDir, trainLabelsDir, 'trainImages/', 'trainMasks/', train=True)
    writeFiles(valdImageFiles, valdLabelFiles, trainImagesDir, trainLabelsDir, 'valdImages/', 'valdMasks/')
    writeFiles(testImageFiles, testLabelFiles, testImagesDir, testLabelsDir, 'testImages/', 'testMasks/')

main()
