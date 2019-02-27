import numpy as np
import matplotlib.pyplot as plt
import torch
import PIL
import os
import AugmentImage


class VisualizeImage(object):
    def __init__(self, list_of_images):
        plotRows,plotCols = 6,5
        cols = []; rows = []
        # fig, axes = plt.subplots(nrows=plotRows, ncols=plotCols)
        i = -1; j = 0
        for imNum, image in enumerate(list_of_images):
            print(image.shape)
            image = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)
            image = image.numpy().astype(dtype=np.float).transpose((1,2,0))
            # C,W,H = image.shape
            # image = np.reshape(image, newshape=(W,H,C))
            if imNum % plotCols == 0:
                i = i + 1
                j = 0
                if (cols == []) and (imNum != 0):
                    cols = rows
                elif cols != []:
                    print('cols vstack', cols.shape, rows.shape)
                    cols = np.vstack((cols, rows))
            else:
                if j == 0:
                    rows = image
                else:
                    rows = np.hstack((rows,image))
                j = j + 1
                print('rows', rows.shape, 'image', image.shape)
            # axes[i, j].imshow(image)
        plt.imshow(cols)
        plt.show()

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
if __name__ == '__main__':
    imagePath = '/home/loxor/Documents/Humpback_Whale_Identification/data/some_imgs_from_train/'
    image = PIL.Image.open(os.path.join(imagePath, '0c5537f22.jpg'))
    transformedImages, _ = AugmentImage.Patches_of_Images()(image)
    VisualizeImage(transformedImages)