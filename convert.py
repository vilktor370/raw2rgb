from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

class RawConvert(object):
    def __init__(self):
        self.wb_multipliers = [2.217041, 1.000000, 1.192484]
        
    def toRGB(self, tiffFilePath:str, outputFileName='convert_rgb.png',isSaved=True):
        """
        If you don't want to go through the conversion step by step. 
        This function takes cares of all the work for you.
        
        Args:
            tiffFilePath (str): input raw image file path/ .tiff file path
            outputFileName (str): specify output file name. Defaults to "convert_rgb.png"
            isSaved (bool): save the image file or not. Defaults to True

        Raises:
            ValueError: _description_
        """
        raw_data = Image.open(tiffFilePath)
        raw = np.array(raw_data).astype(np.float64)
        # step 1: normalize an image
        normalize_img = self.normalize(raw)
        
        # step 2: adjust the white balance
        balanced_img = self.whiteBalance(normalize_img)
        
        # step 3: demosaic the img
        bayer_img = self.demosaic(balanced_img)
        
        # step 4: color correction
        rgb_img = self.colorCorrection(bayer_img)
        
        # step 5: (Optinal) adjust brightness
        polised_img = self.brightnessCorrection(rgb_img, 2.22)
        
        if isSaved:
            plt.imsave(outputFileName, polised_img)
        print("SUCCESS!")
        
    
    def normalize(self, rawImg: np.ndarray) -> np.ndarray:
        """
        Step1: Normalize an Image

        Args:
            rawImg (np.ndarray): numpy array that contains raw images

        Returns:
            np.ndarray: normalized numpy array that contains raw images
        """
        data_min = np.min(rawImg)
        data_max = np.max(rawImg)
        linear_bayer = (rawImg - data_min) / (data_max - data_min)
        return linear_bayer
    
    def whiteBalance(self, normRawImg: np.ndarray, align='rggb') -> np.ndarray:
        """
        Step2: Apply white balance to a normalized image

        Args:
            normRawImg (np.ndarray): normalized raw image numpy array
            align (str, optional): Defaults to 'rggb'.
            'rggb':
                R G
                G B
            'gbrg':
                G B
                R G

        Returns:
            np.ndarray: image array with white balancing
        """
        mask = self.wb_multipliers[1] * np.ones((normRawImg.shape[0], normRawImg.shape[1]))  # Initialize to all green values

        # Fill in the scales for the red and blue pixels across the matrix
        if (align == 'rggb'):
            mask[0::2, 0::2] = self.wb_multipliers[0]  # r
            mask[1::2, 1::2] = self.wb_multipliers[2]  # b
        elif (align == 'bggr'):
            mask[1::2, 1::2] = self.wb_multipliers[0]  # r
            mask[0::2, 0::2] = self.wb_multipliers[2]  # b
        elif (align == 'grbg'):
            mask[0::2, 1::2] = self.wb_multipliers[0]  # r
            mask[0::2, 1::2] = self.wb_multipliers[2]  # b
        elif (align == 'gbrg'):
            mask[1::2, 0::2] = self.wb_multipliers[0]  # r
            mask[0::2, 1::2] = self.wb_multipliers[2]  # b
        balanced_bayer = np.multiply(normRawImg, mask)
        return balanced_bayer
    
    def demosaic(self, rawImg) -> np.ndarray:
        """
        Step3: Bilinear Interpolation of the missing pixels
        
        Assumes a Bayer CFA in the 'rggb' layout
          R G R G
          G B G B
          R G R G
          G B G B

        Args:
            rawImg (_type_): HxWx1 channel image array

        Returns:
            np.ndarray: HxWx3 RGB color image
        """

        #
        # Input: Single-channel rggb Bayered image
        # Returns: A debayered 3-channels RGB image
        #
        img = rawImg.astype(np.float64)

        m = img.shape[0]
        n = img.shape[1]

        # First, we're going to create indicator masks that tell us
        # where each of the color pixels are in the bayered input image
        # 1 indicates presence of that color, 0 otherwise
        red_mask = np.tile([[1, 0], [0, 0]], (int(m / 2), int(n / 2)))

        # TODO: Complete the following two lines to generate
        # indicator masks for the green and blue channels
        #
        green_mask = np.tile([[0, 1], [1, 0]], (int(m / 2), int(n / 2)))
        blue_mask = np.tile([[0, 0], [0, 1]], (int(m / 2), int(n / 2)))

        r = np.multiply(img, red_mask)
        g = np.multiply(img, green_mask)
        b = np.multiply(img, blue_mask)

        # Confirm for yourself:
        # - What are the patterns of values in the r,g,b images?
        # Sketch them out to help yourself.

        # Next, we're going to fill in the missing values in r,g,b
        # For this, we're going to use filtering - convolution - to implement bilinear interpolation.
        # - We know that convolution allows us to perform a weighted sum
        # - We know _where_ our pixels lie within a grid, and where the missing pixels are
        # - We know filters come in odd sizes

        # Interpolating green:
        filter_g = 0.25 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        missing_g = convolve2d(g, filter_g, 'same')
        g = g + missing_g

        # See how it works?
        # The filter only produces output if the surrounding pixels match its pattern.
        # When they do, it produces their mean value.

        # Note that we're going to have some incorrect values at the image boundaries,
        # but let's ignore that for this exercise.

        # Now, let's try it for blue. This one is a two-step process.
        # - Step 1: We fill in the 'central' blue pixel in the location of the red pixel
        # - Step 2: We fill in the blue pixels at the locations of the green pixels,
        #           similar to how the green interpolation worked, but offset by a row/column
        #
        # Sketch out the matrices to help you follow.
        # Remember, we'll still have some incorrect value at the image boundaries.

        # Interpolating blue:
        # Step 1:
        filter1 = 0.25 * np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        missing_b1 = convolve2d(b, filter1, 'same')
        # Step 2:
        filter2 = 0.25 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        missing_b2 = convolve2d(b + missing_b1, filter2, 'same')
        b = b + missing_b1 + missing_b2

        # OK! Only red left.

        # Interpolation for the red at the missing points
        filter_r1 = 0.25 * np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
        missing_r1 = convolve2d(r, filter_r1, 'same')
        filter_r2 = 0.25 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        missing_r2 = convolve2d(r + missing_r1, filter_r2, 'same')
        r = r + missing_r1 + missing_r2

        output = np.stack((r, g, b), axis=2)
        return output
    
    def colorCorrection(self, rawImg:np.ndarray) -> np.ndarray:
        """
        Final Step: Color Correction(fine tunning the RGB color)

        Args:
            rawImg (np.ndarray): image array with normalized, white balancing and demosicing

        Returns:
            np.ndarray: Final polished image in sRGB color space.
        """
        rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
        xyz2cam = np.array([[0.6653, -0.1486, -0.0611],
                            [-0.4221, 1.3303, 0.0929],
                            [-0.0881, 0.2416, 0.7226]])
        rgb2cam = xyz2cam * rgb2xyz  # Assuming previously defined matrices
        denom = np.tile(np.reshape(np.sum(rgb2cam, axis=1), (3, -1)), (1, 3))
        rgb2cam = np.divide(rgb2cam, denom)  # Normalize rows to 1
        cam2rgb = np.linalg.inv(rgb2cam)
        
        # Apply camera matrix
        r = cam2rgb[0, 0] * rawImg[:, :, 0] + cam2rgb[0, 1] * rawImg[:, :, 1] + cam2rgb[0, 2] * rawImg[:, :, 2]
        g = cam2rgb[1, 0] * rawImg[:, :, 0] + cam2rgb[1, 1] * rawImg[:, :, 1] + cam2rgb[1, 2] * rawImg[:, :, 2]
        b = cam2rgb[2, 0] * rawImg[:, :, 0] + cam2rgb[2, 1] * rawImg[:, :, 1] + cam2rgb[2, 2] * rawImg[:, :, 2]
        lin_srgb = np.stack((r, g, b), axis=2)
        lin_srgb[lin_srgb > 1.0] = 1.0  # Always keep image clipped b/w 0-1
        lin_srgb[lin_srgb < 0.0] = 0.0
        return lin_srgb
    
    def brightnessCorrection(self, img:np.ndarray, gamma=2.2) -> np.ndarray:
        """
        Optinal: Gamma correction(adjust the brightness of an image).
        Only necessary if the image is too dark or bright

        Args:
            img (np.ndarray): polished image
            gamma (float): brightness value. Usually 2.2 for most cameras.

        Returns:
            np.ndarray: brigh images
        """
            # gamma correction
        nl_srgb = img ** (1 / gamma)
        return nl_srgb