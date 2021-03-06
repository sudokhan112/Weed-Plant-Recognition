import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas

def plot_histogram(image, title, mask = None):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color = color)
        plt.xlim([0, 256])

    plt.show()


def plot_histogram_green(image, title, mask = None):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip(chans, colors):
        # for full range [256], everything was at white pixel value (which is a 256) (because most of the image is white for "masked_replace_white") .
        # Reduced the range to [255], so that we can see the histogram of the green pixels.
        hist = cv2.calcHist([chan], [0], mask, [255], [0, 255])
        plt.plot(hist, color = color)
        plt.xlim([0, 255])

    plt.show()




def otsu_thresh(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    #cv2.imshow("Image gray", image)

    T = mahotas.thresholding.otsu(blurred)
    print('Otsu’s threshold: %d' % T)

    thresh = image.copy()
    thresh[thresh > T] = 255
    thresh[thresh < T] = 0
    thresh = cv2.bitwise_not(thresh)
    cv2.imshow("Otsu", thresh)

def rid_cav_thresh(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    T = mahotas.thresholding.rc(blurred)
    print ('Riddler-Calvard: %d' % T)
    thresh = image.copy()
    thresh[thresh > T] = 255
    thresh[thresh < 255] = 0
    thresh = cv2.bitwise_not(thresh)
    cv2.imshow("Riddler-Calvard", thresh)

def combined (img):
    #b, g, r = cv2.split(img)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    #r_max = g_max = b_max = 255
    r_max = np.amax(r)
    g_max = np.amax(g)
    b_max = np.amax(b)
    #print (r)
    #cv2.imshow("red chaneel",b)

    red_norm = r/r_max
    green_norm = g/g_max
    blue_norm = b/b_max

   # print(red_norm.shape)

    norm = red_norm + blue_norm + green_norm


    small_num = 0.0001
    r = red_norm/(norm+small_num)
    g = green_norm/(norm+small_num)
    b = blue_norm/(norm+small_num)

    #print('normalized r g b values: %d %d %d' %(r, g, b))

    ExG = 2*g - r - b #excess green

    ExGR = ExG -1.4*r - g #excess green minus red

    CIVE = -(0.441*r - 0.811*g + 0.385*b + 18.78745) #color index of vegetation extraction


    #redistribute the weights without VEG
    w_ExG = 0.28
    w_ExGR = 0.34
    w_CIVE = 0.38

    combined = w_ExG * ExG + w_ExGR * ExGR + w_CIVE * CIVE

    return combined

def linear_map (image): ##combined image 'combined', is linearly mapped to range in [0, 255], after which, it is thresholded by applying the Otsu’s
    max_value = np.max(combined(image))
    min_value = np.min(combined(image))
    #print(min_value, max_value)

    #mapped combined image value (which is mostly negative) to 0-255
    new_min = 0
    new_max = 255
    old_range = max_value - min_value
    new_range = new_max - new_min
    lin_map = (((combined(image).astype(np.float64) - min_value) * new_range) / old_range) + new_min
    image_map = lin_map.astype(np.uint8)
    #cv2.imshow("Green Extracted Image", image_map)
    return image_map


#clahe only applicable to gray image. applying clahe to LAB format
# then change to bgr for other functions (histogram, combined: which takes bgr input) to work
#then change back to bgr again
cliplimit = 0


def clahe_bgr(image, cliplimit, gridsize = 8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(cliplimit, tileGridSize=(gridsize, gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

#img = cv2.imread('/home/sayem/Desktop/cs557_project/Final_project/low_quality_image/coc01.jpg')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/Final_project/brightness_control/val_rag_01_middark.jpg')
img = cv2.imread('/home/sayem/Desktop/cs557_project/Final_project/rag053.jpg')
#cv2.imshow("Image", img)
rows, cols, _ = img.shape
print(img.shape)


#showing otsu threshold of green extracted image
T = mahotas.thresholding.otsu(linear_map(img)) #input original image
#T = mahotas.thresholding.otsu(linear_map(clahe_im)) #input CLAHE image
print ('Otsu’s threshold combined: %d' % T)
thresh_com = linear_map(img).copy()
thresh_com[thresh_com > T] = 0
thresh_com[thresh_com > 0] = 255
thresh_com_mask= cv2.bitwise_not(thresh_com)
#cv2.imshow("Otsu Green Extracted", thresh_com_mask) #showing otsu threshold of green extracted image


# convert single channel mask back into 3 channels
mask_rgb = cv2.cvtColor(thresh_com_mask, cv2.COLOR_GRAY2RGB)

# perform bitwise and on mask to obtain cut-out image that is not green
masked_img = cv2.bitwise_and(img, mask_rgb)

# replace the cut-out parts (black) with white
masked_replace_white = cv2.addWeighted(masked_img, 1, cv2.cvtColor(thresh_com, cv2.COLOR_GRAY2RGB), 1, 0)

plt.imshow(cv2.cvtColor(masked_replace_white, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

#plot_histogram(img, "Histogram for Original Image")
plot_histogram_green(masked_replace_white, "Histogram of extracted green image")



'''where m is the mean value of a, i.e. the average gray level
m = summation (a . pa)

The mean determines the average level of brightness, where low, high and
medium values indicate the degree of light which has impacted
the device.

'''

""" Let a be a random variable denoting gray levels, the nth moment of a about the mean is
defined as (Gonzalez & Woods, 2008):
meu_n(a) = summation ((a-m)^n * p(a))

Variance is a measure of gray-level contrast, where high values indicate dispersion of values around
the mean and low values are indicative of a high concentration
of values around the mean. 

"""

"""The skewness measures the asymmetry in the distribution. A right skewness is presented when the histo-
gram displays a large tail oriented towards high brightness values
and high concentration in the part of low brightness values (posi-
tive skewness). In the opposite case the skewness is negative.

skew = mu_3/(mu_2)^1.5
"""

"""kurtosis provides information about the peakedness in the distri-
bution; low kurtosis indicates flat top parts in the histogram
around the mean but high values are indicative of peaks around
the mean with high slopes and large tails. Skewness and kurtosis
are both zero for Gaussian distributions.

kurtosis = mu4/mu2^2
"""
"""An image with sufficient contrast should be identified by mean val-
ues in the central part of histogram, high variance, low skewness
(positive or negative) and high kurtosis. On the contrary, an image
with insufficient contrast is identified by mean values either low or
high, high skewness (positive or negative) and low kurtosis.


"""
# input of the stat parameters are "masked_replace_white". The hypothesis is, as we are calculating stat parameters
#only on green pixels, how light affects the weeds can be determined from the stat parameters. stat parameters can be input of
# adaptive CLAHE. This way contrast of the image is only changed according to the effect of the light on weeds. Other Objects in the image
# will not be increased randomly (noise). Even if noise increases, it will affect the weed classification.
def stat_blue (img):
    # changed range to [255] to exclude the white pixels
    hist_blue = cv2.calcHist([img], [0], None, [255], [0, 255])
    rows, cols, _ = img.shape
    probability_pa = hist_blue / (rows * cols)
    a = np.arange(256)

    #mean
    m = np.sum(hist_blue * probability_pa)
    #m = np.sum(hist_blue/256)
    print("Blue channel mean", m)

    # 1st moment
    mu_1 = np.sum((hist_blue - m) * probability_pa)
    #print("1st moment mu1", mu_1)
    # 2nd moment
    mu_2 = np.sum(np.power((hist_blue - m), 2) * probability_pa)
    print("Blue channel 2nd moment mu2 (Variance)", mu_2)
    # 3rd moment
    mu_3 = np.sum(np.power((hist_blue - m), 3) * probability_pa)
    # 4th moment
    mu_4 = np.sum(np.power((hist_blue - m), 4) * probability_pa)

    skewness = mu_3 / (np.power(mu_2, 1.5))
    #print("skewness", skewness)
    print("Blue channel abs. skewness", np.absolute(skewness))

    kurtosis = mu_4 / (np.power(mu_2, 2))
    print("Blue channel Kurtosis", kurtosis)


def stat_green(img):
    # changed range to [255] to exclude the white pixels
    hist_green = cv2.calcHist([img], [1], None, [255], [0, 255])
    rows, cols, _ = img.shape
    probability_pa = hist_green / (rows * cols)
    a = np.arange(256)

    # mean
    m = np.sum(hist_green * probability_pa)
    #m = np.sum(hist_green / 256)
    print("Green channel mean", m)

    # 1st moment
    mu_1 = np.sum((hist_green - m) * probability_pa)
    # print("1st moment mu1", mu_1)
    # 2nd moment
    mu_2 = np.sum(np.power((hist_green - m), 2) * probability_pa)
    print("Green channel 2nd moment mu2 (Variance)", mu_2)
    # 3rd moment
    mu_3 = np.sum(np.power((hist_green - m), 3) * probability_pa)
    # 4th moment
    mu_4 = np.sum(np.power((hist_green - m), 4) * probability_pa)

    skewness = mu_3 / (np.power(mu_2, 1.5))
    # print("skewness", skewness)
    print("Green channel abs. skewness", np.absolute(skewness))

    kurtosis = mu_4 / (np.power(mu_2, 2))
    print("Green channel Kurtosis", kurtosis)



def stat_red(img):
    # changed range to [255] to exclude the white pixels
    hist_red = cv2.calcHist([img], [2], None, [255], [0, 255])
    rows, cols, _ = img.shape
    probability_pa = hist_red / (rows * cols)
    a = np.arange(256)

    # mean
    m = np.sum(hist_red * probability_pa)
    #m = np.sum(hist_red / 256)
    print("Red channel mean", m)

    # 1st moment
    mu_1 = np.sum((hist_red - m) * probability_pa)
    # print("1st moment mu1", mu_1)
    # 2nd moment
    mu_2 = np.sum(np.power((hist_red - m), 2) * probability_pa)
    print("Red channel 2nd moment mu2 (Variance)", mu_2)
    # 3rd moment
    mu_3 = np.sum(np.power((hist_red - m), 3) * probability_pa)
    # 4th moment
    mu_4 = np.sum(np.power((hist_red - m), 4) * probability_pa)

    skewness = mu_3 / (np.power(mu_2, 1.5))
    # print("skewness", skewness)
    print("Red channel abs. skewness", np.absolute(skewness))

    kurtosis = mu_4 / (np.power(mu_2, 2))
    print("Red channel Kurtosis", kurtosis)


clahe_img = clahe_bgr(img, cliplimit, gridsize=8)
#cv2.imshow("Clahe applied image",clahe_img)

numpy_horizontal = np.hstack((img, clahe_img))
numpy_horizontal_concat = np.concatenate((img, clahe_img), axis=1)
image_resized = cv2.resize(numpy_horizontal_concat, (0, 0), None, .25, .25)
cv2.imshow('Original (LHS) & CLAHE (RHS)', image_resized)

plot_histogram(clahe_img, "Clahe histogram")


#chech how application of CLAHE changed image stat
print("Green masked image stat:")

stat_green(masked_replace_white) #showing results of the green channel for green segmented image without the white pixel results

print("CLAHE modified image stat:")

stat_green(clahe_img) #stat of green channel for clahe applied image

"""
For the following input image:
img = cv2.imread('/home/sayem/Desktop/cs557_project/Final_project/brightness_control/val_rag_01_midbright.jpg')

cliplimit = 18 at CLAHE made the histogram more gaussian (by looking). Calculate the stat parameters for "clahe_img"
to see skewness and kurtosis. Skewness and kurtosis are both zero for Gaussian distributions. Goal is to go towards
gaussian distribution.
"""

cv2.waitKey(0)