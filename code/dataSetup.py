import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import pywt
import time
import DWT as dwt
#%%
img = cv2.imread('data/messi5.jpg', 0)
cman = cv2.imread("data/cman.png", 0)
cman = cv2.resize(cman, (512, 512)) 
cv2.imwrite('data/cman_512_512.png',cman)
N_rows, N_cols = cman.shape
#%%Noise
#Gaussian
def genNoisy():
    mu = 0
    std = 10
    cman_gnoise = np.copy(cman)
    gauss_noise = np.uint8(np.random.normal(loc=mu, scale=std,size=(N_rows, N_cols)))
    std_gauss   = np.std(gauss_noise)
    cman_gnoise = cman + gauss_noise
    cman_gnoise = np.uint8(np.clip(cman_gnoise, 0, 255))
    #Salt and Pepper
    s_vs_p = 0.5
    amount = 0.004
    cman_spnoise = np.copy(cman)
    num_salt_pepper = np.ceil(amount * cman.size * s_vs_p)
    s_coords = [np.random.randint(0, i - 1, int(num_salt_pepper)) for i in cman.shape]
    p_coords = [np.random.randint(0, i - 1, int(num_salt_pepper)) for i in cman.shape]
    cman_spnoise[s_coords] = 255
    cman_spnoise[p_coords] = 0
    return(cman_gnoise, cman_spnoise, std)
#%%
def pltImgs(cman_gnoise, cman_HWT_hard, cman_HWT_soft, cman_IHWT_hard, cman_IHWT_soft):
    plt.figure()
    plt.title('Cman_gnoise')
    plt.imshow(cman_gnoise, cmap='gray')
    
    plt.figure()
    plt.title('HWT_hard')
    plt.imshow(cman_HWT_hard, cmap='gray')
    
    plt.figure()
    plt.title('IHWT_hard')
    plt.imshow(cman_IHWT_hard, cmap='gray')
    
    plt.figure()
    plt.title('HWT_soft')
    plt.imshow(cman_HWT_soft, cmap='gray')
    
    plt.figure()
    plt.title('IHWT_soft')
    plt.imshow(cman_IHWT_soft, cmap='gray')
    
#%%
def pltFilters(conv, blur, gblur, median, bilateral):
    plt.figure()
    plt.title('Conv')
    plt.imshow(conv, cmap='gray')
    
    plt.figure()
    plt.title('blur')
    plt.imshow(blur, cmap='gray')
    
    plt.figure()
    plt.title('Gaussian Blur')
    plt.imshow(gblur, cmap='gray')
    
    plt.figure()
    plt.title('median')
    plt.imshow(median, cmap='gray')
    
    plt.figure()
    plt.title('bilateral')
    plt.imshow(bilateral, cmap='gray')
    
#%%
#Peak Signal to Noise Ratio
def peakSNR(g_out, sp_out, cman_gnoise, cman_spnoise):
    psnr_frame = pd.DataFrame(np.zeros((len(g_out), len(list(g_out.keys())[0]))))
    print('Gaussian Noise')
    for filter in g_out:
        mse = mean_squared_error(g_out[filter],cman_gnoise)    
        psnr = 10*np.log10((g_out[filter].max()**2)/mse)
        print(filter, ': ', psnr)
        
    print('\n', 'Salt and Pepper Noise')
    for filter in sp_out:
        mse = mean_squared_error(sp_out[filter],cman_spnoise)    
        psnr = 10*np.log10((sp_out[filter].max()**2)/mse)
        print(filter, ': ', psnr)
    
#%%Filtering
def standardFilters(img):
    kernel = np.ones((5,5),np.float32)/25
    conv = cv2.filter2D(img,-1,kernel)    
    blur = cv2.blur(img,(5,5))
    gblur = cv2.GaussianBlur(img,(5,5),0)
    median = cv2.medianBlur(img,5)
    bilateral = cv2.bilateralFilter(img,9,75,75)
    return(conv, blur, gblur, median, bilateral)
 
#%%Thresholding
#    pywt.threshold(data, 2, 'hard')
#%%
s_time = time.clock()
cman_gnoise, cman_spnoise, std = genNoisy()
iterations = 1
g_out = {'conv' : np.zeros(0), 'blur' : np.zeros(0), 'gblur' : np.zeros(0),
           'median' : np.zeros(0), 'bilateral' : np.zeros(0), 'IHWT_soft' : np.zeros(0), 'IHWT_hard' : np.zeros(0)}
           
sp_out = {'conv' : np.zeros(0), 'blur' : np.zeros(0), 'gblur' : np.zeros(0),
           'median' : np.zeros(0), 'bilateral' : np.zeros(0), 'IHWT_soft' : np.zeros(0), 'IHWT_hard' : np.zeros(0)}


#Gaussian Noise Haar Wavelet Transform, Thresholding and Filtering
HWT_g = dwt.TwoD_HWT(cman_gnoise,iterations)
HWT_hard_g = pywt.threshold(HWT_g, 3*std, 'hard')
HWT_soft_g = pywt.threshold(HWT_g, 3*std, 'soft')
g_out['IHWT_hard'] = dwt.TwoD_IHWT(HWT_hard_g,iterations)
g_out['IHWT_soft']= dwt.TwoD_IHWT(HWT_soft_g,iterations)

#Gaussian Noise Haar Wavelet Transform, Thresholding, and Filtering
HWT_sp = dwt.TwoD_HWT(cman_spnoise,iterations)
HWT_hard_sp = pywt.threshold(HWT_sp, 3*std, 'hard')
HWT_soft_sp = pywt.threshold(HWT_sp, 3*std, 'soft')
sp_out['IHWT_hard'] = dwt.TwoD_IHWT(HWT_hard_sp,iterations)
sp_out['IHWT_soft']= dwt.TwoD_IHWT(HWT_soft_sp,iterations)


g_out['conv'], g_out['blur'], g_out['gblur'], g_out['median'], g_out['bilateral'] = standardFilters(cman_gnoise)
sp_out['conv'], sp_out['blur'], sp_out['gblur'], sp_out['median'], sp_out['bilateral'] = standardFilters(cman_spnoise)

peakSNR(g_out, sp_out, cman_gnoise, cman_spnoise)
#pltFilters(conv, blur, gblur, median, bilateral)
#pltImgs(cman_gnoise, cman_HWT_hard, cman_HWT_soft, cman_IHWT_hard, cman_IHWT_soft)

e_time = time.clock()
t_time = e_time - s_time
print('Total_Time: ', t_time) 


#%%