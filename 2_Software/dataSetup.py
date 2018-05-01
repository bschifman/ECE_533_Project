import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import csv
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
#    cman_gnoise = np.uint8(np.clip(cman_gnoise, 0, 255))
    cman_gnoise = np.uint8(cman_gnoise)
    #Salt and Pepper
    s_vs_p = 0.5
    amount = 0.008
    cman_spnoise = np.copy(cman)
    num_salt_pepper = np.ceil(amount * cman.size * s_vs_p)
    s_coords = [np.random.randint(0, i - 1, int(num_salt_pepper)) for i in cman.shape]
    p_coords = [np.random.randint(0, i - 1, int(num_salt_pepper)) for i in cman.shape]
    cman_spnoise[s_coords] = 255
    cman_spnoise[p_coords] = 0
    return(cman_gnoise, cman_spnoise, std)

    
#%%
def pltFilters(g_out, sp_out):
    for filter in g_out:
        str_title = filter + ' Gaussian'
        plt.figure()
        plt.title(str_title)
        plt.imshow(g_out[filter], cmap='gray')
        
    for filter in sp_out:
        str_title = filter + ' S&P'
        plt.figure()
        plt.title(str_title)
        plt.imshow(sp_out[filter], cmap='gray')   

#%%
def detFilterLength(img_gnoise, img_spnoise, img):
    
    g_out = {'blur' : np.zeros(0), 'gblur' : np.zeros(0),
           'median' : np.zeros(0), 'bilateral' : np.zeros(0)}
    
    sp_out = {'blur' : np.zeros(0), 'gblur' : np.zeros(0),
           'median' : np.zeros(0), 'bilateral' : np.zeros(0)}
    
    filter_length_temp = np.ones((len(g_out), 2), dtype=int)
    filter_length_best = np.ones((len(g_out), 2), dtype=int)
    psnr_best = np.zeros((len(g_out), 2))
    
    def detFilterPSNR(g_out, sp_out, img):
        psnr_curr = np.zeros((len(g_out),2))
        i = 0
        for filter in g_out:
            mse = mean_squared_error(img, g_out[filter])    
            psnr_curr[i,0] = 10*np.log10((g_out[filter].max()**2)/mse)
            i += 1
            
        i = 0
        for filter in sp_out:
            mse = mean_squared_error(img, sp_out[filter])    
            psnr_curr[i,1] = 10*np.log10((sp_out[filter].max()**2)/mse)
            i += 1    
    
        return(psnr_curr)
        
    i = 0 
    psnr_hist = []
    while(i < 5):
        g_out['blur'], g_out['gblur'], g_out['median'], g_out['bilateral'] = standardFilters(img_gnoise,  filter_length_temp, 0)
        sp_out['blur'], sp_out['gblur'], sp_out['median'], sp_out['bilateral'] = standardFilters(img_spnoise, filter_length_temp, 0)
        psnr_curr = detFilterPSNR(g_out, sp_out, img)
        psnr_hist.append(psnr_curr)
        for j in range(len(filter_length_temp)):
            if(psnr_curr[j,0] > psnr_best[j,0]):
                psnr_best[j,0] = psnr_curr[j,0]
                filter_length_best[j,0] = filter_length_temp[j,0]
                
            if(psnr_curr[j,1] > psnr_best[j,1]):
                psnr_best[j,1] = psnr_curr[j,1]
                filter_length_best[j,1] = filter_length_temp[j,1]
                
        filter_length_temp = np.add(filter_length_temp, 2)
        i += 1
        
    return(filter_length_best, psnr_best, psnr_hist)
#%%
#Peak Signal to Noise Ratio
def peakSNR(g_out, sp_out, img):
    psnr_output = pd.DataFrame(pd.np.zeros((len(g_out), 2)))
    i = 0
    for filter in g_out:
        mse = mean_squared_error(img, g_out[filter])    
        psnr_output.iloc[i,0] = np.round(10*np.log10((g_out[filter].max()**2)/mse), 3)
        i += 1
        
    i = 0        
    for filter in sp_out:
        mse = mean_squared_error(img, sp_out[filter])    
        psnr_output.iloc[i,1] = np.round(10*np.log10((sp_out[filter].max()**2)/mse), 3)
        psnr_output.rename({i: str(filter)}, axis='index')
        i += 1
    
    print(psnr_output)
    psnr_output.to_csv('psnr.csv', header=False)
    
    
#%%
def detThreshold(HWT_g, HWT_sp, DB4T_g, DB4T_sp, std, img):
    N = 20
    psnr = np.zeros((N,8))
    T = np.zeros(8)
    
    def detPeakSNR(iHWTg_soft, iHWTg_hard, iHWTsp_soft, iHWTsp_hard, iDB4Tg_soft, iDB4Tg_hard, iDB4Tsp_soft, iDB4Tsp_hard,
                   img, psnr, i):
        #HAAR WAVELETS
        mse = mean_squared_error(img, iHWTg_soft)    
        psnr[i,0] = 10*np.log10((iHWTg_soft.max()**2)/mse)
        
        mse = mean_squared_error(img, iHWTg_hard)    
        psnr[i,1] = 10*np.log10((iHWTg_hard.max()**2)/mse)
        
        mse = mean_squared_error(img, iHWTsp_soft)    
        psnr[i,2] = 10*np.log10((iHWTsp_soft.max()**2)/mse)
        
        mse = mean_squared_error(img, iHWTsp_hard)    
        psnr[i,3] = 10*np.log10((iHWTsp_hard.max()**2)/mse)
        
# =============================================================================
        #DB4 WAVELETS
        mse = mean_squared_error(img, iDB4Tg_soft)    
        psnr[i,4] = 10*np.log10((iDB4Tg_soft.max()**2)/mse)
         
        mse = mean_squared_error(img, iDB4Tg_hard)    
        psnr[i,5] = 10*np.log10((iDB4Tg_hard.max()**2)/mse)
         
        mse = mean_squared_error(img, iDB4Tsp_soft)    
        psnr[i,6] = 10*np.log10((iDB4Tsp_soft.max()**2)/mse)
         
        mse = mean_squared_error(img, iDB4Tsp_hard)    
        psnr[i,7] = 10*np.log10((iDB4Tsp_hard.max()**2)/mse)

        return (psnr)
    
    i = 1
    while(i < N):
        #HAAR WAVELETS
        HWT_g_hard = pywt.threshold(HWT_g, 0.5*i*std, 'hard')
        HWT_g_soft = pywt.threshold(HWT_g, 0.5*i*std, 'soft')
            
        iHWTg_hard = dwt.TwoD_IHWT(HWT_g_hard, 1)
        iHWTg_soft = dwt.TwoD_IHWT(HWT_g_soft, 1)    

        HWT_sp_hard = pywt.threshold(HWT_sp, 0.5*i*std, 'hard')
        HWT_sp_soft = pywt.threshold(HWT_sp, 0.5*i*std, 'soft')
        
        iHWTsp_hard = dwt.TwoD_IHWT(HWT_sp_hard, 1)
        iHWTsp_soft = dwt.TwoD_IHWT(HWT_sp_soft, 1)
# =============================================================================        
        #DB4 WAVELETS        
        temp1 = np.array(pywt.threshold(DB4T_g[0], 0.5*i*std, 'soft'))
        temp2 = np.array(pywt.threshold(DB4T_g[1], 0.5*i*std, 'soft'))
        
        temp3 = np.array(pywt.threshold(DB4T_sp[0], 0.5*i*std, 'soft'))
        temp4 = np.array(pywt.threshold(DB4T_sp[1], 0.5*i*std, 'soft'))
        
        iDB4Tg_soft = pywt.idwt2((temp1,(temp2)),'db4')
        iDB4Tsp_soft = pywt.idwt2((temp3,(temp4)),'db4')
        
        temp1 = np.array(pywt.threshold(DB4T_g[0], 0.5*i*std, 'hard'))
        temp2 = np.array(pywt.threshold(DB4T_g[1], 0.5*i*std, 'hard'))
        
        temp3 = np.array(pywt.threshold(DB4T_sp[0], 0.5*i*std, 'hard'))
        temp4 = np.array(pywt.threshold(DB4T_sp[1], 0.5*i*std, 'hard'))
        
        iDB4Tg_hard = pywt.idwt2((temp1,(temp2)),'db4')
        iDB4Tsp_hard = pywt.idwt2((temp3,(temp4)),'db4')           
        
        psnr = detPeakSNR(iHWTg_soft, iHWTg_hard, iHWTsp_soft, iHWTsp_hard, iDB4Tg_soft, iDB4Tg_hard, iDB4Tsp_soft, iDB4Tsp_hard,
                   img, psnr, i)
        i += 1
        
    psnr[0,:] = np.nan
    for i in range(psnr.shape[1]):
        T[i] = np.nanargmax(psnr[:,i])
        
    return(psnr, T)
#%%
def pltDetThreshold(psnr):
    plt.figure()
    plt.title('Haar PSNR vs Threshold for Gaussian Noise')
    plt.ylabel('PSNR')
    plt.xlabel('Threshold*0.5*$\sigma$')
    plt.plot(psnr[:,0], color='green', label='Soft Threshold')
    plt.plot(psnr[:,1], color='black', label='Hard Threshold')
    plt.legend()
    
    plt.figure()
    plt.title('Haar PSNR vs Threshold for S&P Noise')
    plt.ylabel('PSNR')
    plt.xlabel('Threshold*0.5*$\sigma$')
    plt.plot(psnr[:,2], color='green', label='Soft Threshold')
    plt.plot(psnr[:,3], color='black', label='Hard Threshold')
    plt.legend()
    
    plt.figure()
    plt.title('DB4 PSNR vs Threshold for Gaussian Noise')
    plt.ylabel('PSNR')
    plt.xlabel('Threshold*0.5*$\sigma$')
    plt.plot(psnr[:,4], color='green', label='Soft Threshold')
    plt.plot(psnr[:,5], color='black', label='Hard Threshold')
    plt.legend()
    
    plt.figure()
    plt.title('DB4 PSNR vs Threshold for S&P Noise')
    plt.ylabel('PSNR')
    plt.xlabel('Threshold*0.5*$\sigma$')
    plt.plot(psnr[:,6], color='green', label='Soft Threshold')
    plt.plot(psnr[:,7], color='black', label='Hard Threshold')
    plt.legend()

#%%
def pltPSNRFilters(psnr_hist):
    blur_psnr_g       = np.zeros(len(psnr_hist))
    gblur_psnr_g      = np.zeros(len(psnr_hist))
    median_psnr_g     = np.zeros(len(psnr_hist))
    bilateral_psnr_g  = np.zeros(len(psnr_hist))
    blur_psnr_sp      = np.zeros(len(psnr_hist))
    gblur_psnr_sp     = np.zeros(len(psnr_hist))
    median_psnr_sp    = np.zeros(len(psnr_hist))
    bilateral_psnr_sp = np.zeros(len(psnr_hist))
    xaxis             = np.zeros(len(psnr_hist))
    k = 1
    i = 0
    while(i < len(psnr_hist)):
        j = 0   
        blur_psnr_g[i]       = psnr_hist[i][0,j]
        gblur_psnr_g[i]      = psnr_hist[i][1,j]
        median_psnr_g[i]     = psnr_hist[i][2,j]
        bilateral_psnr_g[i]  = psnr_hist[i][3,j]
        j += 1
        blur_psnr_sp[i]      = psnr_hist[i][0,j]
        gblur_psnr_sp[i]     = psnr_hist[i][1,j]
        median_psnr_sp[i]    = psnr_hist[i][2,j]
        bilateral_psnr_sp[i] = psnr_hist[i][3,j] 
        xaxis[i]             = k
        k += 2
        i += 1

    
    plt.figure()
    plt.title('PSNR vs Filter Length w/ Gaussian Noise')
    plt.xlabel('Filter Length')
    plt.ylabel('PSNR (dB)')
    plt.plot(range(1,11,2), blur_psnr_g,      label='blur'     )
    plt.plot(range(1,11,2), gblur_psnr_g,     label='gblur'    )
    plt.plot(range(1,11,2), median_psnr_g,    label='median'   )
    plt.plot(range(1,11,2), bilateral_psnr_g, label='bilateral')
    plt.legend()
    
    plt.figure()
    plt.title('PSNR vs Filter Length w/ S&P Noise')
    plt.xlabel('Filter Length')
    plt.xlim(0,5)
    plt.xticks(xaxis)
    plt.ylim(30, 52)
    plt.ylabel('PSNR (dB)')
    plt.plot(range(1,11,2), blur_psnr_sp,     label='blur'     )
    plt.plot(range(1,11,2), gblur_psnr_sp,    label='gblur'    )
    plt.plot(range(1,11,2), median_psnr_sp,   label='median'   )
    plt.plot(range(1,11,2), bilateral_psnr_sp,label='bilateral')
    plt.legend()
    
#%%Filtering
def standardFilters(img, f_length, j):   
    blur = cv2.blur(img,(f_length[0,j],f_length[0,j]))
    gblur = cv2.GaussianBlur(img,(f_length[1,j],f_length[1,j]),0)
    median = cv2.medianBlur(img,f_length[2,j])
    bilateral = cv2.bilateralFilter(img,9,75,75)
    return(blur, gblur, median, bilateral)
#%%Save Images    
def outputImages(g_out, sp_out, cman_gnoise, cman_spnoise):
    cv2.imwrite('data/cman_gnoise.png', cman_gnoise)
    cv2.imwrite('data/cman_spnoise.png', cman_spnoise)
    for filter in g_out:
        file_path = 'data/' + filter + '_g.png'
        cv2.imwrite(file_path, g_out[filter])
    for filter in sp_out:
        file_path = 'data/' + filter + '_sp.png'
        cv2.imwrite(file_path, sp_out[filter])
 
#%%
s_time = time.clock()
cman_gnoise, cman_spnoise, std = genNoisy()
iterations = 1
g_out = {'blur' : np.zeros(0), 'gblur' : np.zeros(0),
           'median' : np.zeros(0), 'bilateral' : np.zeros(0), 'IHWT_soft' : np.zeros(0),
           'IHWT_hard' : np.zeros(0), 'IDB4T_hard' : np.zeros(0), 'IDB4T_hard' : np.zeros(0)}
           
sp_out = {'blur' : np.zeros(0), 'gblur' : np.zeros(0),
           'median' : np.zeros(0), 'bilateral' : np.zeros(0), 'IHWT_soft' : np.zeros(0), 'IHWT_hard' : np.zeros(0),
           'IDB4T_hard' : np.zeros(0), 'IDB4T_hard' : np.zeros(0)}

HWT_g = dwt.TwoD_HWT(cman_gnoise,iterations)
HWT_sp = dwt.TwoD_HWT(cman_spnoise,iterations)
DB4T_g = pywt.dwt2(cman_gnoise, 'db4')
DB4T_sp = pywt.dwt2(cman_spnoise, 'db4')

#This section need only be run once per new image to attain the appropriate Threshold value
#psnr, T, g_out[' = detThreshold(HWT_g, HWT_sp, DB4T_g, DB4T_sp, std, cman)
#pltDetThreshold(psnr)

f_length, psnr_best, psnr_hist = detFilterLength(cman_gnoise, cman_spnoise, cman)
#pltPSNRFilters(psnr_hist)

g_out['blur'], g_out['gblur'], g_out['median'], g_out['bilateral'] = standardFilters(cman_gnoise, f_length, 0)
sp_out['blur'], sp_out['gblur'], sp_out['median'], sp_out['bilateral'] = standardFilters(cman_spnoise, f_length, 1)

#Gaussian Noise Thresholding and Inverese Haar Transform
HWT_soft_g = pywt.threshold(HWT_g, T[0]*0.5*std, 'soft')
HWT_hard_g = pywt.threshold(HWT_g, T[1]*0.5*std, 'hard')
g_out['IHWT_soft']= dwt.TwoD_IHWT(HWT_soft_g,iterations)
g_out['IHWT_hard'] = dwt.TwoD_IHWT(HWT_hard_g,iterations)

#Salt & Pepper Noise, Thresholding, and Inverese Haar Transform
HWT_soft_sp = pywt.threshold(HWT_sp, T[2]*0.5*std, 'soft')
HWT_hard_sp = pywt.threshold(HWT_sp, T[3]*0.5*std, 'hard')
sp_out['IHWT_soft']= dwt.TwoD_IHWT(HWT_soft_sp,iterations)
sp_out['IHWT_hard'] = dwt.TwoD_IHWT(HWT_hard_sp,iterations)

#Gaussian Noise and S&P Thresholding for Inverese DB4 Transform
temp1 = np.array(pywt.threshold(DB4T_g[0], T[4]*0.5*std, 'soft'))
temp2 = np.array(pywt.threshold(DB4T_g[1], T[4]*0.5*std, 'soft'))

temp3 = np.array(pywt.threshold(DB4T_sp[0], T[6]*0.5*std, 'soft'))
temp4 = np.array(pywt.threshold(DB4T_sp[1], T[6]*0.5*std, 'soft'))

g_out['IDB4T_soft'] = pywt.idwt2((temp1,(temp2)),'db4')
sp_out['IDB4T_soft'] = pywt.idwt2((temp3,(temp4)),'db4')

temp1 = np.array(pywt.threshold(DB4T_g[0], T[5]*0.5*std, 'hard'))
temp2 = np.array(pywt.threshold(DB4T_g[1], T[5]*0.5*std, 'hard'))

temp3 = np.array(pywt.threshold(DB4T_sp[0], T[7]*0.5*std, 'hard'))
temp4 = np.array(pywt.threshold(DB4T_sp[1], T[7]*0.5*std, 'hard'))

g_out['IDB4T_hard'] = pywt.idwt2((temp1,(temp2)),'db4')
sp_out['IDB4T_hard'] = pywt.idwt2((temp3,(temp4)),'db4')

peakSNR(g_out, sp_out, cman)
#pltFilters(g_out, sp_out)
outputImages(g_out, sp_out, cman_gnoise, cman_spnoise)

e_time = time.clock()
t_time = e_time - s_time
print('\n', 'Total_Time: ', round(t_time,2)) 


#%%