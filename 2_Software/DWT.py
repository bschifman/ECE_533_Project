import numpy as np
#%%Haar Transform   
# =============================================================================
def OneD_HWT(x):
     w = np.array([0.5, -0.5])
     s = np.array([0.5, 0.5])
#     w = np.array([np.sqrt(2)/2, -np.sqrt(2)/2])
#     s = np.array([np.sqrt(2)/2, np.sqrt(2)/2])
     temp_vector = np.float64((np.zeros(len(x))))
     h = np.int(len(temp_vector)/2)
     i = 0
     while(i < h):  
         k = 2*i
         temp_vector[i] = x[k]*s[0] + x[k+1]*s[1]
         temp_vector[i+h] = x[k]*w[0] + x[k+1]*w[1]
         i += 1
     return(temp_vector)
 #2D Haar Transform
def TwoD_HWT(x, iterations):
     temp = np.float64(np.copy(x))
     N_rows, N_cols = temp.shape
     
     k = 0    
     while(k < iterations):
         lev = 2**k
         lev_Rows = N_rows/lev
         lev_Cols = N_cols/lev        
         row = np.zeros(np.int(lev_Cols))
         i = 0
         while(i < lev_Rows):
             row = temp[i,:]
             temp[i,:] = OneD_HWT(row)
             i += 1
             
         col = np.zeros(np.int(lev_Rows))
         j = 0
         while(j < lev_Cols):
             col = temp[:,j]
             temp[:,j] = OneD_HWT(col)
             j += 1  
             
         k += 1
         
 #    temp = np.clip(temp, 0, 255)
     return(temp)
# =============================================================================

#%%Inverse Haar Transform
# =============================================================================
def OneD_IHWT(x):
     w = np.array([0.5, -0.5])
     s = np.array([0.5, 0.5])
#     w = np.array([np.sqrt(2)/2, -np.sqrt(2)/2])
#     s = np.array([np.sqrt(2)/2, np.sqrt(2)/2])
     temp_vector = np.float64(np.copy(x))
     h = np.int(len(temp_vector)/2)
     i = 0
     while(i < h):  
         k = 2*i
         temp_vector[k] = (x[i]*s[0] + x[i+h]*w[0])/w[0]
         temp_vector[k+1] = (x[i]*s[1] + x[i+h]*w[1])/s[0]
         i += 1
     return(temp_vector)
     
#2D Inverse Haar Transform         
def TwoD_IHWT(x, iterations):
     temp = np.float64(np.copy(x))
     N_rows, N_cols = temp.shape
     k = iterations - 1
     
     
     while(k >= 0):
         lev = 2**k      
         lev_Cols = N_cols/lev
         lev_Rows = N_rows/lev
         col = np.zeros(np.int(lev_Rows))
         
         j = 0
         while(j < lev_Cols):
             col = temp[:,j]
             temp[:,j] = OneD_IHWT(col)
             j += 1
             
         row = np.zeros(np.int(lev_Cols))
         i = 0
         while(i < lev_Rows):
             row = temp[i,:]
             temp[i,:] = OneD_IHWT(row)
             i += 1  
             
         k -= 1
     temp = np.clip(temp, 0, 255)
     return(np.uint8(temp))
# =============================================================================
            
# =============================================================================
def OneD_DB4(x, n):
     h = np.zeros(4)
     g = np.zeros(len(h))
     h[0] = (1+np.sqrt(3))/(4*np.sqrt(2))
     h[1] = (3+np.sqrt(3))/(4*np.sqrt(2))
     h[2] = (3-np.sqrt(3))/(4*np.sqrt(2))
     h[3] = (1-np.sqrt(3))/(4*np.sqrt(2))     
     g[0] = h[3]
     g[1] = -h[2]
     g[2] = h[1]
     g[3] = -h[0]
     
     temp_vector = np.float64((np.zeros(len(x))))
     half = np.int(len(temp_vector)/2)
     i = 0
     j = 0
     while(j < n-3):  
         temp_vector[i] = x[j]*h[0] + x[j+1]*h[1] + x[j+2]*h[2] + x[j+3]*h[3]
         temp_vector[i+half] = x[j]*g[0] + x[j+1]*g[1] + x[j+2]*g[2] + x[j+3]*g[3]
         j += 2
         i += 1
         
     temp_vector[i] = x[n-2]*h[0] + x[n-1]*h[1] + x[0]*h[2] + x[1]*h[3]
     temp_vector[i+half] = x[n-2]*g[0] + x[n-1]*g[1] + x[0]*g[2] + x[1]*g[3]
     return(temp_vector)
 #2D Daubechies Transform
def TwoD_DB4(x, iterations):
     temp = np.float64(np.copy(x))
     N_rows, N_cols = temp.shape
     
     k = 0    
     while(k < iterations):
         lev = 2**k
         lev_Rows = N_rows/lev
         lev_Cols = N_cols/lev        
         row = np.zeros(np.int(lev_Cols))
         i = 0
         while(i < lev_Rows):
             n = len(x)
             while(n >= 4):
                 row = temp[i,:]
                 temp[i,:] = OneD_DB4(row, n)
                 n = np.int(n/2)
             i += 1
             
         col = np.zeros(np.int(lev_Rows))
         j = 0
         while(j < lev_Cols):
             n = len(x)
             while(n >= 4):                 
                 col = temp[:,j]
                 temp[:,j] = OneD_DB4(col, n)
                 n = np.int(n/2)
                 
             j += 1  
             
         k += 1
         
 #    temp = np.clip(temp, 0, 255)
     return(temp)
# =============================================================================
     #%%Inverse Daubechies Transform
# =============================================================================
def OneD_IDB4(x, n):
     h = np.zeros(4)
     g = np.zeros(len(h))
     h[0] = (1+np.sqrt(3))/(4*np.sqrt(2))
     h[1] = (3+np.sqrt(3))/(4*np.sqrt(2))
     h[2] = (3-np.sqrt(3))/(4*np.sqrt(2))
     h[3] = (1-np.sqrt(3))/(4*np.sqrt(2))     
     g[0] = h[3]
     g[1] = -h[2]
     g[2] = h[1]
     g[3] = -h[0]
     
     temp_vector = np.float64(np.copy(x))
     half = np.int(len(temp_vector)/2)
     
     temp_vector[0] = x[half-1]/h[2] + x[n-1]/g[2] + x[0]/h[0] + x[half]/h[3]
     temp_vector[1] = x[half-1]/h[3] + x[n-1]/g[3] + x[0]/h[1] + x[half]/g[1]
     i = 0
     j = 2
     while(i < half-1):  
         temp_vector[j] = x[i]/h[2] + x[i+half]/g[2] + x[i+1]/h[0] + x[i+half+1]/h[3]  
         j += 1
         temp_vector[j] = x[i]/h[3] + x[i+half]/g[3] + x[i+1]/h[1] + x[i+half+1]/g[1]
         j += 1
         i += 1
     return(temp_vector)
     
#2D Inverse Daubechies Transform         
def TwoD_IDB4(x, iterations):
     temp = np.float64(np.copy(x))
     N_rows, N_cols = temp.shape
     k = iterations - 1
     
     
     while(k >= 0):
         lev = 2**k      
         lev_Cols = N_cols/lev
         lev_Rows = N_rows/lev
         col = np.zeros(np.int(lev_Rows))
         
         j = 0
         while(j < lev_Cols):
             n = 4
             while(n <= len(x)):
                 col = temp[:,j]
                 temp[:,j] = OneD_IDB4(col, n)
                 n = np.int(n*2)
             j += 1
             
         row = np.zeros(np.int(lev_Cols))
         i = 0
         while(i < lev_Rows):
             n = 4
             while(n <= len(x)):
                 row = temp[i,:]
                 temp[i,:] = OneD_IDB4(row, n)
                 n = np.int(n*2)
             i += 1  
             
         k -= 1
     temp = np.clip(temp, 0, 255)
     return(np.uint8(temp))
# =============================================================================