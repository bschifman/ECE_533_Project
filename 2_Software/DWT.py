import numpy as np
#%%Haar Transform   
# =============================================================================
def OneD_HWT(x):
     w = np.array([0.5, -0.5])
     s = np.array([0.5, 0.5])
 #    w = np.array([np.sqrt(2)/2, -np.sqrt(2)/2])
 #    s = np.array([np.sqrt(2)/2, np.sqrt(2)/2])
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
 #    w = np.array([np.sqrt(2)/2, -np.sqrt(2)/2])
 #    s = np.array([np.sqrt(2)/2, np.sqrt(2)/2])
     temp = np.float64(np.copy(x))
     h = np.int(len(temp)/2)
     i = 0
     while(i < h):  
         k = 2*i
         temp[k] = (x[i]*s[0] + x[i+h]*w[0])/w[0]
         temp[k+1] = (x[i]*s[1] + x[i+h]*w[1])/s[0]
         i += 1
     return(temp)
     
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
            
   