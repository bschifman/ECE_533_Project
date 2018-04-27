import numpy as np
import cv2
import matplotlib.pyplot as plt



def haar(a):
  scale = np.sqrt(2.)
  if len(a) == 1:
    return a.copy()
  assert len(a) % 2 == 0, "length needs to be even"
  mid = (a[0::2] + a[1::2]) / scale
  side = (a[0::2] - a[1::2]) / scale
  vector = np.hstack((haar(mid), side))
  
  return(vector)
  
def ihaar(a):
  scale = np.sqrt(2.)
  if len(a) == 1:
    return a.copy()
  assert len(a) % 2 == 0, "length needs to be even"
  mid = ihaar(a[0:np.int(len(a)/2)]) * scale
  side = a[int(len(a)/2):] * scale
  out = np.zeros(len(a), dtype=float)
  out[0::2] = (mid + side) / 2.
  out[1::2] = (mid - side) / 2.
  return out

def haar_2d(img):
  h,w = img.shape
  rows = np.zeros(img.shape, dtype=float)
  for y in range(h):
    rows[y] = haar(img[y])
  cols = np.zeros(img.shape, dtype=float)
  for x in range(w):
    cols[:,x] = haar(rows[:,x])
  return cols

def ihaar_2d(coeffs):
  h,w = coeffs.shape
  cols = np.zeros(coeffs.shape, dtype=float)
  for x in range(w):
    cols[:,x] = ihaar(coeffs[:,x])
  rows = np.zeros(coeffs.shape, dtype=float)
  for y in range(h):
    rows[y] = ihaar(cols[y])
  return rows

def keep_ratio(a, ratio):
  """
  Keep only the strongest values.
  """
  magnitude = sorted(np.abs(a.flatten()))
  idx = int((len(magnitude) - 1) * (1. - ratio))
  return np.where(np.abs(a) > magnitude[idx], a, 0)

# --- graphics-related code:

def to_float(img, gamma=2.2):
  """
  Convert uint8 image to linear floating point values.
  """
  return np.power(img.astype(float) / 255, gamma)

def from_float(img, gamma=2.2):
  """
  Convert from floating point, doing gamma conversion and 0,255 saturation,
  into a byte array.
  """
  out = np.power(img.astype(float), 1.0 / gamma)
  out = np.round(out * 255).clip(0, 255)
  return out.astype(np.uint8)

def bipolar(img):
  """
  Negative values are red, positive blue, and zero is black.
  """
  h,w = img.shape
  img = np.float64(img.copy())
  img /= np.max(np.abs(img))
  out = np.zeros((h, w, 3), dtype=float)
  a = .005
  b = 1. - a
  c = .5
  out[:,:,0] = np.where(img < 0, a + b * np.power(img / (img.min() - 0.001), c), 0)
  out[:,:,2] = np.where(img > 0, a + b * np.power(img / (img.max() + 0.001), c), 0)
  return from_float(out)


img = cv2.imread('data/cat.png', 0)
#img = cv2.resize(img, (512, 512)) 
coeffs = haar_2d(img)
strong_coeffs = keep_ratio(coeffs, .05)
lossy = ihaar_2d(strong_coeffs)
lossy2 = ihaar_2d(coeffs)

plt.figure()
plt.imshow(img,cmap='gray')

plt.figure()
plt.imshow(lossy2,cmap='gray')

plt.figure()
plt.imshow(bipolar(coeffs),cmap='gray')
# --- haar-related code:
#%%


# vim:set ts=2 sw=2 et: