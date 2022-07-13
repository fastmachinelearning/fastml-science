import numpy as np
import ot

hexCoords = np.array([
    [0.0, 0.0], [0.0, -2.4168015], [0.0, -4.833603], [0.0, -7.2504044],
    [2.09301, -1.2083969], [2.09301, -3.6251984], [2.09301, -6.042], [2.09301, -8.458794],
    [4.18602, -2.4168015], [4.18602, -4.833603], [4.18602, -7.2504044], [4.18602, -9.667198],
    [6.27903, -3.6251984], [6.27903, -6.042], [6.27903, -8.458794], [6.27903, -10.875603],
    [-8.37204, -10.271393], [-6.27903, -9.063004], [-4.18602, -7.854599], [-2.0930138, -6.6461945],
    [-8.37204, -7.854599], [-6.27903, -6.6461945], [-4.18602, -5.4377975], [-2.0930138, -4.229393],
    [-8.37204, -5.4377975], [-6.27903, -4.229393], [-4.18602, -3.020996], [-2.0930138, -1.8125992],
    [-8.37204, -3.020996], [-6.27903, -1.8125992], [-4.18602, -0.6042023], [-2.0930138, 0.6042023],
    [4.7092705, -12.386101], [2.6162605, -11.177696], [0.5232506, -9.969299], [-1.5697594, -8.760895],
    [2.6162605, -13.594498], [0.5232506, -12.386101], [-1.5697594, -11.177696], [-3.6627693, -9.969299],
    [0.5232506, -14.802895], [-1.5697594, -13.594498], [-3.6627693, -12.386101], [-5.7557793, -11.177696],
    [-1.5697594, -16.0113], [-3.6627693, -14.802895], [-5.7557793, -13.594498], [-7.848793, -12.386101]])


#normalize so that distance between small cells (there are 4 per TC) is 1
oneHexCell = 0.5 * 2.4168015
#oneHexCell = 0.5 * np.min(ot.dist(hexCoords[:16],hexCoords[:16],'euclidean'))
hexCoords = hexCoords / oneHexCell
HexSigmaX = np.std(hexCoords[:,0])
HexSigmaY = np.std(hexCoords[:,1])

# pairwise distances
hexMetric = ot.dist(hexCoords, hexCoords, 'euclidean')
MAXDIST = np.max(hexMetric)

# calculate energy movers distance
# (cost, in distance, to move energy from one config to another)  
def emd(_x, _y, threshold=-1):
    if (np.sum(_x)==0): return -1.
    if (np.sum(_y)==0): return -0.5
    x = np.array(_x, dtype=np.float64)
    y = np.array(_y, dtype=np.float64)
    x = (1./x.sum() if x.sum() else 1.)*x.flatten()
    y = (1./y.sum() if y.sum() else 1.)*y.flatten()

    if threshold > 0:
        # only keep entries above 2%, e.g.
        x = np.where(x>threshold,x,0)
        y = np.where(y>threshold,y,0)
        x = 1.*x/x.sum()
        y = 1.*y/y.sum()

    return ot.emd2(x, y, hexMetric)

# difference in energy-weighted mean
def d_weighted_mean(x, y):
    if (np.sum(x)==0): return -1.
    if (np.sum(y)==0): return -0.5
    x = (1./x.sum() if x.sum() else 1.)*x.flatten()
    y = (1./y.sum() if y.sum() else 1.)*y.flatten()
    dx = hexCoords[:,0].dot(x-y)
    dy = hexCoords[:,1].dot(x-y)
    return np.sqrt(dx*dx+dy*dy)

def get_rms(coords, weights):
    mu_x = coords[:,0].dot(weights)
    mu_y = coords[:,1].dot(weights)
    sig2 = np.power((coords[:,0]-mu_x)/HexSigmaX, 2) \
         + np.power((coords[:,1]-mu_y)/HexSigmaY, 2)
    w2 = np.power(weights,2)
    return np.sqrt(sig2.dot(w2))

def d_weighted_rms(a, b):
    if (np.sum(a)==0): return -1.
    if (np.sum(b)==0): return -0.5
    # weights
    a = (1./a.sum() if a.sum() else 1.)*a.flatten()
    b = (1./b.sum() if b.sum() else 1.)*b.flatten()
    return get_rms(hexCoords,a) - get_rms(hexCoords,b)

def d_abs_weighted_rms(a, b):
    if (np.sum(a)==0): return -1.
    if (np.sum(b)==0): return -0.5
    return np.abs(d_weighted_rms(a, b))

# cross correlation of input/output 
def cross_corr(x,y):
    if (np.sum(x)==0): return -1.
    if (np.sum(y)==0): return -0.5
    cov = np.cov(x.flatten(),y.flatten())
    std = np.sqrt(np.diag(cov))
    stdsqr = np.multiply.outer(std, std)
    corr = np.divide(cov, stdsqr, out=np.zeros_like(cov), where=(stdsqr!=0))
    return corr[0,1]

def zero_frac(y):
    return np.all(y==0)

def ssd(x,y):
    if (np.sum(x)==0): return -1.
    if (np.sum(y)==0): return -0.5
    if (np.sum(x)==0 or np.sum(y)==0): return 1.
    ssd=np.sum(((x-y)**2).flatten())
    ssd = ssd/(np.sum(x**2)*np.sum(y**2))**0.5
    return ssd
