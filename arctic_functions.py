import numpy as np
from scipy.spatial import KDTree

def zero2Nan(arr):
    """Convert Zeros in a given array to NaN
    
    Parameters
    ----------
    arr : ndarray
        The array containing zeros to be converted

    Returns
    -------
    ndarray
        The modified array with zeros converted to Nan
    """
    arr = np.where(arr==0, np.nan, arr)
    return arr

def symlog10(arr, thresh=1):
    """Take a symmetric log of a given array
    
    Parameters
    ----------
    arr : ndarray
        The array containing data to take the symmetric log of
    thresh : int
        The linearity threshold for the logarithm
        
    Returns
    -------
    ndarray
        The symmetric log of the given array
    """
    sign = np.sign(arr)
    mag = np.log10(1 + np.abs(arr)/(10**thresh))
    return sign*mag

def yearAvg(data, var, yearRange):
    """Compute the average of a given field over the given year range

    Parameters
    ----------
    data : dict
        The time series data of all the fields measured
    var : str
        The desired field to be averaged
    yearRange : tuple
        The first and last years (inclusive) to average over

    Returns
    -------
    avgData : 3darray
        The time average of a given field over the given year range
    """
    idxRange = (np.argwhere(np.array(data['Year']) == yearRange[0])[0,0], 
                np.argwhere(np.array(data['Year']) == yearRange[1])[0,0])
    return np.transpose(np.nanmean(data[var][idxRange[0]:idxRange[1]], axis=(0,1)))

def euclideanDist(p1, p2):
    """Compute the distance between two points according to the formula
    .. math:: r = \\sqrt{\\sum \\left(x_2 - x_1\\right)^2}
    .. warning :: The function will fail p1, p2 are not array of equal length
    
    Parameters
    ----------
    p1 : array
        The first point in n dimensions
    p2 : array
        The second point, also in n dimensions

    Returns
    -------
    float
        The Euclidean distance between p1 and p2
    """
    return np.sqrt(np.sum(np.square(p1 - p2)))

def haversineDist(p1, p2):
    """Compute the distance between two points according to the formula
    .. math:: r = 2R\\sin^{-1}\\sqrt{\\sin^{2}\\left(\\dfrac{\\phi_2 - 
        \\phi_1}{2}\\right) + 
        \\cos\\phi_1\\cos\\phi_2\\sin^2\\left(\\dfrac{\\lambda_2 - 
        \\lambda_1}{2}\\right)}
    .. warning :: The function will fail if p1, p2 are not arrays of length 2
    
    Parameters
    ----------
    p1 : array
        The first point in longitude, latitude
    p2 : array
        The second point in longitude, latitude

    Returns
    -------
    float
        The Haversine distance between p1 and p2
    """
    R = 6371.009

    lon1, lat1 = [np.radians(x) for x in p1]
    lon2, lat2 = [np.radians(x) for x in p2]

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2

    return 2*R*np.arcsin(np.sqrt(a))

def compute_mean(arr, w=None):
    """Compute a mean over three dimensions with the option for weighting

    Parameters
    ----------
    arr : 3darray
        The array of the value to be averaged
    w : 3darray (Default: None)
        Optional array of weights

    Returns
    -------
    float
        The (weighted) mean of the input array
    """
    arr_sum = 0
    w_sum = 0
    (m, n, p) = np.shape(arr)
    if w is None:
        w = np.ones((m, n, p))
    for i in range(0,m):
        for j in range(0,n):
            for k in range(0,p):
                if not np.isnan(arr[i,j,k]):
                    w_sum = w_sum + w[i,j,k]
                    arr_sum = arr_sum + arr[i,j,k]*w[i,j,k]
    if w_sum == 0:
        return np.nan
    return arr_sum/w_sum

def resample(oldgrid, olddata, newgrid):
    ocnmsk = newgrid['ocnmsk']
    m, n, p = np.shape(ocnmsk)
    u, v, _ = np.shape(oldgrid['ocnmsk'])
    r = int(u*v/(m*n))
    lons = newgrid['lon']
    lats = newgrid['lat']
    grid_array = np.stack((oldgrid['lon'].flatten(), oldgrid['lat'].flatten()), axis=-1)
    data = np.copy(olddata)
    data[data > 10**3] = np.nan
    targets = [[None, None]]
    target_ids = [[None, None]]
    for i in range(m):
        for j in range(n):
            try:
                data[i,j,:] = np.interp(newgrid['Depth'], oldgrid['Depth'][~np.isnan(data[i,j,:])], data[i,j,:][~np.isnan(data[i,j,:])])
            except:
                data[i,j,:] = np.nan
            targets = np.append(targets, [[lons[i, j], lats[i, j]]], axis=0)
            target_ids = np.append(target_ids, [[i, j]], axis=0)
    targets = targets[1:]
    target_ids = target_ids[1:]
    
    dist, idx = KDTree(grid_array).query(targets, k=r, workers=-1)
    flat_data = np.array([data[:,:,k].flatten() for k in range(p)])
    
    newdata = np.ones((m, n, p))*np.nan
    for x, target_id in enumerate(target_ids):
        for k in range(p):
            if ocnmsk[target_id[0], target_id[1], k] != 0:
                newdata[target_id[0], target_id[1], k] = np.nanmean(flat_data[k][idx[x]])
    return newdata

def makeRegions(grid):
    """Create regional logical arrays for the
    * Eurasian Basin (EB), 
    * Canadian Basin (CB), 
    * Shelf (shelf), 
    * Slope (slope), 
    * Arctic Ocean (ARC), and 
    * North Atlantic Ocean (NAt)
    
    Parameters
    ----------
    grid : dict
        A dictionary of grid information such as cell coordinates
        
    Returns
    -------
    dict
        A dictionary of regional logical 3darrays with keys in parenthesis above
    """
    ocnmsk = grid['ocnmsk']
    depths = grid['Depth']
    lats = grid['lat']
    lons = grid['lon']
    m, n, p = np.shape(ocnmsk)
    reg_logic = {}
    for key in ['ARC', 'EB', 'CB', 'shelf', 'slope', 'NAt']:
        reg_logic[key] = np.zeros((m, n, p))
    
    def isNAt(lat, lon, depth):
        if depth < 0:
            if (lat < 80 and lon > -50 and lon < 20):
                return True
            if (lat < 68 and (lon < -150 or lon > 170)):
                return True
        return False
    
    def isEB(lat, lon, depth):
        if (lon >= -60 and lon <= 140):
            if (lat >= 80 and depth < -3000):
                return True
            if lat > 85:
                return True
        return False
    
    def isCB(lat, lon, depth):
        if(lon > -120 and lon < -60):
            if (lat > 80 and depth < -3000):
                return True
            if lat > 85:
                return True
        if lon < -120:
            if lat > 70 and depth < -3000:
                return True
            if lat > 80:
                return True
        if (lat > 82 and lon > 140):
            return True
        return False
    
    def isShelf(lat, lon, depth):
        if depth > -500:
            return True
        if(lon > -90 and lon < -50 and lat < 80):
            return True
        return False
    
    for i in range(m):
        for j in range(n):
            for k in range(p):
                if ocnmsk[i,j,k] == 0:
                    lat = lats[i,j]
                    lon = lons[i,j]
                    depth = depths[k]
                    if isNAt(lat, lon, depth):
                        reg_logic['NAt'][i,j,:k] = 1
                    else:
                        reg_logic['ARC'][i,j,:k] = 1
                        if isEB(lat, lon, depth):
                            reg_logic['EB'][i,j,:k] = 1
                        elif isCB(lat, lon, depth):
                            reg_logic['CB'][i,j,:k] = 1
                        elif isShelf(lat, lon, depth):
                            reg_logic['shelf'][i,j,:k] = 1
                        else:
                            reg_logic['slope'][i,j,:k] = 1
                    break
    return reg_logic 
    
    
    
    
    
    
    
        