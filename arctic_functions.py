import numpy as np
import gsw as gsw
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

def symlog10(arr):
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
    mag = np.log10(1 + np.abs(arr))
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

def nsquared(salinity, theta, lat, depth):
    SA = salinity
    CT = gsw.conversions.CT_from_pt(SA, theta)
    m, n, p = np.shape(salt)
    if np.shape(CT) != np.shape(SA) or len(depth) != p:
        print(f'Dimensions do not match: Salinity has shape {np.shape(SA)}, Temperature has shape {np.shape(CT)}, and Depth has length {len(depth)}')
        return None
    P = np.moveaxis(gsw.conversions.p_from_z(np.moveaxis(np.ones((m, n, 50))*depth,-1,0),[np.array(lat)]*50),0,-1)
    nsquared = np.ones((m, n, p))*np.nan
    for i in range(m):
        for j in range(n):
            SA_slice = SA[i,j,:]
            CT_slice = CT[i,j,:]
            P_slice = P[i,j,:]
            if sum(~np.isnan(SA_slice)) > 1:
                P_slice = P_slice[~np.isnan(SA_slice)]
                CT_slice = CT_slice[~np.isnan(SA_slice)]
                idx = np.squeeze(np.argwhere(~np.isnan(SA_slice)))
                SA_slice = SA_slice[~np.isnan(SA_slice)]
                n2, _ = gsw.stability.Nsquared(SA_slice, CT_slice, P_slice)
                nsquared[i,j,idx[0:len(n2)]] = n2
    return nsquared

def potentialDensity(salinity, theta, ref=0):
    SA = salinity
    CT = gsw.conversions.CT_from_pt(SA, theta)
    if ref == 0:
        func = gsw.density.sigma0
    elif ref == 1000:
        func = gsw.density.sigma1
    elif ref == 2000:
        func = gsw.density.sigma2
    elif ref == 3000:
        func = gsw.density.sigma3
    elif ref == 4000:
        func = gsw.density.sigma4
    else:
        print(f'{ref} dBar is an invalid reference pressure. Using 0 dBar.')
        func = gsw.density.sigma0
    return 1000 + func(SA, CT)

def resample(old_data, old_grid, new_grid):
    ocnmsk = new_grid['ocnmsk']
    m, n, p = np.shape(ocnmsk)
    u, v, _ = np.shape(old_grid['ocnmsk'])
    r = max(1, int(u*v/(m*n)))
    grid_array = np.stack((old_grid['lon'].flatten('F'), old_grid['lat'].flatten('F')), axis=-1)
    flat_data = old_data.reshape((-1, np.shape(old_data)[-1]), order='F')
    targets = np.stack((new_grid['lon'].flatten('F'), new_grid['lat'].flatten('F')), axis=-1)
    dist, idx = KDTree(grid_array).query(targets, k=r, workers=-1)
    if r > 1:
        new_data = np.nanmean(flat_data[idx], axis=1)
    else:
        new_data = flat_data[idx]
    new_data = np.array([np.interp(np.abs(new_grid['Depth']), np.abs(old_grid['Depth']), new_data[i,:])
                        for i in range(np.shape(new_data)[0])])
        
    new_data = new_data.reshape((m,n,p), order='F')
    return new_data

def makeRegions(old_regions, old_grid, new_grid):
    ocnmsk = new_grid['ocnmsk']
    m, n, p = np.shape(ocnmsk)
    lons = new_grid['lon']
    lats = new_grid['lat']
    grid_array = np.stack((old_grid['lon'].flatten('F'), old_grid['lat'].flatten('F')), axis=-1)
    flat_regions={}
    for key in old_regions:
        flat_regions[key] = old_regions[key][:,:,0].flatten('F')
    targets = np.stack((lons.flatten('F'), lats.flatten('F')), axis=-1)
    dist, idx = KDTree(grid_array).query(targets, workers=-1)
    new_regions={}
    for key in flat_regions:
        new_regions[key] = np.moveaxis([(flat_regions[key][idx]).reshape((m,n), order='F')]*p, 0, -1)*ocnmsk
    return new_regions
    
    
    
    
    
        