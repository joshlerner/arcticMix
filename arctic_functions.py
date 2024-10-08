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

def yearAvg(data, var, yearRange, months=None):
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
    if months is None:
        months = np.arange(12)
    return np.transpose(np.nanmean(data[var][idxRange[0]:1+idxRange[1], months], axis=(0,1)))

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

def zGradient(arr, grid, l=500):
    """Compute a vertical gradient of three dimensional array

    Parameters
    ----------
    arr : 3darray
        The array of scalar data
    grid : dict
        The dictionary of cell grid data
    l: int (Default: 500)
        Optional number of interpolation points. Increase for a smoother derivative.
        
    Returns
    -------
    3darray
        The vertical gradient of the data
    """
    (m, n, p) = np.shape(arr)
    interp_arr = np.zeros((m, n, l+2))
    interp_grad = np.zeros((m, n, l+1))
    grad = np.zeros((m, n, p))
    ocnmsk = grid['ocnmsk']
    z_coords = np.abs(grid['Depth'])
    dz = np.abs(np.nanmax(z_coords) - np.nanmin(z_coords))/l
    new_z = np.linspace(np.nanmin(z_coords-dz), np.nanmax(z_coords)+dz, l+2)
    for i in range(m):
        for j in range(n):
            interp_arr[i,j,:] = np.interp(new_z, z_coords, arr[i,j,:])
            interp_grad[i,j,:] = (interp_arr[i,j,:-1] - interp_arr[i,j,1:])/dz
            grad[i,j,:] = np.interp(z_coords, new_z[:-1]+dz/2, interp_grad[i,j,:])
    grad = grad*zero2Nan(ocnmsk)
    for i in range(m):
        for j in range(n):
            nans = np.where(np.isnan(grad[i,j,:]))[0]
            idx = -1
            if len(nans) > 0:
                idx = np.min(nans) - 1
            grad[i,j,idx] = np.nan
    return grad

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

def nsquared(salinity, theta, grid):
    """Compute the stratification (Squared Brunt-Väisälä Frequency) from Practical Salinity and Potential Temperature
    
    Parameters
    ----------
    salinity : 3darray
        The Practical Salinity field of the data
    theta : 3darray
        The Potential Temperature field of the data
    grid : dict
        The dictionary of grid cell data
        
    Returns
    -------
    3darray
        The stratification field of the data
    """
    lat = grid['lat']
    lon = grid['lon'] + 180
    depth = grid['Depth']
    ocnmsk = grid['ocnmsk']
    m, n, p = np.shape(salinity)
    P = np.moveaxis(gsw.conversions.p_from_z(np.moveaxis(np.ones((m, n, 50))*depth,-1,0),[np.array(lat)]*50),0,-1)
    SA = gsw.conversions.SA_from_SP(salinity, P, np.moveaxis([np.array(lon)]*50, 0, -1), np.moveaxis([np.array(lat)]*50, 0, -1))
    CT = gsw.conversions.CT_from_pt(SA, theta)
    
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
    nsquared = nsquared*zero2Nan(ocnmsk)
    for i in range(m):
        for j in range(n):
            nans = np.where(np.isnan(nsquared[i,j,:]))[0]
            idx = 0
            if len(nans) > 0:
                idx = np.min(nans)
            nsquared[i,j,idx-2:] = np.nan
    return nsquared

def potentialDensity(salinity, theta, ref=0):
    """Compute the Potential Density from the Practical Salinity and Potential Temperature
    
    Parameters
    ----------
    salinity : 3darray
        The Practical Salinity field of the data
    theta : 3darray
        The Potential Temperature field of the data
    ref : One of {0, 1000, 2000, 3000, 4000} (Default: 0)
        The reference pressure
        
    Returns
    -------
    3darray
        The stratification field of the data
    """
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
    """Resample a data field from an old grid onto a new grid of a different size

    Parameters
    ----------
    old_data : 3darray
        The original data field to be resampled
    old_grid : dict
        The grid cell data of the original grid
    new_grid : dict
        The new grid cell data to resample onto

    Returns
    -------
    3darray
        The resampled data on the new grid
    """
    ocnmsk = new_grid['ocnmsk']
    m, n, p = np.shape(ocnmsk)
    r = max(1, int(np.nanmean(new_grid['area'])/np.nanmean(old_grid['area'])))
    grid_array = np.stack((old_grid['lon'].flatten('F'), old_grid['lat'].flatten('F')), axis=-1)
    flat_data = old_data.reshape((-1, np.shape(old_data)[-1]), order='F')
    nanmask = ~np.any(np.isnan(grid_array), axis=-1)
    grid_array = grid_array[nanmask]
    flat_data = flat_data[nanmask]
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
    """Resample Arctic Regional Logicals from an old grid onto a new grid of a different size

    Parameters
    ----------
    old_regions : dict
        The original Arctic Regional Logicals to be resampled
    old_grid : dict
        The grid cell data of the original grid
    new_grid : dict
        The new grid cell data to resample onto

    Returns
    -------
    3darray
        The resampled Arctic Regional Logicals on the new grid
    """
    ocnmsk = new_grid['ocnmsk']
    m, n, p = np.shape(ocnmsk)
    grid_array = np.stack((old_grid['lon'].flatten('F'), old_grid['lat'].flatten('F')), axis=-1)
    
    flat_regions={}
    for key in old_regions:
        flat_regions[key] = old_regions[key][:,:,0].flatten('F')
    targets = np.stack((new_grid['lon'].flatten('F'), new_grid['lat'].flatten('F')), axis=-1)
    nanmask = np.any(np.isnan(targets), axis=-1)
    targets[np.isnan(targets)] = 0
    dist, idx = KDTree(grid_array).query(targets, workers=-1)
    new_regions={}
    for key in flat_regions:
        flat_regions[key][idx[dist > 2]] = 0 # MAGIC NUMBER! ONLY for making ASTE and MIT work
        new_regions[key] = np.moveaxis([(flat_regions[key][idx]).reshape((m,n), order='F')]*p, 0, -1)*ocnmsk
        new_regions[key][np.moveaxis([nanmask.reshape((m,n), order='F')]*p, 0, -1)] = 0
    return new_regions
    
    
    
    
    
        