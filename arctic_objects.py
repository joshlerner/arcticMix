import numpy as np
import cartopy.crs as ccrs
from arctic_functions import *
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib import ticker
from scipy.spatial import KDTree
import warnings

class Field:
    """
    The Field object for fields in the Arctic Ocean
    
    Parameters
    ----------
    desc : str
        The descriptor is used to label the field source
    name : str
        The name is used to label the field data
    units : str
        The units are used to label the field units
    grid : dict
        The grid is used to get geometric information for the field
    data : 3darray
        The data are the values belonging to the field
    reg_logic : dict, optional
        The regional logicals are used to divide the data into regions
    """
    def __init__(self, desc, name, units, grid, data, reg_logic=None):
        self._name = name
        self._long_name = desc +' '+ name
        self._units = units
        self._grid = grid
        self._data = zero2Nan(data*grid['ocnmsk'])
        if reg_logic is None:
            self._reg_logic = makeRegions(grid)
        else:
            self._reg_logic = reg_logic
        
        self._means = None
        self._weighted_means = None
        self._geo_means = None
        self._weighted_geo_means = None
        self._grid['vol'] = self._grid['vol']*self._reg_logic['ARC']

    @property
    def name(self):
        return self._name

    @property
    def long_name(self):
        return self._long_name

    @property
    def data(self):
        return self._data

    @property
    def units(self):
        return self._units
        
    def getMeans(self, weighted, geometric, alt_data=None):
        """Computes the regional means of a field, or loads precomputed means

        Parameters
        ----------
        self : Field
            The field of data
        weighted : boolean
            Whether to weight the means by cell volume
        geometric : boolean
            Whether to compute a geometric mean
        alt_data : 3darray, optional
            An alternative dataset used for averaging

        Returns
        -------
        dict
            The regional means of a field
        """
        if weighted:
            weights = self._grid['vol']
            if geometric:
                ref = self._weighted_geo_means
            else:
                ref = self._weighted_means
        else:
            weights = None
            if geometric:
                ref = self._geo_means
            else:
                ref = self._means
        if alt_data is None:
            if ref is not None:
                return ref
            else:
                data = self._data
        else:
            data = alt_data
        means = {}
        if geometric:
            if np.any(data < 0):
                print('Warning: attempting a geometric mean on signed data is not advised. Negative data will be ignored.')
            data = np.log10(zero2Nan(data))
        for region in ['ARC', 'CB', 'EB', 'basin', 'shelf', 'slope']:
            if region == 'basin':
                means['basin'] = compute_mean(zero2Nan(data*(self._reg_logic['CB'] +
                                                             self._reg_logic['EB'])), w=weights)
            else:
                means[region] = compute_mean(zero2Nan(data*self._reg_logic[region]), w=weights)
            if geometric:
                means[region] = 10**means[region]
        if alt_data is None:
            ref = means
        return means
    
    def grid(self, key):
        """ Gets the specified attribute of the field grid

        Parameters
        ----------
        self : Field
            The field of data
        key : str
            The name of the grid attribute

        Returns
        -------
        ndarray
            A copy of the grid attribute
        
        Raises
        ------
        KeyError
            If the provided key is not in the field grid
        """
        if key in self._grid.keys():
            return np.copy(self._grid[key])
        else:
            raise KeyError(f"{key} is not a valid grid attribute.")
    
    def region(self, region):
        """ Gets the specified regional logical

        Parameters
        ----------
        self : Field
            The field of data
        region : str
            The name of the region

        Returns
        -------
        3darray
            A copy of the logical array for the region

        Raises
        ------
        KeyError
            If the provided region is not in the regional dictionary
        """
        if region in self._reg_logic.keys():
            return np.copy(self._reg_logic[region])
        else:
            raise KeyError(f"{region} is not a valid region.")

    def visualize_regional_profile(self, weighted=True, geometric=False, scale=None, show=True, **kwargs):
        """ Plots a vertical average profile for each region

        Parameters
        ----------
        self : Field
            The field of data
        weighted : boolean, default True
            Whether to use volume weighted means
        geometric : boolean, default False
            Whether to use geometric means
        scale : str, optional
            Apply either a 'log' or 'symlog' scaling to the data
        show : boolean, default False
            Whether to display the figure
        **kwargs
            Additional parameters are passed to the `matplotlib.pyplot.plot` method

        Returns
        -------
        fig
            A figure with the field's vertical average profile in each region
        """
        data = self._data
        means = self.getMeans(weighted, geometric)
        if weighted:
            weights = self._grid['vol']
        else:
            weights = np.ones(np.shape(data))
        if geometric:
            if np.any(data < 0):
                print('Warning: attempting a geometric mean on signed data is not advised. Negative data will be ignored.')
            data = np.log10(zero2Nan(data))
        z_means = {}
        p = len(self._grid['Depth'])
        vec = np.ones((p,1))
        fig = plt.figure(figsize=(10, 12))
        plt.rcParams.update({'font.size': 22})
        colors = plt.get_cmap('tab10')(np.arange(10, dtype=int))
        for i, k in enumerate(means.keys()):
            if k in self._reg_logic.keys():
                z_means[k] = np.ones((p,1))*np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mdata = zero2Nan(data*self._reg_logic[k])
                    for j in range(p):
                        z_means[k][j] = compute_mean(mdata[:,:,j:j+1], w=weights[:,:,j:j+1])
                    if geometric:
                        z_means[k] = 10**z_means[k]
                    plt.plot(z_means[k], self._grid['Depth'], color=colors[i], marker='o', label=k, **kwargs)
                    plt.plot(means[k]*vec, self._grid['Depth'], color=colors[i], linestyle='-.', **kwargs)
    
        plt.ylabel('Depth (m)')
        plt.yscale('symlog')
        plt.xlabel(f'{self._name} ({self._units})')
        if scale == 'symlog':
            plt.xscale('symlog')
        elif scale == 'log':
            plt.xscale('log')
    
        pos = plt.gca().get_position()
        plt.gca().set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        plt.gca().legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    
        plt.title(f'{self._long_name}\n{'Weighted ' if weighted else ''}{'Geometric ' if geometric else ''}Mean Profile')
        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.close()
        return fig
        
    def visualize_distributions(self, scale=None, show=True, **kwargs):
        """ Plots histograms of the field in each region for all depths

        Parameters
        ----------
        self : Field
            The field of data
        scale : str, optional
            Apply either a 'log' or 'symlog' scaling to the data
        show : boolean, default False
            Whether to display the figure
        **kwargs
            Additional parameters are passed to the `matplotlib.pyplot.hist` method

        Returns
        -------
        fig
            A figure with histograms of the field in each region for all depths
        """
        data = self._data
        vol = 100*self._grid['vol']/np.nansum(self._grid['vol'])
        (m, n, p) = np.shape(data)
        fig, axes = plt.subplots(int(p/4), 4, figsize=(24, 5*int(p/4)))
        plt.rcParams.update({'font.size': 22})
        colors = plt.get_cmap('tab10')(np.arange(10, dtype=int))
        for i, depth in enumerate(self._grid['Depth']):
            if i < 4*int(p/4):
                ax = axes[int(i/4), int(i%4)]
                data_slice = data[:,:,i]
                weight_slice = vol[:,:,i]
                for j, key in enumerate(['CB', 'EB', 'shelf', 'slope']):
                    msk = (np.isnan(data_slice) == False) & (self._reg_logic[key][:,:,i] == True)
                    x = data_slice[msk]
                    w = weight_slice[msk]  
                    if scale == 'symlog':
                        x = symlog10(zero2Nan(x))
                    elif scale == 'log':
                        x = np.log10(zero2Nan(x))
                    if len(x) != 0:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                n_bin = int((np.nanmax(x) - np.nanmin(x))/0.2)+1
                                ax.hist(x, bins=n_bin, weights=w, histtype='stepfilled', label=key, ec=colors[j],
                                        fc=np.append(colors[j][:-1], 0.3), **kwargs)
                            except:
                                continue
                if ax.get_legend_handles_labels() == ([],[]):
                    ax.set_axis_off()
                else:
                    ax.set_title(f'z = {np.abs(depth):.3}')
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if scale == 'symlog':
                            ax.set_xticklabels([r'$'+str(int(np.sign(exp)*10))+'^{'+str(int(abs(exp)))+r'}$' 
                                                for exp in ax.get_xticks()])
                        elif scale == 'log':
                            ax.set_xticklabels([r'$10^{'+str(int(exp))+r'}$' 
                                                for exp in ax.get_xticks()])
                        
        axes[0,1].set_xlabel(f'{self._name} ({self._units})')
        axes[0,1].set_ylabel('% Volume')
        handles, labels = axes[0,1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.1, 1))

        st = plt.suptitle(f'{self._long_name} Distribution by Depth')
    
        st.set_y(1)
        fig.subplots_adjust(top=0.85)
        plt.tight_layout()
                
        if show:
            plt.show()
        else:
            plt.close()
        return fig

    def visualize_maps(self, scale=None, show=True, vmin=None, vmax=None, **kwargs):
        """ Plots maps of the field for all depths

        Parameters
        ----------
        self : Field
            The field of data
        scale : str, optional
            Apply either a 'log' or 'symlog' scaling to the data
        show : boolean, default False
            Whether to display the figure
        **kwargs
            Additional parameters are passed to the `matplotlib.pyplot.hist` method

        Returns
        -------
        fig
            A figure with maps of the field for all depths
        """
        data = self._data
        data[self._grid['ocnmsk'] == 0] = np.nan
        (m, n, p) = np.shape(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if scale == 'symlog':
                data = symlog10(zero2Nan(data))
                vmin = -1*vmax
            elif scale == 'log':
                data = np.log10(zero2Nan(data))
            else:
                data = zero2Nan(data)
                
        if vmin is None:
            vmin = np.nanpercentile(data, 1)
        if vmax is None:
            vmax = np.nanpercentile(data, 99)

        if scale == 'log':
            cmap = 'viridis'
        else:
            if vmin < 0:
                vmax = np.max((-vmin, vmax))
                vmin = -vmax
                cmap = 'seismic'
            else:
                cmap = 'viridis'
        lev = np.linspace(vmin, vmax, 100)

        
        lons = self._grid['lon']
        lats = self._grid['lat']
    
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)

        fig, axes = plt.subplots(int(p/4), 4, 
                                 figsize=(24, 5*int(p/4)), subplot_kw={'projection': ccrs.NorthPolarStereo()})
        plt.rcParams.update({'font.size': 22})
        for i, depth in enumerate(self._grid['Depth']):
            if i < 4*int(p/4):
                ax = axes[int(i/4), int(i%4)]
                data_slice = data[:,:,i]
                plt.set_cmap(cmap)
                if not np.all(np.isnan(data_slice)):
                    pcm = ax.pcolormesh(lons, lats, data_slice, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, **kwargs)
                    ax.set_extent([-180, 180, 90, 60], ccrs.PlateCarree())
                    ax.coastlines()
                    ax.gridlines(lw=1, ls=':', draw_labels=True, rotate_labels=False, xlocs=np.linspace(-180, 180, 13), ylocs=[])
                    ax.set_boundary(circle, transform=ax.transAxes)
                    ax.set_title(f'z = {np.abs(depth):.3}')
                else:
                    ax.set_axis_off()
            
        st = plt.suptitle(f'{self._long_name} Arctic Map')
        plt.tight_layout()
        cbar = fig.colorbar(pcm, ax=axes[0,:], location='top', extend='both')
        cbar.locator = ticker.AutoLocator()
        cbar.set_ticks(cbar.locator.tick_values(vmin, vmax))
        if scale == 'symlog':
            cbar.set_ticklabels([r'$'+str(int(np.sign(exp)*10))+'^{'+str(int(abs(exp)))+r'}$' for exp in cbar.get_ticks()])
        elif scale == 'log':
            cbar.set_ticklabels([r'$10^{'+str(int(exp))+r'}$' for exp in cbar.get_ticks()])
        cbar.minorticks_off()
        cbar.set_label(f'{self._name} ({self._units})')

        st.set_y(1)
        fig.subplots_adjust(top=0.94)
            
        if show:
            plt.show()
        else:
            plt.close()
        return fig
        
class KappaBG(Field):
    """
    A subclass of Fields for background diffusivity in the Arctic Ocean

    Parameters
    ----------
    desc : str
        The descriptor is used to label the field source
    grid : dict
        The grid is used to get geometric information for the field
    obs_data : 3darray
        The observational data used to build the background data
    reg_logic : dict, optional
        The regional logicals are used to divide the data into regions

    Methods
    -------
    createKappa(ver='OBS')
        Creates background diffusivities from the observed data
    """
    def __init__(self, desc, grid, obs_data, reg_logic, ver='OBS'):
        super().__init__(desc, 'Background Diffusivity', r'$m^2/s$', grid, obs_data, reg_logic=reg_logic)
        self._data = self.__createKappa(ver)
        
        
    def __createKappa(self, ver='OBS'):
        (m, n, p) = np.shape(self._data)
        kappa = np.zeros((m, n, p))
        ocnmsk = self._grid['ocnmsk']
        weighted_means = self.getMeans(True, False)
        if ver == 'CTL':
            kappa += self._reg_logic['ARC']*weighted_means['ARC']
        elif ver == 'HI':
            kappa += self._reg_logic['ARC']*weighted_means['shelf']
        elif ver == 'LO':
            kappa += self._reg_logic['ARC']*weighted_means['basin']
        elif ver == 'REG':
            kappa += self._reg_logic['CB']*weighted_means['CB']
            kappa += self._reg_logic['EB']*weighted_means['EB']
            kappa += self._reg_logic['shelf']*weighted_means['shelf']
            kappa += self._reg_logic['slope']*weighted_means['slope']
        else:
            return self._data
        kappa += self._reg_logic['NAt']*6.6e-6
        kappa = zero2Nan(kappa)
        kappa = zero2Nan(kappa*ocnmsk)
        if np.any(np.equal(ocnmsk > 0, np.isnan(kappa))):
            print('Error: there are ocean grid cells that do not contain a kappa value')
        return kappa

class Transect:
    """
    The Transect object for coordinate cross sections in the Arctic Ocean

    Parameters
    ----------
    name : str
        The name of the transect
    grid : dict
        The geometric information for the transect
    p1 : tuple
        A longitude, latitude pair of starting coordinates
    p2 : tuple
        A longitude, latitude pair of ending coordinates
    proj : Cartopy Projection, default NorthPolarStereo
        The coordinate projection to make cross sections in

    Methods
    -------
    initializeCoords(p1, p2)
        Converts the starting and ending coordinates into the the projection basis
    makeTransectLine()
        Makes an array of indices for data and distances along the transect line
    """
    def __init__(self, name, grid, p1, p2, proj=ccrs.NorthPolarStereo()):
        self.name = name
        self.proj = proj
        self.base = ccrs.PlateCarree()
        self.__initializeCoords(p1, p2)
        self.node_spacing = 35.849468668883645
        self.node_number = int(np.ceil(euclideanDist(np.array([self.coords['x_s'], self.coords['y_s']]),
                                                     np.array([self.coords['x_t'], self.coords['y_t']]))/self.node_spacing))
        self.grid = grid
        self.__makeTransectLine()
        
    def __initializeCoords(self, p1, p2):
        coords = {'lat_s': p1[1], 
                  'lon_s': p1[0], 
                  'lat_t': p2[1], 
                  'lon_t': p2[0]}
        coords['x_s'], coords['y_s'] = self.proj.transform_point(coords['lon_s'], coords['lat_s'], self.base)
        coords['x_t'], coords['y_t'] = self.proj.transform_point(coords['lon_t'], coords['lat_t'], self.base)
        self.coords = coords
        
    def __makeTransectLine(self):
        dx = (self.coords['x_t'] - self.coords['x_s'])/self.node_number
        dy = (self.coords['y_t'] - self.coords['y_s'])/self.node_number
        
        ts_line = np.zeros((self.node_number, 2))
        ts_line[0] = [self.coords['x_s'], self.coords['y_s']]
        for i in range(1, self.node_number):
            ts_line[i] = ts_line[i-1] + [dx, dy]
        self.line = ts_line
        
        lons = self.grid['lon'].flatten()
        lats = self.grid['lat'].flatten()
        trans_array = self.base.transform_points(self.proj, self.line[:,0], self.line[:,1])[:,:-1]
        grid_array = np.stack((lons, lats), axis=-1)
        _, trans_idx = KDTree(grid_array).query(trans_array)
        self.trans_idx = trans_idx
        
        trans_dist = np.ones((self.node_number, 1))*np.nan
        trans_dist[0] = 0
        for i in range(1, self.node_number):
            trans_dist[i] = trans_dist[i-1] + haversineDist(trans_array[i-1], trans_array[i])
        self.trans_dist = trans_dist
        
    def visualize_transect(self, field, log=True, scale=None,  show=True, **kwargs):
        """ Plots the given field along the cross section

        Parameters
        ----------
        self : Transect
            The cross section to plot the data along
        field : Field
            The field of data
        scale : str, optional
            Apply either a 'log' or 'symlog' scaling to the data
        vmin : float, optional
            The minimum value to display on the colorbar
        vmax : float, optional
            The maximum value to display on the colorbar
        show : boolean, default False
            Whether to display the figure
        **kwargs
            Additional parameters are passed to the `matplotlib.pyplot.contourf` method

        Returns
        fig
            A figure with a field along the cross section
        """
        depths = self.grid['Depth']
        ocnmsk = self.grid['ocnmsk']
        fig = plt.figure(figsize=(20, 10))
        plt.rcParams.update({'font.size': 22})
        
        trans_fld = np.ones((len(depths), self.node_number))*np.nan
        data = zero2Nan(field.data*ocnmsk)
        for d in range(len(depths)):
            fld_slice = data[:,:,d]
            if scale=='symlog':
                fld_slice = symlog10(zero2Nan(fld_slice))
            elif scale=='log':
                fld_slice = np.log10(zero2Nan(fld_slice))
            else:
                fld_slice = zero2Nan(fld_slice)
            trans_fld[d,:] = fld_slice.flatten()[self.trans_idx]
        
        X, Y = np.meshgrid(self.trans_dist, depths)

        if vmin is None:
            vmin = np.nanpercentile(trans_fld, 1)
        if vmax is None:
            vmax = np.nanpercentile(trans_fld, 99)
        
        plt.gca().set_facecolor('dimgrey')
        if scale == 'log':
            cmap = 'viridis'
            lev = np.linspace(vmin, vmax, 100)
        else:
            if vmin < 0:
                vmax = np.max((-vmin, vmax))
                vmin = -vmax
                cmap = 'seismic'
            else:
                cmap = 'viridis'
            lev = np.linspace(vmin, vmax, 100)
        
        cs = plt.contourf(X, Y, trans_fld, lev, cmap=cmap, extend='both', **kwargs)
        ymin = depths[np.count_nonzero(np.nansum(trans_fld,1)) + 1] - 100
        ymax = depths[1]
        plt.ylim((ymin, ymax))
        
        cbar = plt.colorbar(cs)
        cbar.locator = ticker.AutoLocator()
        cbar.set_ticks(cbar.locator.tick_values(lev[1], lev[-2]))
        if scale == 'symlog':
            cbar.set_ticklabels([r'$'+str(int(np.sign(exp)*10))+'^{'+str(abs(exp))+r'}$' for exp in cbar.get_ticks()])
        elif scale == 'log':
            cbar.set_ticklabels([r'$10^{'+str(exp)+r'}$' for exp in cbar.get_ticks()])
        cbar.minorticks_off()
        cbar.set_label(f'{field.name} ({field.units})')
        plt.xlabel('Along-Track Distance (km)')
        plt.ylabel('Depth (m)')
        if log == True:
            plt.yscale('symlog')
        plt.title(f'{field.long_name}\n along {self.name}')
        plt.tight_layout()
        
        if show:
            plt.show()
        else:
            plt.close()
        return fig
    
    def visualize_line(self, show=True):
        """ Plots the cross section on the map projection

        Parameters
        ----------
        self : Transect
            The cross section to plot the data along
        show : boolean, default False
            Whether to display the figure

        Returns
        -------
        fig
            A figure with the cross section on a map projection
        """
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.NorthPolarStereo()})
        
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        
        ax.coastlines()
        ax.stock_img()
        ax.gridlines(lw=1, ls=':', draw_labels=True, rotate_labels=False, xlocs=np.linspace(-180, 180, 13), ylocs=[])
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.plot(self.line[:,0], self.line[:,1], 'r')
        ax.set_extent([-180, 180, 90, 60], self.base)
        
        plt.title(self.name)
        plt.tight_layout()
        
        if show:
            plt.show()
        else:
            plt.close()
        return fig

def visualize_average_profiles(fields, weighted=True, geometric=False, scale=None, show=True, **kwargs):
    """ Plots the vertical average profiles of each field by region

    Parameters
    ----------
    fields : list(Field)
        The fields  of data
    weighted : boolean, default True
        Whether to use volume weighted means
    geometric : boolean, default False
        Whether to use geometric means
    scale : str, optional
        Apply either a 'log' or 'symlog' scaling to the data
    show : boolean, default False
        Whether to display the figure
    **kwargs
        Additional parameters are passed to the `matplotlib.pyplot.plot` method

    Returns
    -------
    fig
        A figure with the vertical averge profiles of each field by region
    """
    z_means = {}
    w_means = {}
    for field in fields:
        name = field.long_name
        data = field.data
        if weighted:
            weights = field.grid('vol')
        else:
            weights = np.ones(np.shape(data))
        if geometric:
            if np.any(data < 0):
                print('Warning: attempting a geometric mean on signed data is not advised. Negative data will be ignored.')
            data = np.log10(zero2Nan(data))
        z_means[name] = {}
        w_means[name] = field.getMeans(weighted, geometric)
        p = len(field.grid('Depth'))
        for key in w_means[name].keys():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    reg = field.region(key)
                    
                    z_means[name][key] = np.ones((p,1))*np.nan
                    mdata = zero2Nan(data*reg)
                    for k in range(p):
                        z_means[name][key][k] = compute_mean(mdata[:,:,k:k+1], w=weights[:,:,k:k+1])
                    if geometric:
                        z_means[name][key] = 10**z_means[name][key]
            except:
                continue
    fig, axes = plt.subplots(1, 5, sharey=True, figsize=(36,12))
    vec = np.ones((50,1))
    plt.rcParams.update({'font.size': 22})
    colors = plt.get_cmap('tab10')(np.arange(10, dtype=int))
    for i, key in enumerate(z_means[fields[0].long_name].keys()):
        for j, field in enumerate(fields):
            axes[i].plot(z_means[field.long_name][key][1:], field.grid('Depth')[1:], color=colors[j], marker='o', label=field.long_name, **kwargs)
            axes[i].plot(w_means[field.long_name][key]*vec, field.grid('Depth'), color=colors[j], linestyle='-.', **kwargs)
            axes[i].set_title(key)
            axes[i].set_yscale('symlog')
            axes[i].grid(True)
            if scale == 'symlog':
                axes[i].set_xscale('symlog')
            elif scale == 'log':
                axes[i].set_xscale('log')
        
    axes[2].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5))
    plt.xlabel(fields[0].name)
    axes[0].set_ylabel('Depth (m)')
    st = plt.suptitle(f'{fields[0].name}\n{'Weighted ' if weighted else ''}{'Geometric ' if geometric else ''}Mean Profiles')
    st.set_y(1)
    fig.subplots_adjust(top=0.85)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()
    return fig
    
def volumeCensus(xfield, yfield, reg='ARC', contour=None, show=True, **kwargs):
    if np.shape(xfield.data) != np.shape(yfield.data):
        print('Error: Fields must contain data of equal dimensions.')
        print(f'Given {xfield.name}: {np.shape(xfield.data)} and {yfield.name}: {np.shape(yfield.data)}')
        return None
    xdata = zero2Nan(xfield.data*xfield.region(reg)).flatten()
    ydata = zero2Nan(yfield.data*yfield.region(reg)).flatten()[~np.isnan(xdata)]
    vol = 100*xfield.grid('vol').flatten()[~np.isnan(xdata)]/np.nansum(zero2Nan(xfield.grid('vol')*xfield.region('ARC')))
    xdata = xdata[~np.isnan(xdata)]

    fig = plt.figure(figsize=(16, 8))
    plt.rcParams.update({'font.size': 22})
    hh = plt.hist2d(xdata, ydata, bins=100, weights=vol, cmap='ocean_r', norm='log', vmin=10**(-5), **kwargs)
    cbar = plt.colorbar(extend='both')
    cbar.set_label('% Arctic Ocean Volume')
    if contour is not None:
        X, Y = np.meshgrid(hh[1], hh[2])
        Z = contour['func'](X, Y, **contour['args'])
        cs = plt.contour(X, Y, Z, levels=6, colors='black')
        plt.clabel(cs, levels=cs.levels[2:], inline=True, rightside_up=False, fmt=lambda lev : rf'{lev:.0f} kg/m$^3$')
    
    plt.xlabel(f'{xfield.name} ({xfield.units})')
    plt.ylabel(f'{yfield.name} ({yfield.units})')
    plt.title(f'{yfield.long_name} and {xfield.name} {reg} Volume Census')
    plt.tight_layout()
        
    if show:
        plt.show()
    else:
        plt.close()
    return fig

def anomalyVolumeCensus(xfields, yfields, reg='ARC', contour=None, show=True, **kwargs):
    if np.shape(xfields[0].data) != np.shape(yfields[0].data):
        print('Error: Fields must contain data of equal dimensions.')
        print(f'Given {xfields[0].name}: {np.shape(xfields[0].data)} and {yfields[0].name}: {np.shape(yfields[0].data)}')
    if np.shape(xfields[1].data) != np.shape(yfields[1].data):
        print('Error: Fields must contain data of equal dimensions.')
        print(f'Given {xfields[1].name}: {np.shape(xfields[1].data)} and {yfields[1].name}: {np.shape(yfields[1].data)}')
    
    xdata = zero2Nan(xfields[0].data*xfields[0].region(reg)).flatten()
    ydata = zero2Nan(yfields[0].data*yfields[0].region(reg)).flatten()[~np.isnan(xdata)]
    vol = 100*xfields[0].grid('vol').flatten()[~np.isnan(xdata)]/np.nansum(zero2Nan(xfields[0].grid('vol')*xfields[0].region('ARC')))
    xdata = xdata[~np.isnan(xdata)]
    
    hh1 = plt.hist2d(xdata, ydata, bins=100, weights=vol, cmap='ocean_r', vmin=10**(-5), **kwargs)
    plt.close()

    xdata = zero2Nan(xfields[1].data*xfields[1].region(reg)).flatten()
    ydata = zero2Nan(yfields[1].data*yfields[1].region(reg)).flatten()[~np.isnan(xdata)]
    vol = 100*xfields[1].grid('vol').flatten()[~np.isnan(xdata)]/np.nansum(xfields[1].grid('vol'))
    xdata = xdata[~np.isnan(xdata)]
    
    kwargs['range'] = ((hh1[1][0], hh1[1][-1]), (hh1[2][0], hh1[2][-1]))

    hh2 = plt.hist2d(xdata, ydata, bins=100, weights=vol, cmap='ocean_r', vmin=10**(-5), **kwargs)
    plt.close()
    kwargs.pop('range')

    fig = plt.figure(figsize=(16, 8))
    plt.rcParams.update({'font.size': 22})
    h = symlog10(zero2Nan(hh1[0]-hh2[0])).T
    vmax = np.max((np.nanpercentile(h, 99), -np.nanpercentile(h, 1)))
    plt.pcolormesh(hh2[1], hh2[2], h, cmap='seismic', vmin=-1.5*vmax, vmax=1.5*vmax, **kwargs)
    cbar = plt.colorbar(extend='both')
    cbar.locator = ticker.AutoLocator()
    cbar.set_ticks(cbar.locator.tick_values(-vmax, vmax))
    cbar.set_ticklabels([r'$'+str(int(np.sign(exp)*10))+'^{'+str(np.round(abs(exp),2))+r'}$' for exp in cbar.get_ticks()])
    cbar.set_label('Difference in % Arctic Ocean Volume')
    if contour is not None:
        X, Y = np.meshgrid(hh2[1], hh2[2])
        Z = contour['func'](X, Y, **contour['args'])
        cs = plt.contour(X, Y, Z, levels=6, colors='black')
        plt.clabel(cs, levels=cs.levels[2:], inline=True, rightside_up=False, fmt=lambda lev : rf'{lev:.0f} kg/m$^3$')
    plt.xlabel(f'{xfields[0].name} ({xfields[0].units})')
    plt.ylabel(f'{yfields[0].name} ({yfields[0].units})')
    plt.title(f'{yfields[0].long_name.replace(yfields[0].name, "")} - {yfields[1].long_name} and {xfields[0].name} {reg} Volume Census')
    plt.tight_layout()
        
    if show:
        plt.show()
    else:
        plt.close()
    return fig

    

