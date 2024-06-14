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
    
    Methods
    -------
    getName(long=False):
        Returns the field's (long) name
    getData(maskas=None):
        Returns the field's (masked) data
    getUnits():
        Returns the field's units
    getGrid(key):
        Returns the value of 'key' in the field's grid
    getRegion(region):
        Returns the logical of 'region'
    visualize_regional_profiles(scale=None, show=False):
        Returns a figure with the field's vertical average profile in each region
    visualize_distributions(scale=None, show=False):
        Returns a figure with histograms of the field in each region for all depths
    visualize_maps(scale=None, show=False):
        Returns a figure with maps of the field for all depths
    """
    def __init__(self, desc, name, units, grid, data, reg_logic=None):
        self.__name = name
        self.__long_name = desc + name
        self.__data = data
        self.__units = units
        self.__grid = grid
        if reg_logic is None:
            self.__reg_logic = makeRegions(grid)
        else:
            self.__reg_logic = reg_logic
        
        self.__means = None
        self.__weighted_means = None
        self.__geo_means = None
        self.__weighted_geo_means = None
        
    def getMeans(self, weighted, geom, alt_data=None):
        if weighted:
            weights = self.__grid['vol']
            if geom:
                ref = self.__weighted_geo_means
            else:
                ref = self.__weighted_means
        else:
            weights = None
            if geom:
                ref = self.__geo_means
            else:
                ref = self.__means
        if alt_data is None:
            if ref is not None:
                return ref
            else:
                data = self.__data
        else:
            data = alt_data
        means = {}
        for region in ['ARC', 'CB', 'EB', 'basin', 'shelf', 'slope']:
            if region == 'basin':
                means['basin'] = compute_mean(zero2Nan(data*(self.__reg_logic['CB']+self.__reg_logic['EB'])), w=weights)
            else:
                means[region] = compute_mean(zero2Nan(data*self.__reg_logic[region]), w=weights)
        if alt_data is None:
            ref = means
        return means
        
    #def __computeWeightedMeans(self, alt_data=None):
    #    if alt_data is None:
    #        if self.__weighted_means is not None:
    #            return self.__weighted_means
    #        else:
    #            data = self.__data
    #    else:
    #        data = alt_data
    #    weights = self.grid['vol']
    #    weighted_means = {}
    #    for key in ['ARC', 'CB', 'EB', 'basin', 'shelf', 'slope']:
    #        if key == 'basin':
    #            print(fr'Computing Basin Weighted Mean for {self.long_name}')
    #            weighted_means['basin'] = compute_mean(zero2Nan(data*(self.reg_logic['CB'] + self.reg_logic['EB'])), w=weights)
    #        else:
    #            print(fr'Computing {key} Weighted Mean for {self.long_name}')
    #            weighted_means[key] = compute_mean(zero2Nan(data*self.reg_logic[key]), w=weights)   
    #    if alt_data is None:
    #        self.weighted_means = weighted_means
    #    return weighted_means
    #
    #def __computeMeans(self, alt_data=None):
    #    if alt_data is None:
    #        if self.means is not None:
    #            return self.means
    #        else:
    #            data = self.data
    #    else:
    #        data = alt_data
    #    means = {}
    #    for key in ['ARC', 'CB', 'EB', 'basin', 'shelf', 'slope']:
    #        if key == 'basin':
    #            print(fr'Computing Basin Mean for {self.long_name}')
    #            means['basin'] = compute_mean(zero2Nan(data*(self.reg_logic['CB'] + self.reg_logic['EB'])))
    #        else:
    #            print(fr'Computing {key} Mean for {self.long_name}')
    #            means[key] = compute_mean(zero2Nan(data*self.reg_logic[key]))
    #    if alt_data is None:
    #        self.means = means
    #    return means
    
    def getName(self, long=False):
        if long:
            return self.__long_name
        return self.__name
    
    def getData(self, maskas=None):
        cpy = np.copy(self.__data)
        if maskas:
            cpy[self.__grid['ocnmsk'] == 0] = maskas
        return cpy
    
    def getUnits(self):
        return self.__units
    
    def getGrid(self, key):
        if key in self.__grid.keys():
            return np.copy(self.__grid[key])
        else:
            raise KeyError(f"{key} is not a valid grid attribute. Choose from '{k for k in self.__grid.keys()}'.")
    
    def getRegions(self, region):
        return dict(self.__reg_logic)

    def visualize_regional_profile(self, weighted=True, geom=False, scale=None, show=False):
        data = self.__data
        means = self.__computeMeans(weighted, geom)
        z_means = {}
        vec = np.ones((50,1))
        fig = plt.figure(figsize=(8, 10))
        plt.rcParams.update({'font.size': 22})
        colors = plt.get_cmap('tab10')(np.arange(10, dtype=int))
    
        for i, k in enumerate(means.keys()):
            if k in self.__reg_logic.keys():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    z_means[k] = np.squeeze(np.nanmean(zero2Nan(data*self.__reg_logic[k]), axis=(0,1)))
                    plt.plot(z_means[k], self.__grid['zcell'], color=colors[i], marker='o', label=k)
                    plt.plot(means[k]*vec, self.__grid['zcell'], color=colors[i], linestyle='-.')
    
        plt.ylabel('Depth (m)')
        plt.yscale('symlog')
        plt.xlabel(f'{self.__name} ({self.__units})')
        if scale == 'symlog':
            plt.xscale('symlog')
        elif scale == 'log':
            plt.xscale('log')
    
        pos = plt.gca().get_position()
        plt.gca().set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        plt.gca().legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    
        plt.title(f'{self.__long_name} Average Profile')
        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.close()
        return fig
        
    def visualize_distributions(self, scale=None, show=False):
        data = self.__data
        (m, n, p) = np.shape(data)
        fig, axes = plt.subplots(int(p/4), 4, figsize=(12, 3*int(p/4)))
        plt.rcParams.update({'font.size': 22})
        colors = plt.get_cmap('tab10')(np.arange(10, dtype=int))
        for i, depth in enumerate(self.__grid['zcell']):
            if i < 4*int(p/4):
                ax = axes[int(i/4), int(i%4)]
                data_slice = data[:,:,i]
                for j, key in enumerate(['CB', 'EB', 'shelf', 'slope']):
                    msk = (np.isnan(data_slice) == False) & (self.__reg_logic[key][:,:,i] == True)
                    x = data_slice[msk]
                    if scale == 'symlog':
                        x = np.symlog10(zero2Nan(x))
                    elif scale == 'log':
                        x = np.log10(zero2Nan(x))
                    if len(x) != 0:
                        try:
                            n_bin = int((np.nanmax(x) - np.nanmin(x))/0.2)+1
                            ax.hist(x, bins=n_bin, histtype='stepfilled', label=key, ec=colors[j], 
                                    fc=np.append(colors[j][:-1], 0.3))
                        except:
                            n_bin = 0
                ax.set_title(f'z = {np.abs(depth):.3}')
                ax.set_xlabel(f'{self.__name} ({self.__units})')
                
        axes[0,3].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

        st = plt.suptitle(f'{self.__long_name} Distribution by Depth')
    
        st.set_y(1)
        fig.subplots_adjust(top=0.85)
        plt.tight_layout()
                
        if show:
            plt.show()
        else:
            plt.close()
        return fig

    def visualize_maps(self, scale=None, show=False):
        data = self.__data
        data[self.__grid['ocnmsk'] == 0] = np.nan
        (m, n, p) = np.shape(data)
        lons = self.__grid['lon']
        lats = self.__grid['lat']
    
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)

        fig, axes = plt.subplots(int(p/4), 4, 
                                 figsize=(20, 5*int(p/4)), subplot_kw={'projection': ccrs.NorthPolarStereo()})
        plt.rcParams.update({'font.size': 22})
        for i, depth in enumerate(self.__grid['zcell']):
            if i < 4*int(p/4):
                ax = axes[int(i/4), int(i%4)]
                data_slice = data[:,:,i]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if scale == 'symlog':
                        data_slice = np.symlog10(zero2Nan(data_slice))
                        cmap = plt.get_cmap('viridis')
                    elif scale == 'log':
                        data_slice = np.symlog10(zero2Nan(data_slice))
                        cmap = plt.get_cmap('coolwarm')
                    else:
                        cmap = plt.get_cmap('coolwarm')
                plt.set_cmap(cmap)
                ax.pcolormesh(lons, lats, data_slice, transform=ccrs.PlateCarree())
                ax.set_extent([-180, 180, 90, 60], ccrs.PlateCarree())
                ax.coastlines()
                ax.gridlines(lw=1, ls=':', draw_labels=True, rotate_labels=False, xlocs=np.linspace(-180, 180, 13), ylocs=[])
                ax.set_boundary(circle, transform=ax.transAxes)
                ax.set_title(f'z = {np.abs(depth):.3}')
            
        st = plt.suptitle(f'{self.__long_name} Arctic Map')

        st.set_y(1)
        fig.subplots_adjust(top=0.85)
        plt.tight_layout()
            
        if show:
            plt.show()
        else:
            plt.close()
        return fig
        
class KappaBG(Field):
    def __init__(self, name, grid, obs_data, reg_logic, ver='OBS'):
        super().__init__(ver, name, grid, obs_data, reg_logic=reg_logic)
        self.__data = self.__createKappa(ver)
        
        
    def __createKappa(self, ver='OBS'):
        (m, n, p) = np.shape(self.__data)
        kappa = np.zeros((m, n, p))
        ocnmsk = self.__grid['ocnmsk']
        weighted_means = self.__computeMeans(True, False)
        if ver == 'CTL':
            kappa += self.__reg_logic['ARC']*weighted_means['ARC']
        elif ver == 'HI':
            kappa += self.__reg_logic['ARC']*weighted_means['shelf']
        elif ver == 'LO':
            kappa += self.__reg_logic['ARC']*weighted_means['basin']
        elif ver == 'REG':
            kappa += self.__reg_logic['CB']*weighted_means['CB']
            kappa += self.__reg_logic['EB']*weighted_means['EB']
            kappa += self.__reg_logic['shelf']*weighted_means['shelf']
            kappa += self.__reg_logic['slope']*weighted_means['slope']
        else:
            return self.data
        kappa += self.__reg_logic['NAt']*6.6e-6
        kappa = zero2Nan(kappa)
        kappa[ocnmsk == 0] = np.nan
        if np.any(np.equal(ocnmsk > 0, np.isnan(kappa))):
            print('Error: there are ocean gridcells that do not contain a kappa value')
        return kappa

class Transect:
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
        
    def visualize_transect(self, Field, scale=None, show=False):
        depths = self.grid['zcell'][:-1]
        ocnmask = self.grid['ocnmsk']
        fig = plt.figure(figsize=(16, 8))
        plt.rcParams.update({'font.size': 22})
        
        trans_fld = np.ones((len(depths), self.node_number))*np.nan
        data = Field.getData(maskas=np.nan)
        for d in range(len(depths)):
            fld_slice = data[:,:,d]
            if scale=='symlog':
                fld_slice = np.symlog10(zero2Nan(fld_slice))
            if scale=='log':
                fld_slice = np.log10(zero2Nan(fld_slice))
            trans_fld[d,:] = fld_slice.flatten()[self.trans_idx]
        
        X, Y = np.meshgrid(self.trans_dist, depths)
        
        plt.gca().set_facecolor('dimgrey')
        if scale == 'log':
            cmap = 'viridis'
            lev = np.linspace(np.floor(np.nanmin(trans_fld)-1),np.ceil(np.nanmax(trans_fld)+1), 50)
        elif scale == 'symlog':
            cmap = 'coolwarm'
            lev = np.linspace(np.floor(np.nanmin(trans_fld)-1),np.ceil(np.nanmax(trans_fld)+1), 50)
        else:
            cmap = 'coolwarm'
            lev = np.linspace(-1, 1, 50)
        
        cs = plt.contourf(X, Y, trans_fld, lev, cmap=cmap, extend='both')
        ymin = depths[np.count_nonzero(np.nansum(trans_fld,1)) + 1] - 100
        ymax = -10
        plt.ylim((ymin, ymax))
        
        cbar = plt.colorbar(cs)
        cbar.locator = ticker.AutoLocator()
        cbar.set_ticks(cbar.locator.tick_values(lev[0], lev[-1]))
        if scale == 'symlog':
            cbar.set_ticklabels([r'$'+str(int(np.sign(exp)*10))+'^{'+str(int(abs(exp)))+r'}$' for exp in cbar.get_ticks()])
        elif scale == 'log':
            cbar.set_ticklabels([r'$10^{'+str(int(exp))+r'}$' for exp in cbar.get_ticks()])
        cbar.minorticks_off()
        cbar.set_label(Field.getName())
        plt.xlabel('Along-Track Distance (km)')
        plt.ylabel('Depth (m)')
        plt.title(Field.getName(long=True) + ' along ' + self.name)
        plt.tight_layout()
        
        if show:
            plt.show()
        else:
            plt.close()
        return fig
    
    def visualize_line(self, show=False):
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

def visualize_average_profiles(fields, weighted=True, geom=False, scale=None, show=False):
        z_means = {}
        w_means = {}
        for field in fields:
            data = field.getData(maskas=np.nan)
            z_means[field.getName(long=True)] = {}
            w_means[field.getName(long=True)] = field.getMeans(weighted, geom)
            for key in field.getMeans(weighted, geom).keys():
                if key in field.getRegions.keys():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        z_means[field.long_name][key] = np.squeeze(np.nanmean(zero2Nan(data*field.getRegions[key]), axis=(0,1)))
        fig, axes = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(30,10))
        vec = np.ones((50,1))
        plt.rcParams.update({'font.size': 22})
        colors = plt.get_cmap('tab10')(np.arange(10, dtype=int))
        for i, key in enumerate(z_means[list(z_means.keys())[0]].keys()):
            for j, field in enumerate(fields):
                axes[i].plot(z_means[field.getName(long=True)][key], field.getGrid('zcell'), color=colors[j], marker='o', label=field.getName(long=True))
                axes[i].plot(w_means[field.getName(long=True)][key]*vec, field.getGrid('zcell'), color=colors[j], linestyle='-.')
                axes[i].set_title(key)
                axes[i].set_yscale('symlog')
                if scale == 'symlog':
                    axes[i].set_xscale('symlog')
                elif scale == 'log':
                    axes[i].set_xscale('log')
        
        axes[-1].legend(loc='center right', bbox_to_anchor=(1.8, 0.5))
        plt.xlabel(fields[0].getName())
        axes[0].set_ylabel('Depth (m)')
        st = plt.suptitle(f'{fields[0].getName()} Average Profiles')
        st.set_y(1)
        fig.subplots_adjust(top=0.85)
        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.close()
        return fig