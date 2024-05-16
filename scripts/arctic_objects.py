import numpy as np
import cartopy.crs as ccrs
from arctic_functions import *
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.spatial import KDTree
import warnings

class Field:
    def __init__(self, name, mgrid, data):
        self.name = name
        self.data = data
        self.mgrid = mgrid

    def visualize_maps(self, show=False):
        data = self.data
        (m, n, p) = np.shape(data)
        lons = self.mgrid['lon'][0,0]
        lats = self.mgrid['lat'][0,0]
    
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)

        fig, axes = plt.subplots(int(p/4), 4, figsize=(16, 3*int(p/4)), subplot_kw={'projection': ccrs.NorthPolarStereo()})
        for i, depth in enumerate(self.mgrid['zcell'][0,0]):
            if i < 4*int(p/4):
                ax = axes[int(i/4), int(i%4)]
                data_slice = data[:,:,i]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ax.pcolormesh(lons, lats, np.log10(data_slice), transform=ccrs.PlateCarree())
                ax.set_extent([-180, 180, 90, 60], ccrs.PlateCarree())
                ax.coastlines()
                ax.gridlines(lw=1, ls=':', draw_labels=True, rotate_labels=False, xlocs=np.linspace(-180, 180, 13), ylocs=[])
                ax.set_boundary(circle, transform=ax.transAxes)
                ax.set_title(f'z = {np.abs(depth[0]):.3}')
            
        st = plt.suptitle(f'{self.name} Arctic Map')
    
        st.set_y(1)
        fig.subplots_adjust(top=0.85)
            
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        return fig
class RegionalField(Field):
    def __init__(self, name, mgrid, reg_logic, data):
        super().__init__(name, mgrid, data)
        self.reg_logic = reg_logic
        
        self.means = None
        self.weighted_means = None
        
    def getWeightedMeans(self, alt_data=None):
        if alt_data is None:
            if self.weighted_means is not None:
                return self.weighted_means
            else:
                data = self.data
        else:
            data = alt_data
        weights = self.mgrid['vol'][0,0]
        weighted_means = {}
        weighted_means['ARC'] = compute_mean(zero2Nan(data*self.reg_logic['ARC']), w=weights)
        weighted_means['CB'] = compute_mean(zero2Nan(data*self.reg_logic['CB']), w=weights)
        weighted_means['EB'] = compute_mean(zero2Nan(data*self.reg_logic['EB']), w=weights)
        weighted_means['basin'] = compute_mean(zero2Nan(data*(self.reg_logic['CB'] + self.reg_logic['EB'])), w=weights)
        weighted_means['shelf'] = compute_mean(zero2Nan(data*self.reg_logic['shelf']), w=weights)
        weighted_means['slope'] = compute_mean(zero2Nan(data*self.reg_logic['slope']), w=weights)
        
        if alt_data is None:
            self.weighted_means = weighted_means
        return weighted_means
    
    def getMeans(self, alt_data=None):
        if alt_data is None:
            if self.means is not None:
                return self.means
            else:
                data = self.data
        else:
            data = alt_data
        means = {}
        means['ARC'] = compute_mean(zero2Nan(data*self.reg_logic['ARC']))
        means['CB'] = compute_mean(zero2Nan(data*self.reg_logic['CB']))
        means['EB'] = compute_mean(zero2Nan(data*self.reg_logic['EB']))
        means['basin'] = compute_mean(zero2Nan(data*(self.reg_logic['CB'] + self.reg_logic['EB'])))
        means['shelf'] = compute_mean(zero2Nan(data*self.reg_logic['shelf']))
        means['slope'] = compute_mean(zero2Nan(data*self.reg_logic['slope']))
        
        if alt_data is None:
            self.means = means
        return means

    def visualize_average_profiles(self, log=True, show=False):
        data = self.data
        weighted_means = self.getWeightedMeans(data)
        z_means = {}
        vec = np.ones((50,1))
        fig = plt.figure(figsize=(8, 10))
    
        colors = plt.get_cmap('tab10')(np.arange(10, dtype=int))
    
        for i, k in enumerate(weighted_means.keys()):
            if k in self.reg_logic.keys():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    z_means[k] = np.squeeze(np.nanmean(zero2Nan(data*self.reg_logic[k]), axis=(0,1)))
                    plt.plot(z_means[k], self.mgrid['Depth'][0,0], color=colors[i], marker='o', label=k)
                    plt.plot(weighted_means[k]*vec, self.mgrid['Depth'][0,0], color=colors[i], linestyle='-.')
    
        plt.ylabel('Depth (m)')
        plt.yscale('symlog')
        plt.xlabel(self.name)
        if log:
            plt.xscale('log')
    
        pos = plt.gca().get_position()
        plt.gca().set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        plt.gca().legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    
        plt.title(f'{self.name} Average Profile')

        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        return fig
        
    def visualize_distributions(self, show=False):
        data = self.data
        (m, n, p) = np.shape(data)
        fig, axes = plt.subplots(int(p/4), 4, figsize=(12, 3*int(p/4)))
        colors = plt.get_cmap('tab10')(np.arange(10, dtype=int))
        for i, depth in enumerate(self.mgrid['zcell'][0,0]):
            if i < 4*int(p/4):
                ax = axes[int(i/4), int(i%4)]
                data_slice = data[:,:,i]
                for j, key in enumerate(['CB', 'EB', 'shelf', 'slope']):
                    msk = (np.isnan(data_slice) == False) & (self.reg_logic[key][:,:,i] == True)
                    x = np.log10(data_slice[msk])
                    if len(x) != 0: 
                        n_bin = int((np.max(x) - np.min(x))/0.2)+1
                        ax.hist(x, bins=n_bin, histtype='stepfilled', label=key, ec=colors[j], fc=np.append(colors[j][:-1], 0.3))
                ax.set_title(f'z = {np.abs(depth[0]):.3}')
                ax.set_xlabel(self.name)
    
        axes[0,3].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

        st = plt.suptitle(f'{self.name} Distribution by Depth')
    
        st.set_y(1)
        fig.subplots_adjust(top=0.85)
                
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        return fig
        
class KappaBG(RegionalField):
    def __init__(self, name, mgrid, reg_logic, obs_data, ver='OBS'):
        super().__init__(name, mgrid, reg_logic, obs_data)
        self.data = self.__createKappa(ver)
        
    def __createKappa(self, ver='OBS'):
        (m, n, p) = np.shape(self.data)
        kappa = np.zeros((m, n, p))
        ocnmsk = self.mgrid['ocnmsk'][0,0]
        weighted_means = self.getWeightedMeans()
        if ver == 'CTL':
            kappa += self.reg_logic['ARC']*weighted_means['ARC']
        elif ver == 'HI':
            kappa += self.reg_logic['ARC']*weighted_means['shelf']
        elif ver == 'LO':
            kappa += self.reg_logic['ARC']*weighted_means['basin']
        elif ver == 'REG':
            kappa += self.reg_logic['CB']*weighted_means['CB']
            kappa += self.reg_logic['EB']*weighted_means['EB']
            kappa += self.reg_logic['shelf']*weighted_means['shelf']
            kappa += self.reg_logic['slope']*weighted_means['slope']
        else:
            return self.data
        kappa += self.reg_logic['NAt']*6.6e-6
        kappa = zero2Nan(kappa)
        kappa[ocnmsk == 0] = np.nan
        if np.any(np.equal(ocnmsk > 0, np.isnan(kappa))):
            print('Error: there are ocean gridcells that do not contain a kappa value')
        return kappa

class Transect:
    def __init__(self, mgrid, p1, p2, proj=ccrs.NorthPolarStereo()):
        self.proj = proj
        self.base = ccrs.PlateCarree()
        self.__initializeCoords(p1, p2)
        self.node_spacing = 35.849468668883645
        self.node_number = int(np.ceil(euclideanDist(np.array([self.coords['x_s'], self.coords['y_s']]),
                                                     np.array([self.coords['x_t'], self.coords['y_t']]))/self.node_spacing))
        self.mgrid = mgrid
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
        
        lons = self.mgrid['lon'][0,0].flatten()
        lats = self.mgrid['lat'][0,0].flatten()
        trans_array = self.base.transform_points(self.proj, self.line[:,0], self.line[:,1])[:,:-1]
        grid_array = np.stack((lons, lats), axis=-1)
        _, trans_idx = KDTree(grid_array).query(trans_array)
        self.trans_idx = trans_idx
        
        trans_dist = np.ones((self.node_number, 1))*np.nan
        trans_dist[0] = 0
        for i in range(1, self.node_number):
            trans_dist[i] = trans_dist[i-1] + haversineDist(trans_array[i-1], trans_array[i])
        self.trans_dist = trans_dist
        
    def visualize_transect(self, Field, log=True, show=False):
        depths = np.squeeze(self.mgrid['zcell'][0,0])[:-1]
        ocnmask = self.mgrid['ocnmsk'][0,0]
        fig = plt.figure(figsize=(12, 5))
        
        trans_fld = np.ones((len(depths), self.node_number))*np.nan
        for d in range(len(depths)):
            fld_slice = Field.data[:,:,d]
            if log:
                fld_slice = np.log10(fld_slice)
            fld_slice[ocnmask[:,:,d] == 0] = np.nan
            trans_fld[d,:] = fld_slice.flatten()[self.trans_idx]
        
        X, Y = np.meshgrid(self.trans_dist, depths)
        
        plt.gca().set_facecolor('dimgrey')
        plt.contourf(X, Y, trans_fld)
        ymin = depths[np.count_nonzero(np.nansum(trans_fld,1)) + 1] - 100
        ymax = -10
        plt.ylim((ymin, ymax))
        
        plt.colorbar()
        plt.xlabel('Along-Track Distance (km)')
        plt.ylabel('Depth (m)')
        plt.yscale('symlog')
        
        if show:
            plt.tight_layout()
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
        
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        return fig