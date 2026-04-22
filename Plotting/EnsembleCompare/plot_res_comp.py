from plot_ensemble import glosat_ensemble_analysis
import glob
import xarray as xr
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
import numpy as np
from scipy.interpolate import griddata

class NEMO_case(object):

    def __init__(self, case, dom_cfg=None, zcoord="MES"):
        root = "/gws/ssde/j25a/verify_oce/NEMO/"
        self.nemo_path = root + "Outputs/" 
        self.case_path = self.nemo_path + case
        if dom_cfg:
            self.dom_path = root + "Preprocessing/DOM/NAARC/" + dom_cfg
        self.save_path = root + f"PostProcessing/NAARC/{zcoord}/"


    def calc_barotropic_stream_function(self, y0, y1):
        """
        calcualte BSF for NEMO output
        """

        gea = glosat_ensemble_analysis()

        y_set = []
        for y in range(y0, y1):
            print (y)
            paths = glob.glob(self.case_path + "/" + str(y) +
                             f"/*/*1m*grid_V.nc")

            vvel_series = gea.get_mfda(paths, "vo")
            e3v_series = gea.get_mfda(paths, "thkcello")

            depth_weight_vvel_series = vvel_series * e3v_series

            with ProgressBar():
                baroV = depth_weight_vvel_series.sum(dim="depthv").load()
            #baroV_mean = baroV.mean("time_centered")
            #baroV = baroV.expand_dims(year=[y])
            y_set.append(baroV)

            baroV = xr.concat(y_set, "time_centered")

        # get depth integrated velocities
        domcfg = xr.open_dataset(self.dom_path, chunks="auto").squeeze()
        for coord in ["x", "y", "yy"]:
            if coord in domcfg.coords.keys():
                domcfg = domcfg.drop_vars(coord)
        e1v = domcfg.e1v
        e1v = e1v.assign_coords({"nav_lon":domcfg.glamv,
                                 "nav_lat":domcfg.gphiv})
        print (e1v)

        baroV = baroV.stack(a=["y","x"])
        e1v = e1v.stack(a=["y","x"])

        baroV, e1v = xr.align(baroV, e1v)

        e1v = e1v.unstack("a")
        baroV = baroV.unstack("a")

        transport = ( baroV * e1v ).sortby("x",
                                     ascending=False).cumsum(dim="x")/1e6

        BSF_masked = transport.sortby("x").where(domcfg.top_level == 1)

        BSF_masked = BSF_masked.assign_coords({"nav_lon":domcfg.glamv,
                                               "nav_lat":domcfg.gphiv})

        BSF_masked.name = "BSF"
        BSF_masked.to_netcdf(self.save_path + 
                             f"BSF_{y0}_{y1-1}.nc")
        
    def get_barotropic_stream_function(self, y0, y1):
        """ access saved bsf """
        self.bsf = xr.open_dataarray(self.save_path + 
                             f"BSF_{y0}_{y1-1}.nc")

    def get_density_snapshot(self, yyyy, mm):
        """ access density """
        self.rho = xr.open_dataset(self.case_path +
              f"/{yyyy}/{mm}/VERIFY_1m_{yyyy}{mm}01_{yyyy}{mm}30_grid_T.nc",
                chunks="auto").rhop

    def interpolate_to_pts(self, da, tgt_lons, tgt_lats):
        """ interpolate to section """

        domcfg = xr.open_dataset(self.dom_path, chunks="auto").squeeze()
        domcfg = domcfg.drop_vars("x")

        lons = da.nav_lon.load()
        lats = da.nav_lat.load()
        print (tgt_lons)

        da = da.where((lons > tgt_lons[0]) &
                      (lons < tgt_lons[-1]) &
                      (lats > tgt_lats[0]) &
                      (lats < tgt_lats[-1]), drop=True)

        domcfg = domcfg.where((lons > tgt_lons[0]) &
                              (lons < tgt_lons[-1]) &
                              (lats > tgt_lats[0]) &
                              (lats < tgt_lats[-1]), drop=True)

        target = (tgt_lons, tgt_lats)

        src_mdep = np.nan_to_num(domcfg.gdept_0.stack(z=("x","y","nav_lev")).values, nan=-9999)
        src_mlon = da.nav_lon.broadcast_like(domcfg.gdept_0).values
        src_mlat = da.nav_lat.broadcast_like(domcfg.gdept_0).values

        points = (src_mlon.flatten(), src_mlat.flatten(), src_mdep)

        values = da.values.flatten()
        values_masked = (np.nan_to_num(values))
        print (points[0].shape)
        print (values_masked.shape)
            
        n_grid_all = griddata(points, values_masked, target,
                              method="linear")[:,:,np.newaxis]

        section = xr.DataArray(
                             data=n_grid_all,
                             dims=["d","time"],
                             coords={"longitude": (["d"],tgt_lon.values),
                                     "latitude": (["d"], tgt_lat.values),
                                     "time": self.da.time},
                             name=self.var_str)
        return section

    def extract_section(self, da, section="denmark_strait", res=1/12):

        if section == "denmark_strait":
            lat0, lon0 = 63.5, -31
            lat1, lon1 = 66.5, -26


            #ind0 = (ds.nav_lon - lon0).argmin("x")
            #ind0 = (ds.nav_lat[ind0] - lat0).argmin("y")

            #ind1 = (ds.nav_lon - lon1).argmin("x")
            #ind1 = (ds.nav_lat[ind1] - lat1).argmin("y")

            x_pts = np.linspace(lon0, lon1, int(abs(lon1-lon0) / res))
            y_pts = np.linspace(lat0, lat1, int(abs(lat1-lat0) / res))

            ds_den_str = self.interpolate_to_pts(da, x_pts, y_pts)

        else:
            # TODO should be an exception that is raied
            print ("error - section not implemented")

        
class NEMO_compare(object):
    """
    """

    def __init__(self, case_dict):
        self.nemo_path = "/gws/ssde/j25a/verify_oce/NEMO/Outputs/"
        self.mes_case = "EXP_mes_LSM_new_radiation/"
        self.zlevel_case = "EXP_zlevel_LSM_new_radiation/"

        self.cases = {}
        for i in range(len(case_dict)):
            self.cases = {f"case{i}": NEMO_case(case_dict[0]["case"],
                            dom_cfg="domain_cfg_mes.nc")}
    
    def get_glosat_var(self, y, var, grid_str):
    
        gea = glosat_ensemble_analysis()
    
        paths0 = glob.glob(gea.glosat_path + gea.ensemble_list[0] +
                           "/" + str(y) + f"*/*1m_{y}*grid-{grid_str}.nc")
        paths1 = glob.glob(gea.glosat_path + gea.ensemble_list[0] +
                           "/" + str(y+1) + f"*/*1m_{y}*grid-{grid_str}.nc")
        paths = paths0 + paths1
    
        da = gea.get_da(paths, var)

    def get_nemo(self, fn, var):

        self.mes = xr.open_dataset(self.nemo_path + self.mes_case + fn,
                                   chunks="auto")[var]

        self.zlevel = xr.open_dataset(self.nemo_path + self.zlevel_case + fn,
                                      chunks="auto")[var]

    def plot_nemo(self):
        fig, axs = plt.subplots(3, figsize=(5,12))

        self.mes.plot(ax=axs[0], vmin=-1, vmax=18)
        self.zlevel.plot(ax=axs[1], vmin=-1, vmax=18)

        diff = self.mes - self.zlevel

        diff.plot(ax=axs[2], vmin=-2, vmax=2)
        plt.show()

    def plot_bsf_timeseries(self, y0, y1):
        """
        compare barotropic streamfunction timeseries for multiple cases
        """

        # initialise figure
        fig, axs = plt.subplots(3, figsize=(5,12))

        # get streamfunctions
        print (self.cases)
        for i in range(len(self.cases)):
             self.cases[f"case{i}"].get_barotropic_stream_function(y0, y1)

        axs[0].pcolormesh(self.cases["case0"].bsf.isel(year=-1).T)
        plt.show()


    def plot_denmark_strait(self):
        """
        plot density section for denmark strait
        """

        # initialise figure
        fig, axs = plt.subplots(3, figsize=(6.5,4))

        # get density for three models
        for i in range(len(self.cases)):
             mod = self.cases[f"case{i}"]
             mod.get_density_snapshot(1858,12)

             # extract section
             sec = mod.extract_section(mod.rho)

        # plot section

#case = NEMO_case("EXP_mes_LSM_new_radiation", "domain_cfg_mes.nc")
#case.calc_barotropic_stream_function(1850, 1858)
case = NEMO_case("EXP_zlevel_LSM_new_radiation", "domain_cfg_zps.nc",
        zcoord="ZPS")
case.calc_barotropic_stream_function(1850, 1860)
#case_dict = [{"case": "EXP_mes_LSM_new_radiation"}]
#nemo_comp = NEMO_compare(case_dict)
#nemo_comp.plot_denmark_strait()

#case_dict = [{"case": "EXP_mes_LSM_new_radiation"}]
#comp = NEMO_compare(case_dict, 1850, 1854)
#comp.plot_bsf_timeseries()


#nemo_comp.get_nemo("1854/12/VERIFY_1m_18541201_18541230_grid_T.nc", "tos_con")
#nemo_comp.plot_nemo()

