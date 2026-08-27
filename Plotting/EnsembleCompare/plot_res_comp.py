from Plotting.EnsembleCompare.plot_ensemble import glosat_ensemble_analysis
import glob
import xarray as xr
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
import numpy as np
from scipy.interpolate import griddata

class NEMO_case(object):

    def __init__(self, case, dom_cfg=None, zcoord="MES", label=""):
        self.case_name = case
        self.zcoord = zcoord
        self.label = label
        self.root = "/gws/ssde/j25a/verify_oce/NEMO/"
        self.nemo_path = self.root + "Outputs/Historical/" 
        self.case_path = self.nemo_path + case
        if dom_cfg:
            self.dom_path = self.root + "Preprocessing/DOM/NAARC/" + dom_cfg
        self.save_path = self.root + f"PostProcessing/NAARC/{case}/"


    def get_paths(self, y, freq="1m", grid="grid_T"):

        paths = glob.glob(self.case_path + "/" + str(y) +
                         f"/*/*{freq}*{grid}.nc")
        print (paths)
        print (self.case_path)
        
        return paths

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
        return self.bsf

    def get_tos(self, y0, y1):
        self.tos = xr.open_dataarray(self.save_path + 
                             f"naarc_SPG_tos_{y0}_{y1}.nc")
        return self.tos

    def get_density_snapshot(self, yyyy, mm):
        """ access density """
        self.rho = xr.open_dataset(self.case_path +
              f"/{yyyy}/{mm}/VERIFY_1m_{yyyy}{mm}01_{yyyy}{mm}30_grid_T.nc",
                chunks="auto").rhop

    def restrict_to_SPG(self, da):
        lat_lims = [45,65]
        lon_lims = [-60,10]

        # restrict to area
        da = da.where((da.nav_lon > lon_lims[0]) &
                      (da.nav_lon < lon_lims[1]) &
                      (da.nav_lat > lat_lims[0]) &
                      (da.nav_lat < lat_lims[1]), drop=False)
        #da = da.isel(x=slice(1,-1), y=slice(None,-1))

        return da

    def area_mean(self, da, weights):

        da = self.restrict_to_SPG(da)

        # area weighted mean
        da = da.weighted(weights).mean(["x","y"])

        return da

    def calc_SPG_temperature_naarc(self, y0, y1):
        """ get 2d spg tos """

        # get area
        #case = NEMO_case(case=case_dict["case"],
                         #dom_cfg=case_dict["domcfg"],
                         #zcoord=case_dict["zcoord"])
        domcfg = xr.open_dataset(self.dom_path, chunks="auto").squeeze()
        area = (domcfg.e1f * domcfg.e2f)

        y_set = []
        for y in range(y0, y1):
            print (y)
            year_paths = self.get_paths(y)
            m_set = []
            for path in year_paths:
                print (path)
                month_da = xr.open_dataset(path, chunks="auto").thetao_con
                top_month_da = month_da.isel(deptht=0)
                m_set.append(self.restrict_to_SPG(top_month_da))
            temp_series_year = xr.concat(m_set, "time_counter")

            y_set.append(temp_series_year)

        temp_series_full = xr.concat(y_set, "time_counter")
        temp_series_full = temp_series_full.assign_attrs(
                   {"case":self.case_name})

        fn = self.save_path + f"naarc_SPG_tos_2d_{y0}_{y1}.nc"
        temp_series_full.to_netcdf(fn)

    def interpolate_to_pts(self, da, tgt_lons, tgt_lats):
        """ interpolate to section """

        domcfg = xr.open_dataset(self.dom_path, chunks="auto").squeeze()
        domcfg = domcfg.drop_vars("x")

        lons = da.nav_lon.load()
        lats = da.nav_lat.load()

        # restrict to area of section
        with ProgressBar():
            da = da.where((lons > tgt_lons[0]) &
                          (lons < tgt_lons[-1]) &
                          (lats > tgt_lats[0]) &
                          (lats < tgt_lats[-1]), drop=True).load()

            domcfg = domcfg.where((lons > tgt_lons[0]) &
                                  (lons < tgt_lons[-1]) &
                                  (lats > tgt_lats[0]) &
                                  (lats < tgt_lats[-1]), drop=True).load()


        tgt_deps = domcfg.gdept_1d.values
        print (domcfg.data_vars)
        print (domcfg)
        print (asdkf)
        print (tgt_lons.shape)
        print (tgt_lats.shape)
        print (tgt_deps.shape)
        tgt_mlons, _ = np.meshgrid(tgt_lons, tgt_deps)
        tgt_mlats, tgt_mdeps = np.meshgrid(tgt_lats, tgt_deps)
        print (tgt_mlons.shape)
        print (tgt_mlats.shape)
        print (tgt_mdeps.shape)
        target = (tgt_mlons, tgt_mlats, tgt_mdeps)

        src_mdep = np.nan_to_num(
                 domcfg.gdept_0.stack(z=("x","y","nav_lev")).values, nan=-9999)
        src_mlon = da.nav_lon.broadcast_like(domcfg.gdept_0).values
        src_mlat = da.nav_lat.broadcast_like(domcfg.gdept_0).values

        points = (src_mlon.flatten(), src_mlat.flatten(), src_mdep)

        values = da.values.flatten()
        values_masked = (np.nan_to_num(values))
        print (points[0].shape)
        print (values_masked.shape)
            
        n_grid_all = griddata(points, values_masked, target,
                              method="linear")

        section = xr.DataArray(
                             data=n_grid_all,
                             dims=["d","depth"],
                             coords={"longitude": (["d"],tgt_lon.values),
                                     "latitude": (["d"], tgt_lat.values),
                                     "depth": tgt_deps.values},
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
        self.root = "/gws/ssde/j25a/verify_oce/NEMO/"
        self.nemo_path = "/gws/ssde/j25a/verify_oce/NEMO/Outputs/"
        self.mes_case = "EXP_mes_LSM_new_radiation/"
        self.zlevel_case = "EXP_zlevel_LSM_new_radiation/"

        self.cases = {}
        for i in range(len(case_dict)):
            self.cases[f"case{i}"] = NEMO_case(case_dict[i]["case"],
                                       dom_cfg=case_dict[i]["dom_cfg"],
                                       zcoord=case_dict[i]["zcoord"],
                                       label=case_dict[i]["label"])
            self.cases[f"case{i}"].y0 = case_dict[i]["y0"]
            self.cases[f"case{i}"].y1 = case_dict[i]["y1"]
    
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

    def plot_bsf_timeseries(self, y0, y1, add_glosat=True):
        """
        compare barotropic streamfunction timeseries for multiple cases
        """

        # initialise figure
        fig, ax = plt.subplots(1, figsize=(5,3))
        plt.subplots_adjust(bottom=0.2, top=0.98)
        
        gea = glosat_ensemble_analysis()

        # get streamfunctions
        print (self.cases)
        for i in range(len(self.cases)):
             bsf = self.cases[f"case{i}"].get_barotropic_stream_function(y0, y1)
             bsf = bsf.convert_calendar(calendar='gregorian',
                     dim="time_centered", align_on='date')
             bsf_na = gea.restrict_to_NA(bsf, domain="ocean")
             SPG = bsf_na.max(["x","y"])
             ax.plot(SPG.time_centered, SPG, label=self.cases[f"case{i}"].zcoord)
             #axs[i+1].pcolor(bsf.nav_lon, bsf.nav_lat, bsf)
        if add_glosat:
             path="/gws/ssde/j25a/verify_oce/NEMO/PostProcessing/GloSat/u-ck651/"
             bsf = xr.open_dataarray(path + "glosat_annual_mean_BSF_1850_2015.nc")
             bsf_na = gea.restrict_to_NA(bsf, domain="ocean")
             SPG = bsf_na.max(["x","y"])
             time_min = str(SPG.year.min().values)
             time_max = str(SPG.year.max().values)
             time_max = str(np.datetime64(time_max) + np.timedelta64(1,"Y"))
             print (time_max)
             
             time = np.arange(time_min, time_max, dtype="datetime64[Y]")
             ax.plot(time, SPG, label="GloSat")

        ax.legend()
        ax.set_ylabel("SPG strength (Sv)")
        ax.set_xlabel("Date")
        plt.savefig(self.root + "PostProcessing/Plots/bsf_comp.png", dpi=600)

        #axs[0].pcolormesh(self.cases["case0"].bsf.isel(time_centered=-1).T)
        #plt.show()

    def plot_tos_timeseries(self, y0, y1, add_glosat=True):
        """ """

        fig, ax = plt.subplots(1, figsize=(5,3))
        plt.subplots_adjust(bottom=0.2, top=0.98)

        for i in range(len(self.cases)):
             tos = self.cases[f"case{i}"].get_tos(y0, y1)
             tos = tos.convert_calendar(calendar='gregorian',
                     dim="time_counter", align_on='date')


             ax.plot(tos.time_counter, tos, label=self.cases[f"case{i}"].zcoord,
                     lw=0.5)

        if add_glosat:
             path="/gws/ssde/j25a/verify_oce/NEMO/PostProcessing/GloSat/u-ck651/"
             tos = xr.open_dataarray(path + "glosat_SPG_tos_1850_2015.nc")
             tos = tos.convert_calendar(calendar='gregorian',
                     dim="time_centered", align_on='date')

             ax.plot(tos.time_centered, tos, label="GloSat", lw=0.5)
        ax.legend()

        ax.set_ylabel("SPG Sea Surface Temperature")
        ax.set_xlabel("Date")
        plt.savefig(self.root + "PostProcessing/Plots/tos_comp.png", dpi=600)

    def plot_tos_bsf_timeseries(self, add_glosat=True):
        """ plot sea surface temperature and barotropic streamfunction """
        ### under construction 28 June 2026 ###
        ### note to RDP ###

        fig, axs = plt.subplots(2, figsize=(5.5,3))
        plt.subplots_adjust(bottom=0.15, top=0.98, right=0.95, hspace=0.1)

        for i in range(len(self.cases)):
             case = self.cases[f"case{i}"]
             y0 = case.y0
             y1 = case.y1
             tos = case.get_tos(y0, y1)
             tos = tos.convert_calendar(calendar='gregorian',
                     dim="time_counter", align_on='date')


             axs[0].plot(tos.time_counter, tos, label=self.cases[f"case{i}"].label,
                     lw=1.0)

        if add_glosat:
             path="/gws/ssde/j25a/verify_oce/NEMO/PostProcessing/GloSat/u-ck651/"
             tos = xr.open_dataarray(path + "glosat_SPG_tos_1850_2015.nc")
             tos = tos.convert_calendar(calendar='gregorian',
                     dim="time_centered", align_on='date')

             axs[0].plot(tos.time_centered, tos, label="GloSat", lw=1.0)

        axs[0].set_ylabel("SPG Sea Surface\nTemperature")
        
        gea = glosat_ensemble_analysis()

        # get streamfunctions
        print (self.cases)
        for i in range(len(self.cases)):
             case = self.cases[f"case{i}"]
             y0 = case.y0
             y1 = case.y1
             bsf = case.get_barotropic_stream_function(y0, y1)
             bsf = bsf.convert_calendar(calendar='gregorian',
                     dim="time_centered", align_on='date')
             bsf_na = gea.restrict_to_NA(bsf, domain="ocean")
             SPG = bsf_na.max(["x","y"])
             axs[1].plot(SPG.time_centered, SPG, label=case.label, lw=1.0)
        if add_glosat:
             path="/gws/ssde/j25a/verify_oce/NEMO/PostProcessing/GloSat/u-ck651/"
             bsf = xr.open_dataarray(path + "glosat_annual_mean_BSF_1850_2015.nc")
             bsf_na = gea.restrict_to_NA(bsf, domain="ocean")
             SPG = bsf_na.max(["x","y"])
             time_min = str(SPG.year.min().values)
             time_max = str(SPG.year.max().values)
             time_max = str(np.datetime64(time_max) + np.timedelta64(1,"Y"))
             print (time_max)
             
             time = np.arange(time_min, time_max, dtype="datetime64[Y]")
             axs[1].plot(time, SPG, label="GloSat", lw=1.0)

        axs[1].legend(loc="upper left")
        axs[1].set_ylabel("SPG strength\n(Sv)")
        axs[1].set_xlabel("Date")
        axs[0].set_xticklabels([])
        for ax in axs:
            ax.set_xlim(np.datetime64("1855-01-01"),
                        np.datetime64("1858-01-01"))
        plt.savefig(self.root + "PostProcessing/Plots/tos_bsf_comp_new_fw.png",
                    dpi=600)

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
        print (sec)
        print(dskfj)

if __name__ == "__main__":
    #case = NEMO_case("EXP_mes_climatology_1850_1870_rnf_fix", "domain_cfg_mes.nc")
    #case.calc_barotropic_stream_function(1855, 1858)
    #case.calc_SPG_temperature_naarc(1850, 1854)
    #case = NEMO_case("EXP_mes_climatology_1850_1870_fw10_rnf_fix", "domain_cfg_mes.nc")
    #case.calc_barotropic_stream_function(1855, 1858)
    
    #case_dict = [{"case": "EXP_mes_LSM_new_radiation"}]
    #nemo_comp = NEMO_compare(case_dict)
    #nemo_comp.plot_denmark_strait()
    
    def plot_tos_bsf_compare():
        case_dict = [{"case": "EXP_mes_climatology_1850_1870_rnf_fix",
                      "dom_cfg":"domain_cfg_mes.nc",
                      "zcoord":"MES",
                      "label":"rnf_fw1",
                      "y0":1855,
                      "y1":1858},
                {"case": "EXP_mes_climatology_1850_1870_fw10_rnf_fix", 
                      "dom_cfg":"domain_cfg_mes.nc",
                      "zcoord":"MES",
                      "label":"rnf_fw10",
                      "y0":1855,
                      "y1":1858}]
        comp = NEMO_compare(case_dict)
        comp.plot_tos_bsf_timeseries()
    plot_tos_bsf_compare()
    #case.plot_tos_timeseries(y0, y1)
    #case.plot_bsf_timeseries(y0, y1)
    #for case_i in [case_dict[0]]:
    #    case = NEMO_case(**case_i)
    #    #case.calc_SPG_temperature_naarc(1950, 1952)
    
    
    #nemo_comp.get_nemo("1854/12/VERIFY_1m_18541201_18541230_grid_T.nc", "tos_con")
    #nemo_comp.plot_nemo()
    
