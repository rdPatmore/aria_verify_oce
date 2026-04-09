from plot_ensemble import glosat_ensemble_analysis
import glob
import xarray as xr
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

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

            baroV = depth_weight_vvel_series.sum(dim="depthv") 
            #baroV_mean = baroV.mean("time_centered")
            #baroV = baroV.expand_dims(year=[y])
            y_set.append(baroV)

        with ProgressBar():
            baroV = xr.concat(y_set, "time_centered").load()

        # get depth integrated velocities
        domcfg = xr.open_dataset(self.dom_path, chunks="auto").squeeze()
        for coord in ["x", "yy"]:
            if coord in domcfg.coords.keys():
                domcfg = domcfg.drop_vars(coord)
        e1v = domcfg.e1v
        e1v = e1v.assign_coords({"nav_lon":domcfg.glamu,
                                 "nav_lat":domcfg.gphiu})

        baroV = baroV.stack(a=["x","y"])
        e1v = e1v.stack(a=["x","y"])

        baroU, e1v = xr.align(baroV, e1v)

        e1v = e1v.unstack("a")
        baroV = baroV.unstack("a")

        transport = ( baroV * e1v ).sortby("x",
                                     ascending=False).cumsum(dim="x")/1e6

        BSF_masked = transport.sortby("x").where(domcfg.top_level == 1)

        BSF_masked = BSF_masked.assign_coords({"nav_lon":domcfg.nav_lon,
                                               "nav_lat":domcfg.nav_lat})

        BSF_masked.name = "BSF"
        BSF_masked.to_netcdf(self.save_path + 
                             f"BSF_{y0}_{y1}.nc")
        
    def get_barotropic_stream_function(self, y0, y1):
        """ access saved bsf """
        self.bsf = xr.open_dataarray(self.save_path + 
                             f"BSF_{y0}_{y1}.nc")
        
class NEMO_compare(object):
    """
    """

    def __init__(self, case_dict, y0, y1):
        self.nemo_path = "/gws/ssde/j25a/verify_oce/NEMO/Outputs/"
        self.mes_case = "EXP_mes_LSM_new_radiation/"
        self.zlevel_case = "EXP_zlevel_LSM_new_radiation/"
        self.y0 = y0
        self.y1 = y1

        self.cases = {}
        for i in range(len(case_dict)):
            self.cases = {f"case{i}": NEMO_case(case_dict[0]["case"])}
    
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

    def plot_bsf_timeseries(self):
        """
        compare barotropic streamfunction timeseries for multiple cases
        """

        # initialise figure
        fig, axs = plt.subplots(3, figsize=(5,12))

        # get streamfunctions
        print (self.cases)
        for i in range(len(self.cases)):
             self.cases[f"case{i}"].get_barotropic_stream_function(self.y0, self.y1)

        axs[0].pcolormesh(self.cases["case0"].bsf.isel(year=-1).T)
        plt.show()

case = NEMO_case("EXP_mes_LSM_new_radiation", "domain_cfg_mes.nc")
case.calc_barotropic_stream_function(1850, 1854)
case = NEMO_case("EXP_zlevel_LSM_new_radiation", "domain_cfg_zps.nc")
case.calc_barotropic_stream_function(1850, 1854)

#case_dict = [{"case": "EXP_mes_LSM_new_radiation"}]
#comp = NEMO_compare(case_dict, 1850, 1854)
#comp.plot_bsf_timeseries()


#nemo_comp = NEMO_compare()
#nemo_comp.get_nemo("1854/12/VERIFY_1m_18541201_18541230_grid_T.nc", "tos_con")
#nemo_comp.plot_nemo()

