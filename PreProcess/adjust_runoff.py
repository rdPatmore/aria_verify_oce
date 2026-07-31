import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

class forcing_perturbation(object):
    """
    workflow for adjusting forcing
    """

    def __init__(self):
        self.root = "/gws/ssde/j25a/verify_oce/NEMO/Preprocessing/"
        self.output = "/gws/ssde/j25a/verify_oce/NEMO/Outputs/"

    def get_greenland_mask(self, ds, t_val, f_val):
        """
        get greenland mask according to lat-lon limits
        """

        msk = xr.where((ds.nav_lon > -66) &
                       (ds.nav_lon < -13) &
                       (ds.nav_lat > 58.92) &
                       (ds.nav_lat < 84.56),
                       t_val,
                       f_val)

        return msk

    def plot_greenland_mask(self):
        """ plot mask of greenland """

        # greenland mask
        domcfg = xr.open_dataset(self.root + "DOM/NAARC/domain_cfg_mes.nc",
                                 chunks=-1)
        landmask = domcfg.top_level.squeeze()
        landmask_alt = xr.where(landmask, -1, 0)
      
        #grnlnd_mask = self.get_greenland_mask(domcfg, landmask_alt, landmask)

        p = plt.pcolor(grnlnd_mask)
        plt.colorbar(p)
        plt.show()
        plt.savefig(self.root + "Plots/rnf_msk.png")
        
    def factor_greenland_runoff(self, factor):
        """ multiply the magnitude of greenland runoff by a factor """

        # get runoff data
        path = self.root + "RNF/runoff_1m_nomask.nc"
        rnf = xr.open_dataset(path, chunks=-1)

        # get Greenland Mask
        domcfg = xr.open_dataset(self.root + "DOM/NAARC/domain_cfg_mes.nc",
                                 chunks=-1).squeeze(drop=True)
        grnlnd_mask = self.get_greenland_mask(domcfg, 1, 0)

        # multiply region by factor
        rnf["sorunoff"] = xr.where(grnlnd_mask,
                                   rnf.sorunoff * factor,
                                   rnf.sorunoff)

        # save
        fn_out = self.root + f"RNF/runoff_1m_nomask_grnlnd_f{str(factor)}.nc"
        rnf.to_netcdf(fn_out)

    def plot_runoff_magnitude(self):
        """ plot Greenland runoff climatology in Sv """

        # units for runoff kg/(m^2.s)
        # units for Sv 1e6 m^3/s

        # get runoff data
        path = self.root + "RNF/runoff_1m_nomask_grnlnd_f10.nc"
        rnf = xr.open_dataset(path, chunks=-1).sorunoff

        # get mask and domain_cfg
        domcfg = xr.open_dataset(self.root + "DOM/NAARC/domain_cfg_mes.nc",
                                 chunks=-1).squeeze(drop=True)
        grnlnd_mask = self.get_greenland_mask(domcfg, 1, 0)

        rho0=1026

        # mask greenland area 
        grnlnd_rnf = grnlnd_mask * rnf

        # convert units to Sv
        rnf_Sv = grnlnd_rnf * domcfg.e1t * domcfg.e2t / (rho0 * 1e6)

        # sum river points
        rnf_Sv_sum = rnf_Sv.sum(["x","y"])

        p = plt.plot(rnf_Sv_sum)
        plt.show()

    def plot_runoff_spatially(self):
        """ plot new and old runoff for greenland from model output"""

        year = 1858
        data_path = self.output + "Historical/"

        # get mask and domain_cfg
        domcfg = xr.open_dataset(self.root + "DOM/NAARC/domain_cfg_mes.nc",
                                 chunks=-1).squeeze()
        grnlnd_mask = self.get_greenland_mask(domcfg, 1, np.nan)

        # get old runoff
        case = "EXP_mes_climatology_1850_1870"
        fn = f"{data_path}/{case}/{year}/12/VERIFY_1m_{year}1201_{year}1230_grid_T.nc"
        old = xr.open_dataset(fn, chunks=-1).friver.squeeze()
        #grnlnd_mask = old.where(old>0)
        old = old.where(old>0)
        old_msk = self.get_greenland_mask(old, old, np.nan)
        old_msk = old_msk.dropna(dim="x", how="all").dropna(dim="y",
                                                            how="all")

        # get new runoff
        case = "EXP_mes_climatology_1850_1870_fw10"
        fn = f"{data_path}/{case}/{year}/12/VERIFY_1m_{year}1201_{year}1230_grid_T.nc"
        new = xr.open_dataset(fn, chunks=-1).friver.squeeze()
        print (new)
        print (grnlnd_mask)
        #new_msk = new * grnlnd_mask
        new = new.where(new>0)
        new_msk = self.get_greenland_mask(new, new, np.nan)
        new_msk = new_msk.dropna(dim="x", how="all").dropna(dim="y",
                                                            how="all")
        diff = old_msk - new_msk
        

        fig, axs = plt.subplots(1,3)

        p= axs[0].pcolor(old_msk)
        axs[1].pcolor(new_msk)
        axs[2].pcolor(diff)
        plt.colorbar(p)
        plt.show()

forper =  forcing_perturbation()
#forper.factor_greenland_runoff(10)
#forper.plot_greenland_mask()
#forper.plot_runoff_magnitude()
forper.plot_runoff_spatially()
