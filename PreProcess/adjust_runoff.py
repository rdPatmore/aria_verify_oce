import xarray as xr
import matplotlib.pyplot as plt

class forcing_perturbation(object):
    """
    workflow for adjusting forcing
    """

    def __init__(self):
        self.root = "/gws/ssde/j25a/verify_oce/NEMO/Preprocessing/"

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
        domcfg = xr.open_dataset(self.root + "DOM/NAARC/domain_cfg_mes.nc")
        landmask = domcfg.top_level.squeeze()
        landmask_alt = xr.where(landmask, -1, 0)

      
        grnlnd_mask = self.get_greenland_mask(domcfg, landmask_alt, landmask)

        p = plt.pcolor(grnlnd_mask)
        plt.colorbar(p)
        plt.savefig(self.root + "Plots/rnf_msk.png")
        
        
    def factor_greenland_runoff(self, factor):
        """ multiply the magnitude of greenland runoff by a factor """

        # get runoff data
        path = self.root + "RNF/runoff_1m_nomask.nc"
        rnf = xr.open_dataset(path)

        # multiply region by factor

        rnf["sorunoff"] = xr.where(grnlnd_mask,
                                   rnf.sorunoff * factor,
                                   rnf.sorunoff)
        print (rnf)

        # save
        #fn_out = self.root + f"RNF/runoff_1m_nomask_grnlnd_f{str(factor)}.nc"
        #rnf.to_netcdf(fn_out)

forper =  forcing_perturbation()
#forper.factor_greenland_runoff(2)
forper.plot_greenland_mask()
