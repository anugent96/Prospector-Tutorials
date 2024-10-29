# import modules
import sys, os
import numpy as np
from sedpy.observate import load_filters
from prospect import prospect_args
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer
from astropy.io import fits
from sedpy import observate
from prospect.models.sedmodel import PolySpecModel, gauss
from scipy import optimize
from prospect.sources import FastStepBasis
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
from astropy.cosmology import WMAP9 as cosmo
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated
from scipy.stats import truncnorm
from pandas import read_csv
from prospect.utils.obsutils import fix_obs
import fsps

# define user-specific paths and filter names
#apps = os.getenv('APPS')

def load_spec_csv(filename):
    f = read_csv(filename, header=0, delimiter=',')
    f.columns = ['wavelength', 'flux', 'err']
    wave = np.array(f['wavelength'])
    flux = np.array(f['flux'])
    err = np.array(f['err'])
    return wave, flux, err



def build_obs(spec =True, spec_file='FRB190608_prospect.csv', maskspec=True,
              err_floor=0.05, **kwargs):
    
    """
    Load photometry and spectra
    """

    ### switch for data
    obs = {}

    # First, make a list of the filters you want and set them equal to "filternames"
    DECaL = ['decam_{}'.format(b) for b in ['g', 'r', 'z']]
    SDSS = ['sdss_{}0'.format(filt) for filt in ['u','i']]
    HST = ['wfc3_ir_f160w', 'wfc3_uvis_f300x']
    VISTA = ['VISTA_{}'.format(b) for b in ['J', 'Ks']]
    WISE = ['wise_w'+filt for filt in ['1','2','3']]

    filternames = DECaL + SDSS + HST + VISTA + WISE

    # Load in the filter transmission curves using SedPy's load_filter() function
    obs["filters"] = load_filters(filternames)

    # Find the wavelengths using .wave_effective from SedPy
    obs["phot_wave"] = [f.wave_effective for f in obs["filters"]]

    # Put in photometry magnitude points (**NOTE: NEEDS TO BE IN SAME ORDER AS FILTERS)
    M_AB = np.array([17.98, 17.41, 16.92, 19.06,  17.12, 16.67, 19.51, 16.76, 16.55, 17.26,17.47,16.78])
    M_AB_unc = np.array([0.001, 0.002, 0.001, 0.043,  0.008, 0.001, 0.014, 0.024, 0.036, 0.025,0.056,0.146])

    # Convert to flux in maggies
    mags = 10**(-0.4*M_AB)

    # Convert magnitude uncertainties to flux uncertainties
    mag_down = [x-y for (x,y) in zip(M_AB, M_AB_unc)]
    flux_down = [10**(-0.4*x) for x in mag_down]
    flux_uncertainty = [y-x for (x,y) in zip(mags, flux_down)]

    # Add in error floor
    flux_uncertainty = np.clip(flux_uncertainty, mags*err_floor, np.inf)

    obs['maggies'] = np.array(mags)
    obs['maggies_unc'] = np.array(flux_uncertainty)

    # Here we mask out any NaNs or infs
    obs['phot_mask'] = np.isfinite(np.squeeze(mags))

    # We will also put the redshift in (if its not known set it = None)
    obs['redshift'] = 0.11778 
    
    if spec:
        spec_wave, spec_fd, spec_fd_unc = load_spec_csv(spec_file)
    
        if maskspec:
            # create spectral mask
            # approximate cut-off for MILES library at 7500 A rest-frame, using SDSS redshift,
            # also mask Sodium D absorption
            wave_rest = spec_wave / (1+obs['redshift'])
            mask2 = (spec_fd_unc != 0) & \
                   (spec_fd != 0) & \
                   (wave_rest < 7500) & \
                   (spec_wave > 4050)   

            # apply standard mask
            spec_wave = spec_wave[mask2]
            spec_fd = spec_fd[mask2]
            spec_fd_unc = spec_fd_unc[mask2]

            # second mask if needed
            mask = []
            for i in range(len(spec_wave)):
                mask.append(True)
            mask = np.array(mask)

            bad_idx = np.where((spec_wave < 5582) & (spec_wave > 5572)) #2nd condition
            mask[bad_idx] = False

            bad_idx2 = np.where((spec_wave < 6310) & (spec_wave > 6295)) #2nd condition
            mask[bad_idx2] = False

            # apply second mask
            obs['wavelength'] = spec_wave[mask]
            obs['spectrum'] = spec_fd[mask] 
            obs['unc'] = spec_fd_unc[mask] 
            
        else:
            obs['wavelength'] = spec_wave
            obs['spectrum'] = spec_fd
            obs['unc'] = spec_fd_unc
    
    else:
        # No spectrum
        obs['wavelength'] = None
        obs['spectrum'] = None
        obs['unc'] = None
        
        
    return obs


# --------------
# Transformation Functions
# --------------
def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2

def dust2_to_dust1(dust2=None, **kwargs):
    return dust2

# add redshift scaling to agebins, such that
# t_max = t_univ
def zred_to_agebins(zred=None,agebins=None,**extras):
    tuniv = cosmo.age(zred).value[0]*1e9
    tbinmax = (tuniv*0.9)
    agelims = [0.0,7.4772] + np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)]
    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T

def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
    agebins = zred_to_agebins(zred=zred)
    logsfr_ratios = np.clip(logsfr_ratios,-10,10) # numerical issues...
    nbins = agebins.shape[0]
    sratios = 10**logsfr_ratios
    dt = (10**agebins[:,1]-10**agebins[:,0])
    coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (10**logmass) / coeffs.sum()
    return m1 * coeffs

def massmet_to_logmass(massmet=None,**extras):
    return massmet[0]

def massmet_to_logzsol(massmet=None,**extras):
    return massmet[1]

##### Mass-metallicity prior ######
class MassMet(priors.Prior):
    """A Gaussian prior designed to approximate the Gallazzi et al. 2005 
    stellar mass--stellar metallicity relationship.
    """

    prior_params = ['mass_mini', 'mass_maxi', 'z_mini', 'z_maxi']
    distribution = truncnorm
    massmet = np.loadtxt('gallazzi_05_massmet.txt')
    def __len__(self):
        """ Hack to work with Prospector 0.3
        """
        return 2

    def scale(self,mass):
        upper_84 = np.interp(mass, self.massmet[:,0], self.massmet[:,3]) 
        lower_16 = np.interp(mass, self.massmet[:,0], self.massmet[:,2])
        return (upper_84-lower_16)

    def loc(self,mass):
        return np.interp(mass, self.massmet[:,0], self.massmet[:,1])

    def get_args(self,mass):
        a = (self.params['z_mini'] - self.loc(mass)) / self.scale(mass)
        b = (self.params['z_maxi'] - self.loc(mass)) / self.scale(mass)
        return [a, b]

    @property
    def range(self):
        return ((self.params['mass_mini'], self.params['mass_maxi']),\
                (self.params['z_mini'], self.params['z_maxi']))

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        """Compute the value of the probability density function at x and
        return the ln of that.

        :params x:
            x[0] = mass, x[1] = metallicity. Used to calculate the prior

        :param kwargs: optional
            All extra keyword arguments are used to update the `prior_params`.

        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        p = np.atleast_2d(np.zeros_like(x))
        a, b = self.get_args(x[...,0])
        p[...,1] = self.distribution.pdf(x[...,1], a, b, loc=self.loc(x[...,0]), scale=self.scale(x[...,0]))
        with np.errstate(invalid='ignore'):
            p[...,1] = np.log(p[...,1])
        return p

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.

        :param nsample: (optional)
            Unused
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = np.random.uniform(low=self.params['mass_mini'],high=self.params['mass_maxi'],size=nsample)
        a, b = self.get_args(mass)
        met = self.distribution.rvs(a, b, loc=self.loc(mass), scale=self.scale(mass), size=nsample)

        return np.array([mass, met])

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = x[0]*(self.params['mass_maxi'] - self.params['mass_mini']) + self.params['mass_mini']
        a, b = self.get_args(mass)
        met = self.distribution.ppf(x[1], a, b, loc=self.loc(mass), scale=self.scale(mass))
        return np.array([mass,met])

# --------------
# Model Definition
# --------------
nbins_sfh = 8 # 7-8 bins good for phot-only, can go up to 10-12 for phot+spec
def build_model(opt_spec = True, add_duste=True, add_neb=True, add_agn=True, mixture_model=True,
                remove_spec_continuum=True,
                marginalize_neb=True, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.
    :param object_redshift:
        If given, given the model redshift to this value.
    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.
    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.
    """

    # input basic continuity SFH
    model_params = TemplateLibrary["continuity_sfh"]

    # fit for redshift
    # use catalog value as center of the prior
    obs = build_obs()
    
    if obs['redshift'] is not None:
        # We have a redshift
        zred = obs['redshift']
        model_params["zred"]['isfree'] = True
        model_params["zred"]["init"] =  zred
        model_params["zred"]["prior"] = priors.TopHat(mini=zred-0.01, maxi=zred+0.01)
    else:
        # We do not have a redshift - set a reasonable range
        model_params["zred"]['isfree'] = True
        model_params["zred"]["prior"] = priors.TopHat(mini=0.001, maxi=0.5)

    
    # mass-metallicity prior
    # If you have reason to believe your galaxy is a dwrf, can set mass_mini = 6
    model_params['massmet'] = {'name': 'massmet',
                               'N': 2,
                               'isfree': True,
                               "init": np.array([8.0,0.0]),
                               'prior': MassMet(z_mini=-2.00, z_maxi=0.19, mass_mini=8, mass_maxi=12)}
    model_params['logmass'] = {'N': 1,
                               'isfree': False,
                               'depends_on': massmet_to_logmass,
                               'init': 10.0,
                               'units': 'Msun',
                               'prior': None}
    model_params['logzsol']['depends_on'] = massmet_to_logzsol
    model_params['logzsol']['isfree'] = False
    

    # modify to increase nbins
    model_params['agebins']['N'] = nbins_sfh
    model_params['mass']['N'] = nbins_sfh
    model_params['logsfr_ratios']['N'] = nbins_sfh-1
    model_params['logsfr_ratios']['init'] = np.full(nbins_sfh-1,0.0) # constant SFH
    model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
                                                                      scale=np.full(nbins_sfh-1,0.3),
                                                                      df=np.full(nbins_sfh-1,2))

    model_params['agebins']['depends_on'] = zred_to_agebins
    model_params['mass']['depends_on'] = logmass_to_masses

    # complexify the dust
    model_params['dust_type']['init'] = 4
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)
    model_params["dust_index"] = {"N": 1, 
                                  "isfree": True,
                                  "init": 0.0, "units": "power-law multiplication of Calzetti",
                                  "prior": priors.TopHat(mini=-1.0, maxi=0.4)}

    model_params['dust1'] = {"N": 1, 
                             "isfree": False, 
                             'depends_on': to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars",
                             "prior": None}
    model_params['dust1_fraction'] = {'N': 1,
                                      'isfree': True,
                                      'init': 1.0,
                                      'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    # velocity dispersion
    print('ADDING velocity dispersion')
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=40.0, maxi=400.0)

    # Change the model parameter specifications based on some keyword arguments
    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        print('adding dust emission')
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_gamma']['init'] = 0.01
        model_params['duste_qpah']['prior'] = priors.TopHat(mini=0.5, maxi=7.0)
        model_params['duste_gamma']['isfree'] = False
        #model_params['duste_gamma']['prior'] = priors.LogUniform(mini=0.001, maxi=0.5) 
        model_params['duste_umin']['isfree'] = False
        #model_params['duste_umin']['prior'] = priors.TopHat(mini=0.1, maxi=25)

    if add_agn:
        # Allow for the presence of an AGN in the mid-infrared
        print('adding an AGN component')
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True
        # note that fagn > 2 is unphysical, but it can be set arbitrarily high
        model_params['fagn']['prior'] = priors.LogUniform(mini=1e-5, maxi=2.0)
        model_params['agn_tau']['isfree'] = True
        model_params['agn_tau']['prior'] = priors.LogUniform(mini=5.0, maxi=150.)

    if add_neb:
        # Add nebular emission
        if opt_spec:
            # You have a spectrum
            model_params.update(TemplateLibrary["nebular"])
            model_params['gas_logu']['isfree'] = True
            model_params['gas_logz']['isfree'] = True
            model_params['nebemlineinspec'] = {'N': 1,
                                           'isfree': False,
                                           'init': False}
            _ = model_params["gas_logz"].pop("depends_on")

            if marginalize_neb:
                #Fit and marginalize over observed emissionlines
                model_params.update(TemplateLibrary['nebular_marginalization'])
                
                # Sets prior on emission line width
                model_params['eline_prior_width']['init'] = 1.0
                model_params['use_eline_prior']['init'] = True

            else:
                model_params['nebemlineinspec']['init'] = True
        else:
            # Photometry only fit
            # Turn nebular emission on, do not fit lines, do not fit gas-phase metallicity params
            model_params['nebemlineinspec']['init'] = False
            model_params['gas_logu']['isfree'] = False
            model_params['gas_logz']['isfree'] = False
            
    if opt_spec:
        # Only do these routines if there is a spectrum!
        
        if remove_spec_continuum:
            # This removes the continuum from the spectroscopy. Highly recommend
            # using when modeling both photometry & spectroscopy
            print('REMOVING spec continuum')
            model_params.update(TemplateLibrary['optimize_speccal'])
            model_params['spec_norm']['isfree'] = False
            model_params['spec_norm']['prior'] = priors.Normal(mean=1.0, sigma=0.3)
        
        elif remove_spec_continuum == False:
            model_params.update(TemplateLibrary["optimize_speccal"])
            # fit for normalization of spectrum
            model_params['spec_norm'] = {'N': 1,'init': 1.0,'isfree': True,'prior': 
                                     priors.Normal(sigma=0.2, mean=1.0), 'units': 'f_true/f_obs'}
            # Increase the polynomial size to 12
            model_params['polyorder'] = {'N': 1, 'init': 12,'isfree': False}
            run_params["opt_spec"] = True

        # This is a pixel outlier model. It helps to marginalize over
        # poorly modeled noise, such as residual sky lines or
        # even missing absorption lines
        if mixture_model:
            print('ADDING mixture model')
            model_params['f_outlier_spec'] = {"N": 1, 
                                              "isfree": True, 
                                              "init": 0.01,
                                              "prior": priors.TopHat(mini=1e-5, maxi=0.5)}
            model_params['nsigma_outlier_spec'] = {"N": 1, 
                                                  "isfree": False, 
                                                  "init": 50.0}
            model_params['f_outlier_phot'] = {"N": 1, 
                                              "isfree": False, 
                                              "init": 0.00,
                                              "prior": priors.TopHat(mini=0, maxi=0.5)}
            model_params['nsigma_outlier_phot'] = {"N": 1, 
                                                  "isfree": False, 
                                                  "init": 50.0}


            # This is a multiplicative noise inflation term. It inflates the noise in
            # all spectroscopic pixels as necessary to get a good fit.
            model_params['spec_jitter'] = {"N": 1, 
                                           "isfree": True, 
                                           "init": 1.0,
                                           "prior": priors.TopHat(mini=1.0, maxi=3.0)}


    # Now instantiate the model using this new dictionary of parameter specifications
    model = PolySpecModel(model_params)

    return model


# --------------
# SPS Object
# --------------
def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous,
                        compute_vega_mags=compute_vega_mags)  # special to remove redshifting issue
    return sps

# -------------------------------------------------
# Noise Model - If you are using the "jitter" model
# --------------------------------------------------
def build_noise(**extras):
    jitter = Uncorrelated(parnames = ['spec_jitter'])
    spec_noise = NoiseModel(kernels=[jitter],metric_name='unc',weight_by=['unc'])
    return spec_noise, None

# -----------
# Everything
# ------------
def build_all(**kwargs):
    return (build_obs(**kwargs), build_model(**kwargs), build_sps(**kwargs), build_noise(**kwargs))


if __name__=='__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # - Add custom arguments -
    parser.add_argument('--add_neb', default=True,type=str2bool,
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--remove_spec_continuum', default=True,type=str2bool,
                        help="If set, fit continuum.")
    parser.add_argument('--add_duste',default=True,type=str2bool,
                        help="If set, add dust emission to the model.")
    parser.add_argument('--add_agn',default=True,type=str2bool,
                        help="If set, add agn emission to the model.")
    parser.add_argument('--objname', default='92942',
                        help="Name of the object to fit.")
    parser.add_argument('--mixture_model', default=True,type=str2bool,
                        help="If set, remove spectrum from obs.")
    parser.add_argument('--marginalize_neb', default=True,type=str2bool,
                        help="If set, remove spectrum from obs.")

    args = parser.parse_args()
    #run_params = vars(args)
    run_params = {}

    # add in dynesty settings
    run_params['dynesty'] = True
    run_params['nested_weight_kwargs'] = {'pfrac': 1.0}
    run_params['nested_nlive_batch'] = 500
    run_params['nested_walks'] = 48  # sampling gets very inefficient w/ high S/N spectra
    run_params['nested_nlive_init'] = 500 
    run_params['nested_dlogz_init'] = 0.01
    run_params['nested_maxcall'] = 7500000
    run_params['nested_maxcall_init'] = 7500000
    run_params['nested_sample'] = 'rwalk'
    run_params['nested_maxbatch'] = None
    run_params['nested_posterior_thresh'] = 0.03
    run_params['nested_first_update'] = {'min_ncall': 20000, 'min_eff': 7.5}
    #run_params['objname'] = str(run_params['objname'])

    obs, model, sps, noise = build_all(**run_params)
    print(model)
    run_params["param_file"] = __file__

    # Set up MPI. Note that only model evaluation is parallelizable in dynesty,
    # and many operations (e.g. new point proposal) are still done in serial.
    # This means that single-core fits will always be more efficient for large
    # samples. having a large ratio of (live points / processors) helps efficiency
    # Scaling is: S = K ln(1 + M/K), where M = number of processes and K = number of live points
    # Run as: mpirun -np <number of processors> python demo_mpi_params.py
    try:
        import mpi4py
        from mpi4py import MPI
        from schwimmbad import MPIPool

        mpi4py.rc.threads = False
        mpi4py.rc.recv_mprobe = False

        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        withmpi = comm.Get_size() > 1
    except ImportError:
        print('Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.')
        withmpi = False

    # Evaluate SPS over logzsol grid in order to get necessary data in cache/memory
    # for each MPI process. Otherwise, you risk creating a lag between the MPI tasks
    # caching data depending which can slow down the parallelization
    if (withmpi) & ('logzsol' in model.free_params):
        dummy_obs = dict(filters=None, wavelength=None)

        logzsol_prior = model.config_dict["logzsol"]['prior']
        lo, hi = logzsol_prior.range
        logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)

        sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
        for logzsol in logzsol_grid:
            model.params["logzsol"] = np.array([logzsol])
            _ = model.predict(model.theta, obs=dummy_obs, sps=sps)

    # ensure that each processor runs its own version of FSPS
    # this ensures no cross-over memory usage
    from prospect.fitting import lnprobfn
    from functools import partial
    lnprobfn_fixed = partial(lnprobfn, sps=sps)

    if withmpi:
        with MPIPool() as pool:
            # The subprocesses will run up to this point in the code
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            nprocs = pool.size
            output = fit_model(obs, model, sps, noise, pool=pool, queue_size=nprocs, lnprobfn=lnprobfn_fixed, **run_params)
    else:
        print(model)
        output = fit_model(obs, model, sps, noise, lnprobfn=lnprobfn_fixed, **run_params)

    hfile = 'FRB190608_nonparam_mcmc.h5'

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass
