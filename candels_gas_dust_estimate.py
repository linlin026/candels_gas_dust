import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import convert as convert0
import size_v1.gas_dust_estimate as convert

import pdb


cat=pd.read_table('../Progenitors/allfield_cat2.csv', sep=',')
nn=len(cat)
z_best=cat['Z_BEST']
sf_flag=cat['sf_flag']
lmass=cat['M_MED']
mass=10**lmass
SMA = cat['sma_F160W_kpc']
lSMA=np.log10(cat['sma_F160W_kpc'])
lR90=np.log10(cat['sma_F160W_kpc']*2.6)
lssfr=cat['ssfr_uv_corr']
lsfr=cat['ssfr_uv_corr']+lmass
sfr=10**lsfr
dlSMA=cat['delta_lSMA']
dlssfr=cat['delta_ssfr']
Av=cat['med_av']
lAv=np.log10(Av)
b2a=cat['b2a_f160w']
ssfr_flag=cat['ssfr_flag']


lsigsfr = convert.ltot2sig(lsfr, lsma=lSMA)           # Msun/Kpc^2
lsigstar = convert.ltot2sig(lmass, lsma=lSMA)         # Msun/Kpc^2

#----

lsiggas_KS, lmgas_KS = convert.KS(lsigsfr, lSMA)      # Msun/kpc^2
lsiggas_eKS, lmgas_eKS = convert.eKS(lsigsfr, lsigstar-6, lSMA)
lsiggas_KSdyn_TF, lmgas_KSdyn_TF = convert.KSdyn_TF(lsigsfr, lmass, z_best, SMA)
lsiggas_KSdyn_GMs, lmgas_KSdyn_GMs = convert.KSdyn_GMs(lsigsfr, lmass, SMA)
lsiggas_KSdyn_GMtot, lmgas_KSdyn_GMtot = convert.KSdyn_GMtot(lsigsfr, lmass, SMA)
lsiggas_re = convert.newKS(lsigsfr, dlssfr) + 6      # Msun/kpc^2
lmgas_re = lsiggas_re + np.log10(np.pi) + 2*lSMA     # sigma * pi * R**2

lsigmol_T18, lmmol_T18 = convert.T18_curved(z_best=z_best,lmass=lmass,lssfr=lssfr,lSMA=lSMA)
lsigmol_mff, lmmol_mff = convert.F17_mff(lsigsfr, b2a, SMA)
lsigmol_mff_varyH, lmmol_mff_varyH = convert.F17_mff_varyH(lsigsfr, b2a, SMA)
lmmol_re = lmmol_T18 - np.log10(2)


#----

NHtot_KS = convert.lsiggas_to_NH(lsiggas_KS)
NHtot_eKS = convert.lsiggas_to_NH(lsiggas_eKS)
NHtot_KSdyn_TF = convert.lsiggas_to_NH(lsiggas_KSdyn_TF)
NHtot_KSdyn_GMs = convert.lsiggas_to_NH(lsiggas_KSdyn_GMs)
NHtot_KSdyn_GMtot = convert.lsiggas_to_NH(lsiggas_KSdyn_GMtot)
NHtot_re = convert.lsiggas_to_NH(lsiggas_re)

NHmol_T18 = convert.lsiggas_to_NH(lsigmol_T18)
NHmol_mff = convert.lsiggas_to_NH(lsigmol_mff)
NHmol_mff_varyH = convert.lsiggas_to_NH(lsigmol_mff_varyH)

#----

Avtot_KS = convert.NH_to_Av_MZ(NHtot_KS,lmass=lmass,z_best=z_best)
Avtot_eKS = convert.NH_to_Av_MZ(NHtot_eKS,lmass=lmass,z_best=z_best)
Avtot_KSdyn_TF = convert.NH_to_Av_MZ(NHtot_KSdyn_TF,lmass=lmass,z_best=z_best)
Avtot_KSdyn_GMs = convert.NH_to_Av_MZ(NHtot_KSdyn_GMs,lmass=lmass,z_best=z_best)
Avtot_KSdyn_GMtot = convert.NH_to_Av_MZ(NHtot_KSdyn_GMtot,lmass=lmass,z_best=z_best)
Avtot_re = convert.NH_to_Av_MZ(NHtot_re,lmass=lmass,z_best=z_best)

Avmol_T18 = convert.NH_to_Av_MZ(NHmol_T18,lmass=lmass,z_best=z_best)
Avmol_mff = convert.NH_to_Av_MZ(NHmol_mff,lmass=lmass,z_best=z_best)
Avmol_mff_varyH = convert.NH_to_Av_MZ(NHmol_mff_varyH,lmass=lmass,z_best=z_best)

#----

tautot_KS = convert.Av_to_tau(Avtot_KS)
tautot_eKS = convert.Av_to_tau(Avtot_eKS)
tautot_KSdyn_TF = convert.Av_to_tau(Avtot_KSdyn_TF)
tautot_KSdyn_GMs = convert.Av_to_tau(Avtot_KSdyn_GMs)
tautot_KSdyn_GMtot = convert.Av_to_tau(Avtot_KSdyn_GMtot)
tautot_re = convert.Av_to_tau(Avtot_re)

taumol_T18 = convert.Av_to_tau(Avmol_T18)
taumol_mff = convert.Av_to_tau(Avmol_mff)
taumol_mff_varyH = convert.Av_to_tau(Avmol_mff_varyH)

##########

df = pd.DataFrame({'lmstar': lmass, 'z_best': z_best, 'lsma': lSMA,
                   'lssfr': lssfr, 'dlsma': dlSMA, 'dlssfr': dlssfr,
                   'lAv': lAv, 'b2a': b2a, 'ssfr_flag': ssfr_flag,
                   'lsigsfr': lsigsfr, 'sigstar': lsigstar,
                   'lSIGgas_KS': lsiggas_KS,
                   'lSIGgas_eKS': lsiggas_eKS,
                   'lSIGgas_dyn_TF': lsiggas_KSdyn_TF,
                   'lSIGgas_dyn_GMs': lsiggas_KSdyn_GMs,
                   'lSIGgas_dyn_GMtot': lsiggas_KSdyn_GMtot,
                   'lSIGgas_re': lsiggas_re,
                   'lSIGmol_T18': lsigmol_T18,
                   'lSIGmol_mff': lsigmol_mff,
                   'lSIGmol_mff_varyH': lsigmol_mff_varyH,
                   'lmgas_KS': lmgas_KS,
                   'lmgas_eKS': lmgas_eKS,
                   'lmgas_dyn_TF': lmgas_KSdyn_TF,
                   'lmgas_dyn_GMs': lmgas_KSdyn_GMs,
                   'lmgas_dyn_GMtot': lmgas_KSdyn_GMtot,
                   'lmgas_re': lmgas_re,
                   'lmmol_T18': lmmol_T18,
                   'lmmol_mff': lmmol_mff,
                   'lmmol_mff_varyH': lmmol_mff_varyH,
                   'lmmol_re': lmmol_re,
                   'lAvtot_KS': np.log10(Avtot_KS),
                   'lAvtot_eKS': np.log10(Avtot_eKS),
                   'lAvtot_dyn_TF': np.log10(Avtot_KSdyn_TF),
                   'lAvtot_dyn_GMs': np.log10(Avtot_KSdyn_GMs),
                   'lAvtot_dyn_GMtot': np.log10(Avtot_KSdyn_GMtot),
                   'lAvtot_re': np.log10(Avtot_re),
                   'lAvmol_T18': np.log10(Avmol_T18),
                   'lAvmol_mff': np.log10(Avmol_mff),
                   'lAvmol_mff_varyH': np.log10(Avmol_mff_varyH),
                   'ltautot_KS': np.log10(tautot_KS),
                   'ltautot_eKS': np.log10(tautot_eKS),
                   'ltautot_dyn_TF': np.log10(tautot_KSdyn_TF),
                   'ltautot_dyn_GMs': np.log10(tautot_KSdyn_GMs),
                   'ltautot_dyn_GMtot': np.log10(tautot_KSdyn_GMtot),
                   'ltautot_re': np.log10(tautot_re),
                   'ltaumol_T18': np.log10(taumol_T18),
                   'ltaumol_mff': np.log10(taumol_mff),
                   'ltaumol_mff_varyH': np.log10(taumol_mff_varyH)})


df.to_csv('candels_gas_dust_estimate.csv', index=False, float_format='%.3f')


