#!/usr/bin/env python
import numpy as np, healpy, argparse, sys, warnings
from taylens import *

# This generates correlated T,E,B and Phi maps
def simulate_tebp_correlated(cl_tebp_arr,nside,lmax) :
	alms=healpy.synalm(cl_tebp_arr,lmax=lmax,new=True)
	aphi=alms[-1]
	acmb=alms[0:-1]
#Set to zero above map resolution to avoid aliasing
	beam_cut=np.ones(3*nside)
	for ac in acmb :
		healpy.almxfl(ac,beam_cut,inplace=True)
	cmb=np.array(healpy.alm2map(acmb,nside,pol=True))

	return cmb,aphi

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="Simulate CMB and lensing potentials maps, use the latter to lens the former, and output the power spectrum of the resulting lensed map.")
parser.add_argument("specs", help="Input power spectrum file in CAMB format.")
parser.add_argument("odir",  help="Directory for simulated output spectra. Must exist.")
parser.add_argument("-o", "--output",  action="store", default="s", help="String describing what to output. A set of one or more of the characters 'u' (unlensed cmb), 'p' (lensing potential), 'g' (lensing gradient), 'l' (lensed cmb), 's' (lensed cmb spectrum).")
parser.add_argument(      "--nside",   type=int, default=512, help="Simulation HEALPix Nside.")
parser.add_argument("-n", "--nsim",    type=int, default=1,   help="Number of simulations.")
parser.add_argument("-O", "--order",   type=int, default=3,   help="Taylor expansion order, typically in the range 2-4.")
parser.add_argument("-s", "--seed",    type=int, help="Random number generator seed. If left out, a random seed is used. If specified, the MPI task id is added to the seed before being used.")
parser.add_argument("-g", "--geodesic",action="store_true",   help="Whether to apply tiny corrections for geodesic parallel transport.")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-p", "--pol",     action="store_true",   help="If specified, simulations will be full TEB-simulations rather than just T.")
parser.add_argument("--synlmax",       type=int, help="The maximum multipole to use when generating the lensing field.")
parser.add_argument("-l", "--lmax",    type=int, help="The maximum multipole to use in the Taylor expansion.")
args = parser.parse_args()

prt = args.verbose and tprint or silent
prt("Start")

try:
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	myid, nproc = comm.Get_rank(), comm.Get_size()
except ImportError:
	myid, nproc = 0, 1

synlmax = args.synlmax
if synlmax is None: synlmax = 8*args.nside

if args.seed is not None:
	np.random.seed(args.seed+myid)

# Read the input spectra (in CAMB format)
data=np.transpose(np.loadtxt(args.specs))
lmax_cl=len(data[0])+1
l=np.arange(int(lmax_cl+1))
synlmax = min(synlmax, l[-1])
if not args.pol :
	cl_tebp_arr=np.zeros([3,lmax_cl+1])
	cl_tebp_arr[0,2:]=2*np.pi*data[1]/(l[2:]*(l[2:]+1))      #TT
	cl_tebp_arr[1,2:]=2*np.pi*data[5]/(l[2:]*(l[2:]+1))**2   #PP
	cl_tebp_arr[2,2:]=2*np.pi*data[6]/(l[2:]*(l[2:]+1))**1.5 #TP
else :
	cl_tebp_arr=np.zeros([10,lmax_cl+1])
	cl_tebp_arr[0,2:]=2*np.pi*data[1]/(l[2:]*(l[2:]+1))      #TT
	cl_tebp_arr[1,2:]=2*np.pi*data[2]/(l[2:]*(l[2:]+1))      #EE
	cl_tebp_arr[2,2:]=2*np.pi*data[3]/(l[2:]*(l[2:]+1))      #BB
	cl_tebp_arr[3,2:]=2*np.pi*data[5]/(l[2:]*(l[2:]+1))**2   #PP
	cl_tebp_arr[4,2:]=2*np.pi*data[4]/(l[2:]*(l[2:]+1))      #TE
	cl_tebp_arr[5,:] =np.zeros(lmax_cl+1)                    #EB
	cl_tebp_arr[6,:] =np.zeros(lmax_cl+1)                    #BP
	cl_tebp_arr[7,:] =np.zeros(lmax_cl+1)                    #TB
	cl_tebp_arr[8,2:]=2*np.pi*data[7]/(l[2:]*(l[2:]+1))**1.5 #EP
	cl_tebp_arr[9,2:]=2*np.pi*data[6]/(l[2:]*(l[2:]+1))**1.5 #TP

# Coordinates of healpix pixel centers
prt("Computing ipos")
ipos = np.array(healpy.pix2ang(args.nside, np.arange(12*args.nside**2)))

for sim in range(myid, args.nsim, nproc):
	# Simulate a CMB and lensing field
	prt("Simulation %d of %d" % (sim+1,args.nsim))
	prt("Simulating cmb and lensing potential jointly")
	cmb, aphi = simulate_tebp_correlated(cl_tebp_arr,args.nside,synlmax)

	if "u" in args.output:
		prt("Writing unlensed map")
		healpy.write_map("%s/ucmb%03d_%d.fits" % (args.odir, sim, args.order),
				 cmb)
	if cmb.ndim == 1: cmb = np.reshape(cmb, [1,cmb.size])
		
	# Compute the offset positions
	prt("Computing lensing gradient")
	phi, phi_dtheta, phi_dphi = healpy.alm2map_der1(aphi, args.nside, lmax=synlmax)
	if "p" in args.output:
		prt("Writing lensing field")
		healpy.write_map("%s/phi%03d_%d.fits" % (args.odir, sim, args.order), phi)
	if "g" in args.output:
		prt("Writing gradient")
		healpy.write_map("%s/grad%03d_%d.fits" % (args.odir, sim, args.order), [phi_dtheta, phi_dphi])
	del aphi
	prt("Computing lensed positions")
	opos, rot = offset_pos(ipos, phi_dtheta, phi_dphi, pol=args.pol, geodesic=args.geodesic)
	del phi, phi_dtheta, phi_dphi

	prt("Starting taylor expansion")
	# Interpolate maps one at a time
	maps  = []
	for comp in cmb:
		for m in taylor_interpol_iter(comp, opos, args.order, verbose=args.verbose, lmax=args.lmax):
			pass
		maps.append(m)
	del opos, cmb
	prt("Rotating")
	rm = apply_rotation(maps, rot)
	if "l" in args.output:
		prt("Writing lensed map")
		healpy.write_map("%s/lcmb%03d_%d.fits" % (args.odir, sim, args.order), rm)
	if "s" in args.output:
		prt("Computing power spectrum")
		alm  = healpy.map2alm(rm, use_weights=True, iter=1)
		spec = np.array(healpy.alm2cl(alm))
		if spec.ndim == 1: spec = np.reshape(spec, [1,spec.size])
		n = spec.shape[1]
		l = np.arange(spec.shape[1])
		spec[:,1:] *= l[1:n]*(l[1:n]+1)/(2*np.pi)
		prt("Writing spectrum")
		np.savetxt("%s/spec%03d_%d.txt" % (args.odir, sim, args.order), spec.T, fmt=" %15.7e")
