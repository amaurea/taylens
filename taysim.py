#!/usr/bin/env python
import numpy as np, healpy, argparse, sys, warnings
from taylens import *
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
specs = readspec(args.specs)
Cl = np.zeros((4,specs.shape[1]))
Cl[[0,1,3],:] = specs[:3]
Cphi = specs[3]

# Scale them appropriately
l = np.arange(specs.shape[1])
Cl[:,1:] /= l[1:]*(l[1:]+1)/(2*np.pi)
Cphi[1:] /= l[1:]**4*2.726e6**2

if not args.pol: Cl = Cl[0]
synlmax = min(synlmax, l[-1])

# Coordinates of healpix pixel centers
prt("Computing ipos")
ipos = np.array(healpy.pix2ang(args.nside, np.arange(12*args.nside**2)))

for sim in range(myid, args.nsim, nproc):
	# Simulate a CMB and lensing field
	prt("Simulation %d of %d" % (sim+1,args.nsim))
	prt("Simulating cmb")
	cmb = np.array(healpy.synfast(Cl,   args.nside, new=True))
	if "u" in args.output:
		prt("Writing unlensed map")
		healpy.write_map("%s/ucmb%03d_%d.fits" % (args.odir, sim, args.order), cmb)
	if cmb.ndim == 1: cmb = np.reshape(cmb, [1,cmb.size])
	prt("Simulating lensing field")
	aphi = healpy.synalm(Cphi, lmax=synlmax, new=True)

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
