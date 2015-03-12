Taylens is a simple python implementation of nearest-neighbor Taylor
interpolation lensing. It is provided in the taylens module, which
defines the core functionality, and a driver script "taysim.py",
which provides a convenient command-line interface.

Taylens depends on reasonably recent versions of python2, scipy and
healpy, as well as optionally depending on mpi4py for MPI support.
It has been tested and found to work with

 python 2.7.5 - numpy 1.7.1 - healpy 1.6.1
 python 2.7.2 - numpy 1.7.1 - healpy 1.6.1
 python 2.7.3 - numpy ?.?.? - healpy 1.6.2

Here are some simple usage examples:

  1. python taysim.py ps.txt out

     Generates a random temperature-only realization of the unlensed CMB
     and lensing field at nside 512 from the power spectrum ps.txt, which must be
     in the same format as CAMB's unlensed+lensing spectrum, i.e. [l,tt,ee,bb,te,dd,dt,de].
     The CMB is then lensed with the lensing field at interpolation
     order 3, and the power spectrum of the result is written to
     out/spec000_3.txt. The output spectrum has the format [tt].

  2. python taysim.py -n 10 -g -O 2 -nside 2048 -p -v ps.txt out

     Generates 10 simulations, each at nside 2048 with interpolation
     order 2, with full polarization (-p), while printing verbose
     status information (-v). The -g switch turns on proper parallel
     transport support, which is not really necessary. The outputs
     are written to out/spec000_2.txt .. out/spec009_2.txt, and have the format
     [tt,ee,bb,te,eb,tb].

  3. python taysim.py -o uls ps.txt out

     As example 1, but outputs the unlensed CMB simulation
     as out/ucmb000_3.fits (u), the lensed CMB as out/lcmb000_3.fits (l),
     and the lensed spectrum as out/spec000_3.txt (s).
     (Other possible outputs are the lensing potential (p) and its
     gradient (g).)

  4. mpirun -npernode 1 python taysim.py -n 256 -p -nside 1024 -v

     Generates 256 polarized realizations using MPI, with
     1 task per node (each task uses Healpy's OpenMP parallelization,
     so the whole node will be used), while being verbose.

  5. OMP_NUM_THREADS=1 mpirun python taysim.py -n 256 -p

     As #4, but at nside 512, and running one task per core,
     with no OpenMP.
