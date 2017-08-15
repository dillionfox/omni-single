from __future__ import division
import numpy as np
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import time
import MDAnalysis

"""
ELECTROSTATIC MAPPING:
is a procedure described by R. Remsing and J. Weeks (2014) that computes the long-ranged
electrostatic potential (LREP) at discrete points at an interface. The potential calculation
must be computed at each point on the interface and considers every water molecule in the
simulation, i.e. there is no long range cutoff. The value of the potential can then be 
interpretted as the collective polarization of water at that point in space.

This code is broken up into roughly two parts. First, you have to determine an interface
between your protein/membrane/whatever and the water. There's a paper by Chandler and Willard
that describes what they call Instantaneous Interfaces (II) that satisfies this requirement.
The first part of the code determines the interfaces for every frame (or just one frame), and
then calculates the LREP at each point on the II.

OUTPUT: 
- .pdb file for each frame containing instantaneous interface points. The beta value
for each point represents the strength of the long range electrostatic potential. Right now
the units are messed up (off by a factor of either 10 or 1/10). 
- There doesn't seem to be a point in returning all of this data back to omnicalc,
so right now I'm just having it return 'y' so omnicalc knows it can skip this
if the calculation has already been performed successfully.

NOTES:
Remsing and Weeks recommend averaging the electrostatic potential over many frames. This is
one of the options. It is highly recommended that the user chooses the grid spacing to be
0.1, but for very large systems (such as lipid bilayers) it is sometimes acceptable to use
0.2. Since there are 3 dimensions, increasing the grid spacing by a factor of 2 makes the 
code 8x faster.

TESTING:
I've tested this on a few systems and found that it does not scale well. The largest system
I was patient enough to test on contained 110,000 atoms and was made up of lipids, a small 
protein, and water. The instantaneous interface was determined in a matter of a few minutes
for each frame, but the electrostatic mapping procedure took about 3.5 hours. Since the code is
parallelized, it actually spat out 8 electrostatic maps every 3.5 hours, so on average it took about
half an hour per frame. 

"""

#--- WRITE OUTPUT
def write_pdb(coor,beta,fr):
	"""
	This function writes the coordinates of the Willard-Chandler instantaneous interface as a pdb,
	and populates the Beta field with the long-range electrostatic potential
	"""
	global name_modifier
	global verbose

	if verbose >= 1:
		print 'writing pdb...'
	outfile = open(str(work.postdir)+str(sn)+"_emap_"+str(fr)+str(name_modifier)+".pdb","w")
	count_zeros = 0
	for i in range(len(coor)):
		if (coor[i][0]!=0 and coor[i][1]!=0 and coor[i][2]!=0):
			t1 = "ATOM"					# ATOM
			t2 = 1						# INDEX
			t3 = "C"					# ATOM NAME
			t4 = ""						# ALTERNATE LOCATION INDICATOR
			t5 = "AAA"					# RESIDUE NAME
			t6 = "X"					# CHAIN
			t7 = 0						# RESIDUE NUMBER
			t8 = ""						# INSERTION CODE
			t9 = float(coor[i][0])				# X
			t10 = float(coor[i][1])				# Y
			t11 = float(coor[i][2])				# Z
			t12 = 0.0					# OCCUPANCY
			t13 = beta[i]					# TEMPERATURE FACTOR
			t14 = ""					# ELEMENT SYMBOL
			t15 = ""					# CHARGE ON ATOM
			outfile.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15))
	outfile.close()
	return 0

#--- EXTRACT COORDINATES
def extract_traj_info(PSF,DCD,selection_key):
	"""
	This function uses MDAnalysis to extract coordinates from the trajectory
	"""
	global verbose

	if verbose >= 1:
        	print 'loading coordinates...'
        # load some variables into global namespace
        global n_heavy_atoms
        global pbc
	global box_shift

        uni = MDAnalysis.Universe(PSF,DCD)
        nframes = len(uni.trajectory)					# number of frames
	box_shift = np.zeros((nframes,3))

        protein = uni.select_atoms(selection_key)			# identify atoms to build interface around
        heavy_atoms = protein.select_atoms('not name H*')		# Only need to consider heavy atoms
        protein_indices = heavy_atoms.indices 
        n_heavy_atoms = len(heavy_atoms.atoms)                          # number of heavy protein atoms
        positions = np.zeros((nframes,n_heavy_atoms,3))

        water = uni.select_atoms("resname TIP3")			# identify atoms to build interface around
        water_indices = water.indices 
        n_water = len(water.atoms)					# number of heavy protein atoms
        water_pos = np.zeros((nframes,n_water,3))

        for fr in range(nframes):                                       # save coordinates for each frame
                uni.trajectory[fr]
        	pbc = uni.dimensions[0:3]				# retrieve periodic bounds
        	sel = uni.select_atoms('all')
		box_shift[fr] = -sel.atoms.center_of_mass()+pbc/2.0	# first center at origin, then put vertx of quadrant 7 at origin
        	sel.atoms.translate(box_shift[fr])			# center selection

        	protein = uni.select_atoms(selection_key)		# identify atoms to build interface around
        	heavy_atoms = protein.select_atoms('not name H*')	# Only need to consider heavy atoms
		positions[fr] = heavy_atoms.positions/scale

        	water = uni.select_atoms("resname TIP3")		# identify atoms to build interface around
		water_pos[fr] = water.positions/scale

        pbc = uni.dimensions[0:3]/scale					# retrieve periodic bounds
        return [nframes,positions,water_pos]

#--- FUNCTIONS FOR COMPUTING RHO
def erf(x):
	"""
	This is a straight-forward implementation of a Numerical Recipe code for computing the error function
	"""

	sign = 1 if x >= 0 else -1
	x = abs(x)
	a1 =  0.254829592
	a2 = -0.284496736
	a3 =  1.421413741
	a4 = -1.453152027
	a5 =  1.061405429
	p  =  0.3275911
	t = 1.0/(1.0 + p*x)
	y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
	return sign*y # erf(-x) = -erf(x)

def phi(x, sig, cutoff):
	"""
	Equation 2 from Chandler paper
	"""

	phic = np.exp(-cutoff*cutoff/(2.0*sig*sig))
	C = 1.0 / ( (2*np.pi)**(0.5) * sig * erf(cutoff / (2.0**(0.5) * sig)) - 2.0*cutoff*phic )
	if np.abs(x) <= cutoff:
		phix = C * ( np.exp(-x*x/(2.0*sig*sig)) - phic )
	else: 
		phix = 0.0
	return phix

def gaussian_convolution(voxel_i,N,ngrid,pos,grid_spacing,dl,phi_bar):
	"""
	Convolve the density field with Gaussians
	"""

	nx = voxel_i-N			# N = int(cutoff/dL)-xg-1, xg: (0,2*int(cutoff/dL))
	if nx<0: nx+=ngrid		# wrap around periodic bounds
	elif nx>=ngrid: nx-=ngrid
	rx = np.abs(N*grid_spacing+(pos-voxel_i*grid_spacing))
	nrx = int(rx/dl)
	phix = phi_bar[nrx] + (phi_bar[nrx+1] - phi_bar[nrx]) * (rx - nrx*dl) / dl
	return phix,nx

def compute_coarse_grain_density(pos):
	"""
	This function takes the positions of the protein/lipids/whatever and computes the coarse grain
	density field
	"""
	global verbose

	if verbose >= 1:
		print '---> computing coarse grain density...'
	# load into global namespace
	global n_grid_pts
	global grid_spacing

	nconf = 1							# number of conformations to consider
	rho_pro = 50.0							# bulk density of protein

	cutoff = 0.7 							# cutoff for Gaussian (for convolution)
	dl = 0.01							# grid spacing for coarse grain density
	npoints = int(cutoff/dl)+1					# number of density points
	sigp = 0.24 							# width of Gaussian smoothing: protein
	phi_bar_p = [phi(i*dl, sigp, cutoff) for i in range(npoints*2)]	# coarse grain density

	Ninc = int(cutoff/dL)						# ~!~~~still not sure what this is~~~!~

	# define voxels
	n_grid_pts = [int(p/dL) for p in pbc]				# number of bins that naturally fit along each axis
	grid_spacing =  [pbc[i]/n_grid_pts[i] for i in range(len(pbc))]	# distance between grid points for each direction
	rho = np.zeros((n_grid_pts[0],n_grid_pts[1],n_grid_pts[2]))	# dummy arrays store coarse grain density

	for i in range(n_heavy_atoms):
		pos_i = pos[i]
		voxel_i = [int(pos_i[dim]/grid_spacing[dim]) for dim in range(3)]	# convert xyz to voxel
		for xg in range(2*Ninc): # 
			phix,nx = gaussian_convolution(voxel_i[0],Ninc-xg-1,n_grid_pts[0],pos_i[0],grid_spacing[0],dl,phi_bar_p)	
			for yg in range(2*Ninc):
				phiy,ny = gaussian_convolution(voxel_i[1],Ninc-yg-1,n_grid_pts[1],pos_i[1],grid_spacing[1],dl,phi_bar_p)	
				for zg in range(2*Ninc):
					phiz,nz = gaussian_convolution(voxel_i[2],Ninc-zg-1,n_grid_pts[2],pos_i[2],grid_spacing[2],dl,phi_bar_p)	
					rho[int(nx)][int(ny)][int(nz)] += phix*phiy*phiz/(rho_pro*nconf)	# Equation 3

	return rho

#--- FUNCTIONS FOR COMPUTING MARCHING CUBES
def GridInterp(grid1, grid2, value1, value2, rhoc):
	"""
	Part of the Marching Cubes algorithm. Note: This was adapted from a common C code that's floating around...
	"""

	gridc = np.zeros(3)
	epsilon = 0.000001 

	if  abs(rhoc - value1) < epsilon: 
		return grid1
	if  abs(rhoc - value2) < epsilon:
		return grid2
	if  abs(value1 - value2) < epsilon: 
		return grid1
	
	mu = (rhoc - value1) / (value2 - value1)
	gridc[0] =  grid1[0] + mu * (grid2[0] - grid1[0])
	gridc[1] =  grid1[1] + mu * (grid2[1] - grid1[1])
	gridc[2] =  grid1[2] + mu * (grid2[2] - grid1[2])
	
	return gridc

def MC_table(gridv, gridp, rhoc, trip):
	"""
	Part of the Marching Cubes algorithm. Note: This was adapted from a common C code that's floating around...
	There are 2^8 = 256 different ways to make a polygon out of vertices of a cube. This table is used to
	figure out how your surface is slicing through the voxel (3D pixel)

	"""

	edgeTable = [0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,\
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,\
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,\
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,\
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,\
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,\
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,\
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,\
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,\
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,\
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,\
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,\
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,\
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,\
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,\
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,\
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,\
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,\
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,\
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,\
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,\
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,\
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,\
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,\
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,\
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,\
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,\
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,\
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,\
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,\
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,\
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0]
	
	triTable = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1], \
	[3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1], \
	[3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1], \
	[3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1], \
	[9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1], \
	[9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1], \
	[2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1], \
	[8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1], \
	[9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1], \
	[4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1], \
	[3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1], \
	[1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1], \
	[4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1], \
	[4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1], \
	[9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1], \
	[1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1], \
	[5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1], \
	[2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1], \
	[9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1], \
	[0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1], \
	[2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1], \
	[10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1], \
	[4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1], \
	[5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1], \
	[5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1], \
	[9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1], \
	[0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1], \
	[1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1], \
	[10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1], \
	[8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1], \
	[2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1], \
	[7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1], \
	[9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1], \
	[2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1], \
	[11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1], \
	[9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1], \
	[5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1], \
	[11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1], \
	[11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1], \
	[1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1], \
	[9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1], \
	[5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1], \
	[2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1], \
	[0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1], \
	[5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1], \
	[6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1], \
	[0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1], \
	[3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1], \
	[6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1], \
	[5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1], \
	[1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1], \
	[10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1], \
	[6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1], \
	[1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1], \
	[8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1], \
	[7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1], \
	[3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1], \
	[5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1], \
	[0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1], \
	[9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1], \
	[8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1], \
	[5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1], \
	[0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1], \
	[6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1], \
	[10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1], \
	[10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1], \
	[8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1], \
	[1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1], \
	[3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1], \
	[0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1], \
	[10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1], \
	[0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1], \
	[3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1], \
	[6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1], \
	[9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1], \
	[8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1], \
	[3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1], \
	[6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1], \
	[0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1], \
	[10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1], \
	[10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1], \
	[1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1], \
	[2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1], \
	[7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1], \
	[7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1], \
	[2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1], \
	[1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1], \
	[11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1], \
	[8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1], \
	[0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1], \
	[7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1], \
	[10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1], \
	[2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1], \
	[6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1], \
	[7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1], \
	[2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1], \
	[1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1], \
	[10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1], \
	[10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1], \
	[0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1], \
	[7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1], \
	[6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1], \
	[8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1], \
	[9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1], \
	[6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1], \
	[4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1], \
	[10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1], \
	[8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1], \
	[0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1], \
	[1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1], \
	[8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1], \
	[10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1], \
	[4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1], \
	[10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1], \
	[5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1], \
	[11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1], \
	[9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1], \
	[6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1], \
	[7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1], \
	[3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1], \
	[7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1], \
	[9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1], \
	[3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1], \
	[6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1], \
	[9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1], \
	[1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],  \
	[4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],\
	[7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1], \
	[6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1], \
	[3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1], \
	[0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1], \
	[6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1], \
	[0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1], \
	[11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1], \
	[6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1], \
	[5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1], \
	[9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1], \
	[1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1], \
	[1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1], \
	[10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1], \
	[0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1], \
	[5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1], \
	[10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1], \
	[11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1], \
	[9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1], \
	[7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1], \
	[2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1], \
	[8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1], \
	[9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1], \
	[9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1], \
	[1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1], \
	[9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1], \
	[9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1], \
	[5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1], \
	[0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1], \
	[10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1], \
	[2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1], \
	[0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1], \
	[0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1], \
	[9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],\
	[2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1], \
	[5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1], \
	[3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1], \
	[5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1], \
	[8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1], \
	[9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1], \
	[0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1], \
	[1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1], \
	[3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1], \
	[4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1], \
	[9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1], \
	[11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1], \
	[11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1], \
	[2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1], \
	[9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1], \
	[3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1], \
	[1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1], \
	[4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1], \
	[4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1], \
	[0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1], \
	[3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1], \
	[3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1], \
	[0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  \
	[3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1], \
	[9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1], \
	[1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], \
	[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
	
	cubeindex = 0

	if gridv[0] < rhoc: cubeindex |= 1
	if gridv[1] < rhoc: cubeindex |= 2
	if gridv[2] < rhoc: cubeindex |= 4
	if gridv[3] < rhoc: cubeindex |= 8
	if gridv[4] < rhoc: cubeindex |= 16
	if gridv[5] < rhoc: cubeindex |= 32
	if gridv[6] < rhoc: cubeindex |= 64
	if gridv[7] < rhoc: cubeindex |= 128
	
	vertex = np.zeros((12,3))

	if edgeTable[cubeindex] == 0:
		return [0,trip]
	if edgeTable[cubeindex] & 1:
		vertex[0] = GridInterp(gridp[0], gridp[1], gridv[0], gridv[1], rhoc)
	if edgeTable[cubeindex] & 2:
		vertex[1] = GridInterp(gridp[1], gridp[2], gridv[1], gridv[2], rhoc)
	if edgeTable[cubeindex] & 4:
		vertex[2] = GridInterp(gridp[2], gridp[3], gridv[2], gridv[3], rhoc)
	if edgeTable[cubeindex] & 8:
		vertex[3] = GridInterp(gridp[3], gridp[0], gridv[3], gridv[0], rhoc)
	if edgeTable[cubeindex] & 16:
		vertex[4] = GridInterp(gridp[4], gridp[5], gridv[4], gridv[5], rhoc)
	if edgeTable[cubeindex] & 32:
		vertex[5] = GridInterp(gridp[5], gridp[6], gridv[5], gridv[6], rhoc)
	if edgeTable[cubeindex] & 64:
		vertex[6] = GridInterp(gridp[6], gridp[7], gridv[6], gridv[7], rhoc)
	if edgeTable[cubeindex] & 128:     
		vertex[7] = GridInterp(gridp[7], gridp[4], gridv[7], gridv[4], rhoc)
	if edgeTable[cubeindex] & 256:
		vertex[8] = GridInterp(gridp[0], gridp[4], gridv[0], gridv[4], rhoc)
	if edgeTable[cubeindex] & 512:
		vertex[9] = GridInterp(gridp[1], gridp[5], gridv[1], gridv[5], rhoc)
	if edgeTable[cubeindex] & 1024:
		vertex[10] = GridInterp(gridp[2], gridp[6], gridv[2], gridv[6], rhoc)
	if edgeTable[cubeindex] & 2048:
		vertex[11] = GridInterp(gridp[3], gridp[7], gridv[3], gridv[7], rhoc)

	ntri = 0
	loop_indices = np.where(triTable[cubeindex][::3] != -1)

	for ii in loop_indices:
		i = ii[0]
		ind = triTable[cubeindex][i]
		trip[ntri][0] = vertex[triTable[cubeindex][i]]
		trip[ntri][1] = vertex[triTable[cubeindex][i+1]]
		trip[ntri][2] = vertex[triTable[cubeindex][i+2]]
		ntri+=1

	return [ntri, trip]

def marching_cubes(rho): 
	"""
	Main implementation of the Marching Cubes algorithm. This is used to smooth out the coarse grain
	density field
	"""
	global verbose

	if verbose >= 1:
		print "---> running marching cubes. this might take a while..."
	# load some more variables into global namespace

	II = 1
	epsilon = 0.000001
	rhoc = 0.1
	gridv = np.zeros(8)
	gridp = np.zeros((8,3))
	cube_coor = np.zeros(3)
	trip = np.zeros((5,3,3))
	ii_coor = np.zeros((II,3))

	for i in range(n_grid_pts[0]):
		start_1 = time.time()
		for j in range(n_grid_pts[1]):
			for k in range(n_grid_pts[2]):
				i1 = i + 1
				j1 = j + 1
				k1 = k + 1
				if i1 >= n_grid_pts[0]: i1 -= n_grid_pts[0]
				if j1 >= n_grid_pts[1]: j1 -= n_grid_pts[1]
				if k1 >= n_grid_pts[2]: k1 -= n_grid_pts[2]
				
				# gridv contains the rho values at the 8 neighboring voxels
				gridv[0] = rho[i][j][k]
				gridv[1] = rho[i][j1][k]
				gridv[2] = rho[i1][j1][k]
				gridv[3] = rho[i1][j][k]
				gridv[4] = rho[i][j][k1]
				gridv[5] = rho[i][j1][k1]
				gridv[6] = rho[i1][j1][k1]
				gridv[7] = rho[i1][j][k1]
				
				# find if the cube is inside bubble, and whether it is near a heavy atom
				cubefactor = 0
				for v in range(8):
					if gridv[v] <= rhoc:
						cubefactor+=1

				# if next to heavy atom
				if cubefactor >=4:
					cube_coor = np.einsum('i,i->i',[i+0.5,j+0.5,k+0.5],grid_spacing)

                                gridp[0] = np.einsum('i,i->i',[i,j,k],grid_spacing)
                                gridp[1] = np.einsum('i,i->i',[i,j+1,k],grid_spacing)
                                gridp[2] = np.einsum('i,i->i',[i+1,j+1,k],grid_spacing)
                                gridp[3] = np.einsum('i,i->i',[i+1,j,k],grid_spacing)
                                gridp[4] = np.einsum('i,i->i',[i,j,k+1],grid_spacing)
                                gridp[5] = np.einsum('i,i->i',[i,j+1,k+1],grid_spacing)
                                gridp[6] = np.einsum('i,i->i',[i+1,j+1,k+1],grid_spacing)
                                gridp[7] = np.einsum('i,i->i',[i+1,j,k+1],grid_spacing)

				[ntri,trip] = MC_table(gridv, gridp, rhoc, trip)

				for nt in range(ntri):
					for qi in range(3):
						vertexflag = 0
                                                for ii in range(II):
                                                        vertexdist = np.dot(trip[nt][qi]-ii_coor[ii],trip[nt][qi]-ii_coor[ii])
                                                        if vertexdist < epsilon:
                                                                vertexflag = 1
                                                                break
                                                if vertexflag == 0:
                                                        ii_coor = np.vstack((ii_coor, np.zeros(3)))
                                                        ii_coor[II] = trip[nt][qi]
                                                        II+=1
                                                elif vertexflag == 1:
                                                        break
		stop_1 = time.time()
		if verbose >= 3:
			print i, "/", n_grid_pts[0]
	return ii_coor

#--- FUNCTION FOR COMPUTING LONG RANGE ELECTROSTATIC POTENTIAL
def compute_VRS(ii_coor,ii_point,water_coor):
	"""
	This function computes the electric potential at *an* interface point from (every) water molecule in the
	system. The electric potential is then weighted by an error function, which gives you the Long-Range
	Electrostatic Potential
	"""

	def erf(x):
		sign = 1 if x >= 0 else -1
		x = abs(x)
		a1 =  0.254829592
		a2 = -0.284496736
		a3 =  1.421413741
		a4 = -1.453152027
		a5 =  1.061405429
		p  =  0.3275911
		t = 1.0/(1.0 + p*x)
		y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
		return sign*y

	vrs=0
	SI_unit_conv = 1.084E8*1.602E-19*1E12 				# pV
	ii_pos = ii_coor[ii_point]
	n_water = water_coor.shape[0]
	for i in range(n_water)[::3]:
		for j in range(3):
			rvec = water_coor[i+j]-ii_pos
			wrap = [int((rvec[i]/pbc[i])+0.5) for i in range(3)]
			rvec = [rvec[i] - wrap[i]*pbc[i] for i in range(3)]
			r = np.sqrt(np.dot(rvec,rvec))
			vrs += chg[j] * erf(r/sigma) / (r)
	return vrs*SI_unit_conv

def compute_refVal(water_coor):
	"""
	This computes the reference value to subtract from every other point in the system. I arbitrarily chose
	the origin, but this is in fact a poor choice for many systems.
	"""

	def erf(x):
		sign = 1 if x >= 0 else -1
		x = abs(x)
		a1 =  0.254829592
		a2 = -0.284496736
		a3 =  1.421413741
		a4 = -1.453152027
		a5 =  1.061405429
		p  =  0.3275911
		t = 1.0/(1.0 + p*x)
		y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
		return sign*y

	vrs=0
	n_water = water_coor.shape[0]
	for i in range(n_water)[::3]:
		for j in range(3):
			r = np.sqrt(np.dot(water_coor[i+j],water_coor[i+j]))
			vrs += chg[j] * erf(r/sigma) / (r)
	return vrs

def compute_LREP(ii_coor,water_coor):
	"""
	This function cycles through *every* interface point, and calls the function that calculates the long-range
	electrostatic potential at each of those points.
	"""
	global verbose

	global chg
	global sigma

	sigma = 0.45
	npro = n_heavy_atoms
	nconf = 1.0
	chg = [-0.834, 0.417, 0.417]	
	sigma = 0.45
	II = len(ii_coor)
	VRS = np.zeros(II)

	if verbose >= 1:
		print 'starting parallel job...'
	### Parallel option: VRS = Parallel(n_jobs=8)(delayed(compute_VRS,has_shareable_memory)(ii_point) for ii_point in range(II)) 
	reminders = int(II/20)
	for ii in range(II):
		if ii%reminders == 0 and verbose >= 3:
			print 'completed', ii, ' out of', II
		VRS[ii] = compute_VRS(ii_coor,ii,water_coor)

	refVal = compute_refVal(water_coor)

	LREP = [V-refVal for V in VRS]
	return LREP 

def compute_av_emaps(fr):
	"""
	This function takes *an* instantaneous interface and computes the LREP at point on the interface
	for every frame. i.e. you only have one II frame and you use it in each frame to compute an average 
	potential
	"""
	global verbose

	#--- RETRIEVE VARIABLES FROM GLOBAL NAMESPACE
	global water
	global first_II_coors

	#--- EXTRACT INFO FOR FRAME, FR
	if verbose >= 3:
		print 'working on frame', fr+1, ' of', nframes
	water_coor = water[fr]

	#--- COMPUTE LONG RANGE ELECTROSTATIC POTENTIAL
	LREP_start = time.time()
	LREP = compute_LREP(first_II_coors,water_coor)
	LREP_stop = time.time()
	if verbose >= 2:
		print 'potential calculation completed. time elapsed:', LREP_stop-LREP_start
	return LREP

def first_II():
	"""
	This function calculates the first instantaneous interface 
	"""
	global verbose

	#--- RETRIEVE VARIABLES FROM GLOBAL NAMESPACE
	global positions

	#--- EXTRACT INFO FOR FRAME, FR
	pos=positions[0] 
	
	#--- COMPUTE RHO
	coarse_grain_start = time.time()
	rho = compute_coarse_grain_density(pos) # defines global variables: n_grid_pts, grid_spacing
	coarse_grain_stop = time.time()
	if verbose >= 2:
		print 'elapsed time to compute coarse grain density:', coarse_grain_stop-coarse_grain_start
	
	#--- MARCHING CUBES
	marching_cubes_start = time.time()
	interface_coors = marching_cubes(rho) # defines global variables: cube_coor
	marching_cubes_stop = time.time()
	if verbose >= 2:
		print 'elapsed time to run marching cubes:', marching_cubes_stop-marching_cubes_start

	return interface_coors

def run_av_emaps():
	"""
	This function calls a function to compute the first instantaneous interface, then uses a parallel scheme to 
	calculate the LREP at each of those interface points for EVERY frame in the trajectory
	"""

	#--- RETRIEVE VARIABLES FROM GLOBAL NAMESPACE
	global first_II_coors
	global nthreads
	global nframes
	frames = range(nframes)
	first_II_coors = first_II()
	all_LREPs = Parallel(n_jobs=nthreads)(delayed(compute_av_emaps,has_shareable_memory)(fr) for fr in frames)
	av_LREP = np.zeros(len(all_LREPs[0]))
	for el in all_LREPs:
		av_LREP += el
	av_LREP/=nframes
	write_pdb(first_II_coors,av_LREP,'av')
	return [0]

def run_emaps(fr):
	"""
	This function takes in a set of II points for each frame and calculates the LREP for each frame using the
	pertinent set of coordinates.
	"""
	global verbose

	#--- RETRIEVE VARIABLES FROM GLOBAL NAMESPACE
	global box_shift
	global scale
	global positions
	global water
	global nframes

	#--- EXTRACT INFO FOR FRAME, FR
	if verbose >= 3:
		print 'working on frame', fr+1, ' of', nframes
	pos=positions[fr] 
	water_coor = water[fr]
	
	#--- COMPUTE RHO
	coarse_grain_start = time.time()
	rho = compute_coarse_grain_density(pos) # defines global variables: n_grid_pts, grid_spacing
	coarse_grain_stop = time.time()
	if verbose >= 2:
		print 'elapsed time to compute coarse grain density:', coarse_grain_stop-coarse_grain_start
	
	#--- MARCHING CUBES
	marching_cubes_start = time.time()
	interface_coors = marching_cubes(rho) # defines global variables: cube_coor
	marching_cubes_stop = time.time()
	if verbose >= 2:
		print 'elapsed time to run marching cubes:', marching_cubes_stop-marching_cubes_start
	
	##--- COMPUTE LONG RANGE ELECTROSTATIC POTENTIAL
	LREP_start = time.time()
	#interface_coors *= scale
	#water_coor *= scale
	LREP = compute_LREP(interface_coors,water_coor)
	LREP_stop = time.time()
	if verbose >= 2:
		print 'potential calculation completed. time elapsed:', LREP_stop-LREP_start
	interface_coors *= scale
	write_pdb(interface_coors,LREP,fr)
	return 0
	
def electrostatic_map(grofile,trajfile,**kwargs):
	"""
	This is the MAIN function. It reads in all data and decides which functions should be called to
	compute either averaged or individual electrostatic maps.
	"""
	
	#--- UNPACK UPSTREAM DATA
	global sn
	global work
	sn = kwargs['sn']
	work = kwargs['workspace']

	print work.postdir
	outfile = open(str(work.postdir)+str(sn)+"_emap_"+str(fr)+str(name_modifier)+".pdb","w")
	exit()

	#--- READ IN PARAMETERS FROM YAML FILE
	global selection_key
	global water_resname
	global dL
	global name_modifier
	global verbose
	if 'selection_key' in kwargs['calc']['specs']['selector']:
		selection_key = kwargs['calc']['specs']['selector']['selection_key']
	else: 
		print 'need to provide selection key in yaml file'
		exit()

	if 'water_resname' in kwargs['calc']['specs']['selector']:
		water_resname = kwargs['calc']['specs']['selector']['water_resname']
	else: 
		print 'did not provide water_resname in yaml file. Using water_resname = resname TIP3'
		water_resname = "resname TIP3"

	if 'grid_spacing' in kwargs['calc']['specs']['selector']:
		dL = kwargs['calc']['specs']['selector']['grid_spacing']
	else: dL = 0.1 

	if 'name_modifer' in kwargs['calc']['specs']:
		name_modifier = "_" + str(kwargs['calc']['specs']['name_modifier'])
	else: name_modifier = ""

	if 'verbose' in kwargs['calc']['specs']:
		verbose = kwargs['calc']['specs']['verbose']
	else: verbose = 1

	if 'writepdb' in kwargs['calc']['specs']:
		writepdb = 'y'
	else: writepdb = 'n'

	if 'average_frames' in kwargs['calc']['specs']:
		av_LREP = 'y'
	else: av_LREP = 'n'

	#--- LOAD VARIABLES INTO GLOBAL NAMESPACE
	global box_shift
	global scale
	global positions
	global water
	global nframes
	global nthreads
	scale = 10.0

	#--- READ DATA
	[nframes, positions, water] = extract_traj_info(grofile,trajfile,selection_key)
			# defines global variables: n_heavy_atoms, pbc

	#--- LOOP THROUGH FRAMES IN TRAJECTORY
	frames = range(nframes)
	#--- use max 1 thread per frame
	if nframes<8:
		nthreads = nframes
	else:
		nthreads = 8

	#--- user has option to make one electrostatic map that represents the average potential at (1) instantaneous interface
	#	over the course of the simulation. This is recommended by Remsing & Weeks, but is not always conducive to
	#	the simulation set up
	if av_LREP != 'y':
		check = Parallel(n_jobs=nthreads)(delayed(run_emaps,has_shareable_memory)(fr) for fr in frames)
	#--- determine (1) instantaneous interface and use those points to calculate the potential at each frame. Then divide
	# 	by the number of frames
	else:
		check = run_av_emaps()
	
	#--- PACK UP RESULTS AND SEND BACK TO OMNICALC
	if all(c == 0 for c in check):
		attrs,result = {},{}
		return attrs,result	
	else:
		print "something went wrong..."
