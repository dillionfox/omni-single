from __future__ import division
import numpy as np
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import time
import MDAnalysis
import CW_interface

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
        global pbc_fr
	global box_shift
	global first_frame
	global last_frame
	global water_names

        uni = MDAnalysis.Universe(PSF,DCD)
        nframes = len(uni.trajectory)					# number of frames
	box_shift = np.zeros((nframes,3))
	pbc_fr = np.zeros((nframes,3))

        protein = uni.select_atoms(selection_key)			# identify atoms to build interface around
        heavy_atoms = protein.select_atoms('not name H*')		# Only need to consider heavy atoms
        protein_indices = heavy_atoms.indices 
        n_heavy_atoms = len(heavy_atoms.atoms)                          # number of heavy protein atoms
        positions = np.zeros((nframes,n_heavy_atoms,3))

        water = uni.select_atoms("resname TIP3")			# identify atoms to build interface around
        water_indices = water.indices 
	water_names = water.names
        n_water = len(water.atoms)					# number of heavy protein atoms
        water_pos = np.zeros((nframes,n_water,3))

	frames = range(nframes)
        for fr in frames:                                       # save coordinates for each frame
                uni.trajectory[fr]
        	pbc = uni.dimensions[0:3]				# retrieve periodic bounds
        	pbc_fr[fr] = pbc
        	sel = uni.select_atoms('all')
		box_shift[fr] = -sel.atoms.center_of_mass()+pbc/2.0	# first center at origin, then put vertx of quadrant 7 at origin
        	sel.atoms.translate(box_shift[fr])			# center selection

        	protein = uni.select_atoms(selection_key)		# identify atoms to build interface around
        	heavy_atoms = protein.select_atoms('not name H*')	# Only need to consider heavy atoms
		positions[fr] = heavy_atoms.positions

        	water = uni.select_atoms("resname TIP3")		# identify atoms to build interface around
		water_pos[fr] = water.positions

        pbc = uni.dimensions[0:3]					# retrieve periodic bounds
        return [nframes,positions,water_pos]

#--- FUNCTION FOR COMPUTING LONG RANGE ELECTROSTATIC POTENTIAL
def compute_potential(ii_coor,water_coor):
	"""
	This function cycles through *every* interface point, and calls the function that calculates the long-range
	electrostatic potential at each of those points.
	"""
	global verbose
	global chg
	global sigma
	global water_names
	global potential

	sigma = 4.5
	npro = n_heavy_atoms
	nconf = 1.0
	water_charge = {"OH2":-0.834, "H1":0.417, "H2":0.417}
	II = len(ii_coor)
	VRS = np.zeros(II)
	SI_unit_conv = 1.084E8*1.602E-19*1E12 				# pV
	n_water = water_coor.shape[0]

	def _LREP(wc,r): return water_charge[wn]*CW_interface.erf(r/sigma)/r 
	def _EP(wc,r): return water_charge[wn]/r 

	for ii in range(II):
		print ii, "of", II
		vrs=0
		ii_pos = ii_coor[ii]
		for wc,wn in zip(water_coor,water_names):
			rvec = wc-ii_pos
			wrap1 = [int((rvec[i]/pbc[i])+0.5) for i in range(3)]
			wrap2 = [int((rvec[i]/pbc[i])-0.5) for i in range(3)]
			rvec = [rvec[i] - wrap1[i]*pbc[i] for i in range(3)]
			rvec = [rvec[i] - wrap2[i]*pbc[i] for i in range(3)]
			r = np.sqrt(np.dot(rvec,rvec))
			vrs += water_charge[wn] * CW_interface.erf(r/sigma) / (r)
			if potential == 'LREP':
				vrs += _LREP(water_charge[wn],r)
			else:
				vrs += _EP(water_charge[wn],r)
		VRS[ii] = vrs*SI_unit_conv
	return VRS 

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
	LREP = compute_potential(first_II_coors,water_coor)
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
	rho = CW_interface.compute_coarse_grain_density(pos,dL,pbc,n_heavy_atoms) # defines global variables: n_grid_pts, grid_spacing
	coarse_grain_stop = time.time()
	if verbose >= 2:
		print 'elapsed time to compute coarse grain density:', coarse_grain_stop-coarse_grain_start
	
	#--- MARCHING CUBES
	marching_cubes_start = time.time()
	interface_coors = CW_interface.marching_cubes(rho) # defines global variables: cube_coor
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
	global first_frame
	global last_frame
	frames = range(first_frame,last_frame)
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
	global box_shift
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
	rho = CW_interface.compute_coarse_grain_density(pos,dL,pbc,n_heavy_atoms) # defines global variables: n_grid_pts, grid_spacing
	coarse_grain_stop = time.time()
	if verbose >= 2:
		print 'elapsed time to compute coarse grain density:', coarse_grain_stop-coarse_grain_start
	
	#--- MARCHING CUBES
	marching_cubes_start = time.time()
	interface_coors = CW_interface.marching_cubes(rho) # defines global variables: cube_coor
	marching_cubes_stop = time.time()
	if verbose >= 2:
		print 'elapsed time to run marching cubes:', marching_cubes_stop-marching_cubes_start
	
	##--- COMPUTE LONG RANGE ELECTROSTATIC POTENTIAL
	LREP_start = time.time()
	LREP = compute_potential(interface_coors,water_coor)
	LREP_stop = time.time()
	if verbose >= 2:
		print 'potential calculation completed. time elapsed:', LREP_stop-LREP_start
	write_pdb(interface_coors,LREP,fr)
	return 0

def electrostatic_mapping(grofile,trajfile,**kwargs):
	"""
	This is the MAIN function. It reads in all data and decides which functions should be called to
	compute either averaged or individual electrostatic maps.
	"""
	
	#--- UNPACK UPSTREAM DATA
	global sn
	global work
	sn = kwargs['sn']
	work = kwargs['workspace']

	#--- READ IN PARAMETERS FROM YAML FILE
	global selection_key
	global water_resname
	global dL
	global name_modifier
	global verbose
	global nthreads
	global first_frame
	global last_frame
	global potential

	if 'grid_method' in kwargs['calc']['specs']['selector']:
		grid_method = kwargs['calc']['specs']['selector']['grid_method']
		print "you are using the grid method which computes the potential at evenly spaced grid points instead of instantaneous interfaces"
	else:
		grid_method = 'n'

	if 'selection_key' in kwargs['calc']['specs']['selector']:
		selection_key = kwargs['calc']['specs']['selector']['selection_key']
	elif grid_method != 'y': 
		print 'need to provide selection key in yaml file'
		exit()
	else:
		selection_key = "resid 0"

	if 'grid_spacing' in kwargs['calc']['specs']['selector']:
		dL = kwargs['calc']['specs']['selector']['grid_spacing']
		if grid_method != 'y':
			if dL > 1:
				print "you are using a grid spacing of:", dL, ". The recommended value is 1.0. This may cause issues."
			if dL < 1:
				print "you are using a grid spacing of:", dL, ". The recommended value is 1.0. You can probably increase this to save time."
	elif grid_method == 'y': 
		dL = 5
	else:
		dL = 1 

	if 'average_frames' in kwargs['calc']['specs']:
		av_LREP = kwargs['calc']['specs']['average_frames']
	else: av_LREP = 'n'

	if 'potential' in kwargs['calc']['specs']:
		potential = kwargs['calc']['specs']['potential']
	else: 
		print "Since no potential was specified, the long range electrostatic potential will be used by default."
		potential = 'LREP'

	if 'water_resname' in kwargs['calc']['specs']['selector']:
		water_resname = kwargs['calc']['specs']['selector']['water_resname']
	else: 
		print 'did not provide water_resname in yaml file. Using water_resname = resname TIP3'
		water_resname = "resname TIP3"

	if 'name_modifier' in kwargs['calc']['specs']:
		name_modifier = "_" + str(kwargs['calc']['specs']['name_modifier'])
	else: name_modifier = ""

	if 'verbose' in kwargs['calc']['specs']:
		verbose = kwargs['calc']['specs']['verbose']
	else: verbose = 1

	if 'first_frame' in kwargs['calc']['specs']:
		first_frame = kwargs['calc']['specs']['first_frame']
	else: first_frame = 1

	if 'last_frame' in kwargs['calc']['specs']:
		last_frame = kwargs['calc']['specs']['last_frame']
	else: last_frame = -1

	if 'nthreads' in kwargs['calc']['specs']:
		nthreads = kwargs['calc']['specs']['nthreads']
	else:
		nthreads = 8

	if 'writepdb' in kwargs['calc']['specs']:
		writepdb = 'y'
	else: writepdb = 'n'

	#--- LOAD VARIABLES INTO GLOBAL NAMESPACE
	global box_shift
	global positions
	global water
	global nframes

	#--- READ DATA
	[nframes, positions, water] = extract_traj_info(grofile,trajfile,selection_key)
			# defines global variables: n_heavy_atoms, pbc

	#--- adjust number of threads so there aren't more threads than frames
	if nthreads > nframes:
		nthreads = nframes

	#--- LOOP THROUGH FRAMES IN TRAJECTORY
	if last_frame == -1:
		print "fuck ", nframes
		last_frame = nframes
	if first_frame > nframes:
		print "first frame requested is larger than the total number of frames!"
		print nframes
		print last_frame
		print first_frame
		exit()
	frames = range(first_frame,last_frame)

	#--- you can either compute electrostatic maps at instantaneous interfaces or on a uniformly spaced grid throughout the simulation
	#--- INSTANTANEOUS INTERFACE METHOD
	if grid_method != 'y':
		#--- user has option to make one electrostatic map that represents the average potential at (1) instantaneous interface
		#	over the course of the simulation. This is recommended by Remsing & Weeks, but is not always conducive to
		#	the simulation set up
		if av_LREP != 'y':
			check = Parallel(n_jobs=nthreads)(delayed(run_emaps,has_shareable_memory)(fr) for fr in frames)
		#--- determine (1) instantaneous interface and use those points to calculate the potential at each frame. Then divide
		# 	by the number of frames
		else:
			check = run_av_emaps()
	#--- GRID METHOD
	else:
		check = run_grid_method()

	#--- PACK UP RESULTS AND SEND BACK TO OMNICALC
	if all(c == 0 for c in check):
		attrs,result = {},{}
		return attrs,result	
	else:
		print "something went wrong..."
