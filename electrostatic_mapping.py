from __future__ import division
import numpy as np
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import MDAnalysis
import time

selection_key = "resname POPC"
trajfile = "membrane_example/sample.dcd"
grofile = "membrane_example/sample.psf"
II_traj = "membrane_example/II_0.pdb"

global scale
scale = 1.0

# still need to:	 
#	test on more cases and 
#	import all possible water resnames

def electrostatic_mapping((grofile,trajfile,**kwargs):

	""" 
	ELECTROSTATIC MAPPING 
	Computes long range electrostatic potential derived
	from local molecule field theory. See Remsing & Weeks paper
	"""

	#---unpack
	sn = kwargs['sn']
	work = kwargs['workspace']

	global water_resname

	if 'water_resname' in kwargs['calc']['specs']['selector']:
		water_resname = kwargs['calc']['specs']['selector']['water_resname']
	else: 
		print 'need to provide resname of water in yaml file. this is a quick fix'
		return 0
	print 'this code is currently only set up for TIP3P water. If you are using a different', \
		'water model, you need to update the vector called (chg) that stores the charges'

	II_traj = kwargs['upstream']['instant_interface']['interface_coors']	

	if 'writepdb' in kwargs['calc']['specs']:
		writepdb = 'y'
	else: writepdb = 'n'

	def write_pdb(coor,beta,fr):
		print 'writing pdb...'
		outfile = open("emap_"+str(fr)+".pdb","w")
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
	
	def extract_traj_info(grofile,trajfile,selection_key):
	        print 'loading...'
	        # load some variables into global namespace
	        global n_heavy_atoms
	        global pbc
		global box_shift
	
	        uni = MDAnalysis.Universe(grofile,trajfile)
	        nframes = len(uni.trajectory)					# number of frames
		box_shift = np.zeros((nframes,3))
	
	        protein = uni.select_atoms(selection_key)			# identify atoms to build interface around
	        heavy_atoms = protein.select_atoms('not name H*')		# Only need to consider heavy atoms
	        protein_indices = heavy_atoms.indices 
	        n_heavy_atoms = len(heavy_atoms.atoms)                          # number of heavy protein atoms
	        positions = np.zeros((nframes,n_heavy_atoms,3))
	
	        water = uni.select_atoms(water_resname)				# identify atoms to build interface around
	        water_indices = water.indices 
	        n_water = len(water.atoms)					# number of heavy protein atoms
	        water_pos = np.zeros((nframes,n_water,3))
	
	        for fr in range(nframes):                                       # save coordinates for each frame
	                uni.trajectory[fr]
	        	sel = uni.select_atoms('all')
			box_shift[fr] = -sel.atoms.center_of_mass()
	        	sel.atoms.translate(box_shift)				# center selection
	
	        	protein = uni.select_atoms(selection_key)		# identify atoms to build interface around
	        	heavy_atoms = protein.select_atoms('not name H*')	# Only need to consider heavy atoms
			positions[fr] = heavy_atoms.positions/scale
	
	        	water = uni.select_atoms(water_resname)			# identify atoms to build interface around
			water_pos[fr] = water.positions/scale
	
	        pbc = uni.dimensions[0:3]/scale					# retrieve periodic bounds
	        return [nframes,positions,water_pos]
	
	def read_ii_coor(II_traj):
		uni = MDAnalysis.Universe(II_traj)
	        nframes = len(uni.trajectory)					# number of frames
		sel = uni.select_atoms('all')
		II = len(sel.atoms)
		ii_coor_all = np.zeros((nframes,II,3))
		for fr in range(nframes):
	                uni.trajectory[fr]
			sel = uni.select_atoms('all')
	        	sel.atoms.translate(box_shift[fr])			# center selection
			ii_coor_all[fr] = sel.positions/scale
		return [ii_coor_all,nframes]
	
	def compute_VRS(ii_point):
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
		for i in range(n_water)[::3]:
			for j in range(3):
				rvec = water_coor[i+j]-ii_pos
				wrap = [int((rvec[i]/pbc[i])+0.5) for i in range(3)]
				rvec = [rvec[i] - wrap[i]*pbc[i] for i in range(3)]
				r = np.sqrt(np.dot(rvec,rvec))
				vrs += chg[j] * erf(r/sigma) / (r)
		return vrs*SI_unit_conv
	
	def compute_refVal():
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
		for i in range(n_water)[::3]:
			for j in range(3):
				r = np.sqrt(np.dot(water_coor[i+j],water_coor[i+j]))
				vrs += chg[j] * erf(r/sigma) / (r)
		return vrs
	
	def scale_potential(ii):
		return VRS[ii]-refVal
	
	def compute_LREP(positions,ii_coor,water_coor):
	
		global VRS
		global chg
		global n_water
		global sigma
	
		sigma = 0.45
		npro = n_heavy_atoms
		nconf = 1.0
		chg = [-0.834, 0.417, 0.417]	
		sigma = 0.45
		II = len(ii_coor)
		VRS = np.zeros(II)
	
		n_water = water_coor.shape[0]
		print 'starting parallel job...'
		VRS = Parallel(n_jobs=8)(delayed(compute_VRS,has_shareable_memory)(ii_point) for ii_point in range(II)) 
	
		global refVal
		refVal = compute_refVal()
	
		LREP = Parallel(n_jobs=8)(delayed(scale_potential,has_shareable_memory)(ii) for ii in range(II))
		return LREP 
	
		time1 =time.time()
		[nframes, positions,water] = extract_traj_info(grofile,trajfile,selection_key)
		[II_coor,nframes_ii] = read_ii_coor(II_traj)
		time2 =time.time()
		print 'data loaded. computing potential... time:', time2-time1
		if nframes_ii != nframes:
			print 'simulations do not match'
			nframes = min(nframes,nframes_ii)
	
		global water_coor
		global ii_coor
		time1 =time.time()
		for fr in range(nframes):
			pos = positions[fr]
			water_coor = water[fr]
			ii_coor = II_coor[fr]
	
			LREP = compute_LREP(pos, ii_coor,water_coor)
			print "ii_coor:", ii_coor.shape, ", LREP:", len(LREP)
			print 'potential calculation completed. writing pdb...'
			write_pdb(ii_coor,LREP,fr)
		time2 =time.time()
		print 'trajectory loop time:', time2-time1
	
	#---pack
	attrs,result = {},{}
	result['electrostatic_map'] = array(LREP) # contacts has to be a numpy array
	return result,attrs	
