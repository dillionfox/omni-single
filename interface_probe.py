#!/usr/bin/python

import time
import numpy as np
import MDAnalysis
from joblib import Parallel,delayed
from joblib.pool import has_shareable_memory
from base.tools import status,framelooper
from base.timer import checktime
import codes.mesh

def probe(fr):
	"""
	"""
	global PSF
	global DCD
	global probe_radius
	global resids_per_frame
	global interface_vector
	global edge_padding
	global vec_to_int
	global int_to_vec
	global nres
	global model

	print "frame:", fr
	uni = MDAnalysis.Universe(PSF,DCD)
	uni.trajectory[fr]

	bilayer = uni.select_atoms('resname POPC')
	selstring = 'around ' + str(5.0) +  ' global (name BB)'

	lipids = bilayer.select_atoms(selstring)
	lipid_av = lipids.positions[:,vec_to_int[interface_vector]].mean(axis=0)

	protein_all = uni.select_atoms('global (name BB)')
	resids = np.zeros(len(protein_all.residues.resids))
	selstring = 'around ' + str(5.0) +  ' global resname POPC'
	protein = protein_all.select_atoms(selstring)
	protein_av = protein.positions[:,vec_to_int[interface_vector]].mean(axis=0)

	if protein_av > lipid_av: 	# if protein is ABOVE bilayer
		ineq = '<'		# then we are looking for water BELOW protein
		ineq_ = '>'
	else: 				# if protein is BELOW bilayer
		ineq = '>'		# then we are looking for water ABOVE protein
		ineq_ = '<'

	for i in protein.residues.resids:
		pc = protein_all.positions[i-1]
		iv = interface_vector # shorter name
		dl = probe_radius # shorter name
		v0 = vec_to_int[iv]; v1 = (v0+1)%3; v2 = (v0+2)%3
		iv1 = int_to_vec[v1]; iv2 = int_to_vec[v2]

		selstring = '(resname W or name BB) and (prop '+str(iv)+ineq+str(pc[v0])
		selstring += ' and prop ' + str(iv) + ineq_ + str(lipid_av) 
		selstring += ' and prop ' + str(iv1) + ' > ' + str(pc[v1]-dl) 
		selstring += ' and prop ' + str(iv1) + ' < ' + str(pc[v1]+dl) 
		selstring += ' and prop ' + str(iv2) + ' > ' + str(pc[v2]-dl) 
		selstring += ' and prop ' + str(iv2) + ' < ' + str(pc[v2]+dl) 
		selstring += ')'
		water_sel = uni.select_atoms(selstring)

		if len(water_sel) == 0:
			print "contact:", i
			resids[i-1] = 1
	return resids 

def interface_probe(grofile,trajfile,**kwargs):

	"""
	PROTEIN LIPID CONTACTS
	Idenitify simulation frames where protein is in contact with membrane
	"""

	global PSF
	global DCD
	global probe_radius
	global resids_per_frame
	global water_bound_min
        global water_bound_max
	global interface_vector
	global edge_padding
	global vec_to_int
	global int_to_vec
	global nres
	global lipid_av
	global model

	if 'probe_radius' in kwargs['calc']['specs']:
		probe_radius = kwargs['calc']['specs']['probe_radius']
	else: 
		print 'did not provide probe radius in yaml file. Using cutoff = 2.0'
		probe_radius = 2.0

	if 'interface_vector' in kwargs['calc']['specs']:
		interface_vector = kwargs['calc']['specs']['interface_vector']
	else: 
		print "need to specify interface vector. Exiting"
		exit()

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

	if 'model' in kwargs['calc']['specs']:
		model = kwargs['calc']['specs']['model']
	else: model = 'AA' # all atom, CG: coarse-grained

	PSF = grofile; DCD = trajfile

	#---unpack
	sn = kwargs['sn']
	work = kwargs['workspace']
	uni = MDAnalysis.Universe(PSF,DCD)

	nframes = len(uni.trajectory)
	if first_frame > nframes:
		print 'first frame exceeds number of frames. Fix it! Exiting...'
		exit()
	if last_frame > nframes:
		print 'last frame exceeds number of frames. using number of frames as last frame'
		last_frame = nframes
	if nthreads > nframes:
		nthreads = nframes

	vec_to_int = {'x':0, 'y':1, 'z':2}
	int_to_vec = {0:'x', 1:'y', 2:'z'}

	edge_padding = 10

	if model != 'CG':
		protein = uni.select_atoms('protein')
	else:
		protein = uni.select_atoms('not resname ION W POPC')
	nres = len(protein.residues)
	first_resid = protein.residues.resids[0]
	resids_per_frame = np.zeros((nframes,nres))
	uni = []

	contact_frames = np.zeros(nframes)
	frames = range(first_frame,last_frame)
	#resids_per_frame = [probe(fr) for fr in frames]
	resids_per_frame = Parallel(n_jobs=nthreads)(delayed(probe,has_shareable_memory)(fr) for fr in frames)
	
	#---pack
	attrs,result = {},{}
	result['frames'] = np.array(contact_frames) # contacts has to be a numpy array
	result['resids_per_frames'] = resids_per_frame
	return result,attrs	

