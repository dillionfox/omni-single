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
	global water_bound_min
        global water_bound_max
	global interface_vector
	global edge_padding
	global vec_to_int
	global int_to_vec
	global nres

	print "frame:", fr
	resids = np.zeros(nres)
	uni = MDAnalysis.Universe(PSF,DCD)
	uni.trajectory[fr]
	protein = uni.select_atoms('global protein and ((prop '+ str(interface_vector) + ' < '+ str(water_bound_min+edge_padding) + ') or (prop ' + str(interface_vector) + ' > '+ str(water_bound_max-edge_padding) + '))')
	# if protein is closer to left edge, then we are looking for things that are more negative 
	protein_av = protein.positions[:,vec_to_int[interface_vector]].mean(axis=0)
	if protein_av-water_bound_min < water_bound_max-protein_av:
		ineq = '<'
	else:
		ineq = '>'

	for i in protein.indices:
		p = uni.select_atoms('bynum ' + str(i))
		pc = p.positions.mean(axis=0)
		# example where protein atom is at (-19, 20,10)
		# water and x < -19 
		iv = interface_vector # shorter name
		dl = probe_radius # shorter name
		v0 = vec_to_int[iv]; v1 = (v0+1)%3; v2 = (v0+2)%3
		iv1 = int_to_vec[v1]; iv2 = int_to_vec[v2]
			#    water        and        x        <        -19
		selstring = 'resname TIP3 and prop '+str(iv)+ineq+str(pc[v0])
		selstring += ' and prop ' + str(iv1) + ' > ' + str(pc[v1]-dl) 
		selstring += ' and prop ' + str(iv1) + ' < ' + str(pc[v1]+dl) 
		selstring += ' and prop ' + str(iv2) + ' > ' + str(pc[v2]-dl) 
		selstring += ' and prop ' + str(iv2) + ' < ' + str(pc[v2]+dl) 
		water_sel = uni.select_atoms(selstring)
		if len(water_sel) == 0:
			resids[p.residues.resids-1] = 1
		
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

	if 'probe_radius' in kwargs['calc']['specs']:
		probe_radius = kwargs['calc']['specs']['probe_radius']
	else: 
		print 'did not provide probe radius in yaml file. Using cutoff = 2.0'
		probe_radius = 2.0

	if 'interface_vector' in kwargs['calc']['specs']:
		first_frame = kwargs['calc']['specs']['interface_vector']
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

	interface_vector = 'x'
	edge_padding = 10

	protein = uni.select_atoms('protein')
	nres = len(protein.residues)
	resids_per_frame = np.zeros((nframes,nres))
	water_bound_min = uni.select_atoms('resname TIP3').positions[:,vec_to_int[interface_vector]].min()
	water_bound_max = uni.select_atoms('resname TIP3').positions[:,vec_to_int[interface_vector]].max()
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

