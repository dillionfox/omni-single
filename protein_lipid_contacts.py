#!/usr/bin/python

import time
from numpy import *
import MDAnalysis
from joblib import Parallel,delayed
from joblib.pool import has_shareable_memory
from base.tools import status,framelooper
from base.timer import checktime
from codes.mesh import *

def protein_lipid_contacts(grofile,trajfile,**kwargs):

	"""
	PROTEIN LIPID CONTACTS
	Idenitify simulation frames where protein is in contact with membrane
	"""

	cutoff = 2.0

	#---unpack
	sn = kwargs['sn']
	work = kwargs['workspace']
	#parallel = kwargs.get('parallel',False)
	uni = MDAnalysis.Universe(grofile,trajfile)

	nframes = len(uni.trajectory)
	protein = uni.select_atoms('protein')
	nres = len(protein.residues)
	resids_per_frame = zeros((nframes,nres))

	contact_frames = np.zeros(nframes)
	start = time.time()
	for fr in range(nframes):
		uni.trajectory[fr]
		protein = uni.select_atoms('global protein')
		selstring = 'around ' + str(cutoff) +  ' global resname POPC or resname DOPC or resname POPE'
		contacts = protein.select_atoms(selstring)

		if len(contacts.atoms) > 0:
			contact_frames[fr] = 1
		else:
			contact_frames[fr] = 0 

		reslist = contacts.residues.resids
		for res in reslist:
			resids_per_frame[fr][res-1] = 1

	#---pack
	attrs,result = {},{}
	result['frames'] = array(contact_frames) # contacts has to be a numpy array
	result['resids_per_frames'] = resids_per_frame
	return result,attrs	

