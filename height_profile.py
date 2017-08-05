#!/usr/bin/env python

import time
import numpy as np
import MDAnalysis
from joblib import Parallel,delayed
from joblib.pool import has_shareable_memory
from base.tools import status,framelooper
from base.timer import checktime
import codes.mesh

def height_profile(grofile,trajfile,**kwargs):

	"""
	HEIGHT PROFILE
	Generate curvature map based on average height of lipids - always center maps based on protein position
	"""

	#---unpack
	sn = kwargs['sn']
	work = kwargs['workspace']
	parallel = kwargs.get('parallel',False)
	#---prepare universe	
	uni = MDAnalysis.Universe(grofile,trajfile)
	nframes = len(uni.trajectory)
	#---MDAnalysis uses Angstroms not nm
	lenscale = 10.
	#---select residues of interest
	selector = kwargs['calc']['specs']['selector']
	nojumps = kwargs['calc']['specs'].get('nojumps','')


	#---pack
	attrs,result = {},{}
	attrs['selector'] = selector
	attrs['nojumps'] = nojumps
	#result['resnames'] = np.array(sel.residues.resnames)
	#result['monolayer_indices'] = np.array(monolayer_indices)
	#result['vecs'] = vecs
	#result['nframes'] = np.array(nframes)
	#result['points'] = coms_out
	#result['resids'] = np.array(np.unique(resids))
	#result['resids_exact'] = resids
	attrs['separator'] = kwargs['calc']['specs']['separator']
	return result,attrs	
