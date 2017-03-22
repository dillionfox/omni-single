#!/usr/bin/env python

import time
import numpy as np
import MDAnalysis
from joblib import Parallel,delayed
from joblib.pool import has_shareable_memory
from base.tools import status,framelooper
from base.timer import checktime
import codes.mesh

def lipid_abstractor(grofile,trajfile,**kwargs):

	"""
	LIPID ABSTRACTOR
	Reduce a bilayer simulation to a set of points.
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

	#---center of mass over residues
	if 'type' in selector and selector['type'] == 'com' and 'resnames' in selector:
		resnames = selector['resnames']
		selstring = '('+' or '.join(['resname %s'%i for i in resnames])+')'
	elif 'type' in selector and selector['type'] == 'select' and 'selection' in selector:
		selstring = selector['selection']
	else: raise Exception('\n[ERROR] unclear selection %s'%str(selector))
	
	#---compute masses by atoms within the selection
	sel = uni.select_atoms(selstring)
	mass_table = {'H':1.008,'C':12.011,'O':15.999,'N':14.007,'P':30.974}
	#---martini mass table estimated from looking at sel.atoms.names and from martini-2.1.itp
	mass_table = {'C':72,'N':72,'P':72,'S':45,'G':72,'D':72,'R':72}
	masses = np.array([mass_table[i[0]] for i in sel.atoms.names])
	resids = sel.resids
	#---create lookup table of residue indices
	divider = [np.where(resids==r) for r in np.unique(resids)]

	#---load trajectory into memory	
	trajectory,vecs = [],[]
	for fr in range(nframes):
		status('loading frame',tag='load',i=fr,looplen=nframes)
		uni.trajectory[fr]
		trajectory.append(sel.positions/lenscale)
		vecs.append(sel.dimensions[:3])
	vecs = np.array(vecs)/lenscale

    #---alternate lipid representation is useful for separating monolayers
	monolayer_cutoff = kwargs['calc']['specs']['separator']['monolayer_cutoff']
	if 'lipid_tip' in kwargs['calc']['specs']['separator']:
		tip_select = kwargs['calc']['specs']['separator']['lipid_tip']
		sel = uni.select_atoms(tip_select)
		atoms_separator = []
		for fr in range(nframes):
			status('loading lipid tips',tag='load',i=fr,looplen=nframes)
			uni.trajectory[fr]
			atoms_separator.append(sel.coordinates()/lenscale)
	else: atoms_separator = coms

	#---identify monolayers
	status('identify leaflets',tag='compute')
	#---randomly select frames for testing monolayers
	random_tries = 3
	for fr in [0]+[np.random.randint(nframes) for i in range(random_tries)]:
		monolayer_indices = codes.mesh.identify_lipid_leaflets(atoms_separator[fr],vecs[fr],
			monolayer_cutoff=monolayer_cutoff)
		if type(monolayer_indices)!=bool: break
	checktime()

	#---parallel
	start = time.time()
	if parallel:
		coms = Parallel(n_jobs=work.nprocs,verbose=0)(
			delayed(codes.mesh.centroid)(trajectory[fr],masses,divider)
			for fr in framelooper(nframes,start=start))
	else:
		coms = []
		for fr in range(nframes):
			status('computing centroid',tag='compute',i=fr,looplen=nframes,start=start)
			coms.append(codes.mesh.centroid(trajectory[fr],masses,divider))
	checktime()
	coms_out = np.array(coms)

	#---remove jumping in some directions if requested
	if nojumps:
		nojump_dims = ['xyz'.index(j) for j in nojumps]
		nobjs = coms_out.shape[1]
		displacements = np.array([(coms_out[1:]-coms_out[:-1])[...,i] for i in range(3)])
		for d in nojump_dims:
			shift_binary = (np.abs(displacements)*(1.-2*(displacements<0))/
				(np.transpose(np.tile(vecs[:-1],(nobjs,1,1)))/2.))[d].astype(int)
			shift = (np.cumsum(-1*shift_binary,axis=0)*np.transpose(np.tile(vecs[:-1,d],(nobjs,1))))
			coms_out[1:,:,d] += shift

	#---pack
	attrs,result = {},{}
	#attrs['selector'] = selector
	attrs['nojumps'] = nojumps
	result['resnames'] = np.array(sel.residues.resnames)
	result['monolayer_indices'] = np.array(monolayer_indices)
	result['vecs'] = vecs
	result['nframes'] = np.array(nframes)
	result['points'] = coms_out
	result['resids'] = np.array(np.unique(resids))
	attrs['separator'] = kwargs['calc']['specs']['separator']
	return result,attrs	

def topologize(pos,vecs):

	"""
	Join a bilayer which is broken over periodic boundary conditions by translating each point by units
	of length equal to the box vectors so that there is a maximum amount of connectivity between adjacent
	points. This method decides how to move the points by a consensus heuristic.
	This function is necessary only if the bilayer breaks over a third spatial dimension.
	"""

	step = 0
	kp = array(pos)
	natoms = len(pos)
	move_votes = zeros((1,natoms,3))
	while sum(abs(move_votes[0])>len(pos)/2)>len(pos)*0.05 or step == 0:
		move_votes = concatenate((move_votes,zeros((1,natoms,3))))
		pos = array(kp)
		pd = [scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(pos[:,d:d+1])) 
			for d in range(3)]
		for ind,bead in enumerate(pos):
			#---compute distances to the probe
			for d in range(3):
				adds_high = zeros(natoms)
				adds_low = zeros(natoms)
				adds_high[where(all((pd[d][ind]>vecs[d]/2.,pos[ind,d]<pos[:,d]),axis=0))] = 1
				adds_low[where(all((pd[d][ind]>vecs[d]/2.,pos[ind,d]>pos[:,d]),axis=0))] = -1
				#---for each spatial dimension, we tally votes to move the point by a box vector
				move_votes[step][:,d] += adds_high+adds_low
		kp = array(pos)
		step += 1
		move_votes = concatenate((move_votes,zeros((1,natoms,3))))
	moved = transpose([sum([1*(move_votes[it][:,d]<(-1*natoms/2.))-1*(move_votes[it][:,d]>(natoms/2.)) 
		for it in range(step+1)],axis=0) for d in range(3)])
	return moved