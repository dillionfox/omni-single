#!/usr/bin/env python

"""
Plot all of the upstream loops for the curvature undulation coupling analysis.

"""

from codes.curvature_coupling.curvature_coupling_plots import individual_reviews_plotter

#---autoplot settings
plotrun.routine = ['individual_reviews']
global seepspace

@autoload(plotrun)
def loader():
	"""Load data."""
	#---only load once
	if 'data' not in globals():
		#---begin loading sequence
		if plotname not in work.plots: raise Exception('add %s to the plots metadata'%plotname)
		plotspecs = work.plots[plotname].get('specs',{})
		calcname = plotspecs.get('calcname',plotname)
		#---new method for getting all upstream calculations in the loop
		combodat = work.collect_upstream_calculations_over_loop(plotname)
		data,datas,calcs = combodat['data'],combodat['datas'],combodat['calcs']
		protein_abstractor_name = plotspecs.get('protein_abstractor_name','protein_abstractor')
		undulations_name = plotspecs.get('undulations_name','undulations')
		#---check for alternate loaders
		alt_loader = plotspecs.get('loader',None)
		if alt_loader:
			from base.tools import gopher
			postdat = gopher(alt_loader)(data=data)
		#---compile the necessary (default) data into a dictionary
		else:
			postdat = dict([(sn,dict(
				vecs=data[undulations_name][sn]['data']['vecs'].mean(axis=0),
				points_protein=data[protein_abstractor_name][sn]['data']['points_all']))
				for sn in work.sns()])
		#---end loading sequence
		#---export to globals
		global seepspace
		seepspace = 'data,datas,postdat,undulations_name,calcs,protein_abstractor_name'.split(',')
		for key in seepspace: globals()[key] = locals()[key]

@autoplot(plotrun)
def individual_reviews():
	"""
	Plot a few versions of the main figure.
	This is the standard plot. See other functions below for spinoffs.
	"""
	plotspec = {
		'coupling_review':{
			'viewnames':['average_height','average_height_pbc','neighborhood_static',
				'neighborhood_dynamic','average_field','example_field','example_field_pbc',
				'spectrum','spectrum_zoom']},}
	global seepspace
	seep = dict([(key,globals()[key]) for key in seepspace])
	for out_fn,details in plotspec.items(): 
		individual_reviews_plotter(out_fn=out_fn,seep=seep,**details)

@autoplot(plotrun)
def individual_reviews_ocean():
	"""
	Custom version of individual_reviews for the ocean project.
	Not included in the default plots.
	"""
	plotspec = {
		'coupling_review.simple':{
			'viewnames':['average_height','example_field_no_neighborhood'],'figsize':(6,6),'horizontal':True},
		'coupling_review.center_debug':{
			'viewnames':['neighborhood_static','neighborhood_dynamic',
				'average_height','average_height_pbc','average_field','example_field','example_field_pbc',
				'average_height_center','average_field_center','average_field_center_smear',
				'curvature_field_center'],
				'figsize':(16,16)},
		#---! this fails on the helix-0 tests because they have multiple proteins
		'coupling_review.simple_centered':{
			'viewnames':['average_height_center','curvature_field_center'],
			'figsize':(8,8),'horizontal':True,'wspace':0.7},}
	#---turn some off when developing
	for i in ['coupling_review.center_debug']: plotspec.pop(i)
	global seepspace
	seep = dict([(key,globals()[key]) for key in seepspace])
	for out_fn,details in plotspec.items(): 
		individual_reviews_plotter(out_fn=out_fn,seep=seep,**details)

@autoplot(plotrun)
def compare_curvature_estimates():
	"""Find the best curvature estimates for each method."""
	sns = work.sns()
	comp = dict([(sn,(datas.keys()[i],datas.values()[i][sn]['bundle'][sn]['fun'])) 
		for i in [np.argmin([datas[tag][sn]['bundle'][sn]['fun'] for tag in datas])] for sn in sns])
	print(comp)
