#!/usr/bin/env python

"""
...
"""

import scipy
import scipy.stats
import scipy.interpolate
import itertools,time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import brewer2mpl

#---settings
#---we must declare variables here. this should match the top of the reload function
variables = 'sns,collect'.split(',')

from codes.looptools import basic_compute_loop

###---STANDARD

#---declare standard variables
required_variables = 'printers,routine'.split(',')
for v in variables+required_variables:
	if v not in globals(): globals()[v] = None

def register_printer(func):
	"""Add decorated functions to a list of "printers" which are the default routine."""
	#---! note that you cannot add the decorator without restarting
	global printers
	if printers is None: printers = []
	printers.append(func.__name__)
	return func

###---PLOTS

def get_upstream_mesh_partners(abstractor_key):
	"""Filter the upstream sweep for presence/absence of cholesterol."""
	data = dict()
	for sn in sns:
		keys = [k for k in collect['calcs'] if 
			#---! extremely verbose!
			collect['calcs'][k][sn]['calcs']['specs']['upstream'][
				'lipid_mesh']['upstream']['lipid_abstractor']['selector']==abstractor_key]
		if len(keys)!=1: raise Exception('failed to find a unique mesh for this request')
		data[sn] = collect['datas'][keys[0]][sn]
	return data

def prep_postdat(nn,sns,abstractor='lipid_com'):
	data = get_upstream_mesh_partners(abstractor)
	postdat = dict()
	for sn in sns:
		top_mono = work.meta[sn].get('index_top_monolayer',0)
		combonames = data[sn]['combonames_%d'%nn]
		#---! deprecated method before using more basic method
		if False:
			counts_random = data[sn]['counts_random_%d'%nn].mean(axis=0)[top_mono].mean(axis=0)
			counts_obs = data[sn]['counts_observed_%d'%nn][top_mono].mean(axis=0)
			counts_random_err = data[sn]['counts_random_%d'%nn].std(axis=0)[top_mono].mean(axis=0)
			inds = np.where(counts_random>0)[0]
			ratios = counts_obs[inds]/counts_random[inds]
			ratios_err = np.array([data[sn]['counts_random_%d'%nn][i][top_mono].mean(axis=0) 
				for i in range(len(data[sn]['counts_random_%d'%nn]))]).std(axis=0)
			rename_ptdins = lambda x: 'PtdIns' if x==work.meta[sn]['ptdins_resname'] else x
			combonames_std = np.array([[rename_ptdins(i) for i in j] for j in combonames[inds]])
			postdat[sn] = dict(ratios=ratios,ratios_err=ratios_err,combos=combonames_std)
		combi = scipy.misc.comb
		combos = data[sn]['combonames_%d'%nn]
		top_mono = work.meta[sn].get('index_top_monolayer',0)
		reslist = data[sn]['reslist']
		comps = dict([(reslist[int(i)],j) 
			for i,j in zip(*np.unique(data[sn]['monolayer_residues_%d'%top_mono],return_counts=True))])
		inds = np.where([np.in1d(c,comps.keys()).all() for c in combos])[0]
		ratios_random = np.array([np.product([combi(comps[k],v) 
			for k,v in zip(*np.unique(c,return_counts=True))])/combi(sum(comps.values()),nn) 
			for c in combos[inds]])
		counts_obs = data[sn]['counts_observed_%d'%nn][top_mono].mean(axis=0)
		counts_obs_std = data[sn]['counts_observed_%d'%nn][top_mono].std(axis=0)
		nmols = data[sn]['monolayer_residues_%d'%top_mono].shape[0]
		ratios = counts_obs[inds]/(counts_obs[inds].sum())/ratios_random
		ratios_err = counts_obs_std[inds]/(counts_obs[inds].sum())/ratios_random
		rename_ptdins = lambda x: 'PtdIns' if x==work.meta[sn]['ptdins_resname'] else x
		combonames_std = np.array([[rename_ptdins(i) for i in j] for j in combonames[inds]])
		postdat[sn] = dict(ratios=ratios,ratios_err=ratios_err,combos=combonames_std)
	if not all([np.all(postdat[sns[0]]['combos']==postdat[sn]['combos']) for sn in sns]):
		for sn in sns:
			print(sn)
			print(postdat[sn]['combos'])
		print([sn for sn in sns if not np.all(postdat[sns[0]]['combos']==postdat[sn]['combos'])])
		raise Exception('uneven combination names')
	postdat['combos'] = postdat[sns[0]]['combos']
	return postdat

@register_printer
def plot_partners():
	"""Create several plots of the partner data."""
	#---alternate layouts
	specs = {
		'summary':{
			'sns':work.specs['collections']['position'],
			'extras':{'special':True,'summary':True,'legend_below':True},
			'specs':{
				0:dict(nn=3,abstractor='lipid_com',combos=[['Ptdins','Ptdins','Ptdins']]),
				1:dict(nn=2,abstractor='lipid_com'),},
			'panelspec':dict(figsize=(12,8),
				layout={'out':{'grid':[1,1]},'ins':[{'grid':[1,2],
				'wratios':[1,6],'wspace':0.1},]}),},
		'comprehensive_core':{
			'sns':work.specs['collections']['position'],
			'extras':{'small_labels_ax3':True,'all_y_labels':True,'legend_everywhere':True},
			'specs':{
				0:dict(nn=2,abstractor='lipid_com'),
				1:dict(nn=3,abstractor='lipid_com'),
				2:dict(nn=2,abstractor='lipid_chol_com'),
				3:dict(nn=3,abstractor='lipid_chol_com'),},
			'panelspec':dict(figsize=(18,30),
				layout={'out':{'grid':[1,1]},
				'ins':[{'grid':[4,1],'wspace':0.1} for i in range(1)]}),},
		'comprehensive_core_wide':{
			'sns':work.specs['collections']['position'],
			'extras':{'four_plots':True},
			'specs':{
				0:dict(nn=2,abstractor='lipid_com'),
				2:dict(nn=3,abstractor='lipid_com'),
				1:dict(nn=2,abstractor='lipid_chol_com'),
				3:dict(nn=3,abstractor='lipid_chol_com'),},
			'panelspec':dict(figsize=(36,18),
				layout={'out':{'grid':[1,2],'wratios':[1,2],'wspace':0.1},
				'ins':[{'grid':[2,1]} for i in range(2)]}),},
		'comprehensive':{
			#---! what's wrong with v536 has only POPC-POPC
			'sns':[i for i in work.specs['collections']['asymmetric_all'] if i!='membrane-v536'],
			'extras':{'small_labels_ax3':True,'error_bars':False,
				'all_y_labels':True,'legend_everywhere':True},
			'specs':{
				0:dict(nn=2,abstractor='lipid_com'),
				2:dict(nn=3,abstractor='lipid_com'),
				1:dict(nn=2,abstractor='lipid_chol_com'),
				3:dict(nn=3,abstractor='lipid_chol_com'),},
			'panelspec':dict(figsize=(18,30),
				layout={'out':{'grid':[1,1]},
				'ins':[{'grid':[4,1],'wspace':0.1} for i in range(1)]}),},}
	for figname,spec in specs.items(): 
		plot_partners_basic(figname=figname,**spec)

def plot_partners_basic(sns,figname,specs,panelspec,extras):
	"""Summarize the lipid mesh partners."""
	baseline = 1.0
	sns_reorder = sns
	plotspec = ptdins_manuscript_settings()
	axes,fig = panelplot(**panelspec)
	try: axes = [i for j in axes for i in j]
	except: pass
	for axnum,details in specs.items():
		nn = details['nn']
		postdat = prep_postdat(nn=nn,sns=sns,abstractor=details['abstractor'])
		combos = details.get('combos',postdat['combos'])
		ax = axes[axnum]
		max_y,min_y = 0,10
		combo_spacer,half_width = 0.5,0.5
		for snum,sn in enumerate(sns_reorder):
			#---already ensured each simulation has the same combinations
			for cnum,combo in enumerate(combos):
				xpos = (cnum)*len(sns)+snum+cnum*combo_spacer
				ypos = postdat[sn]['ratios'][cnum]-baseline
				ax.bar([xpos],[ypos],
					width=2*half_width,bottom=baseline,
					color=plotspec['colors'][plotspec['colors_ions'][work.meta[sn]['cation']]],
					hatch=plotspec['hatches_lipids'][work.meta[sn]['ptdins_resname']],**(
						{'label':work.meta[sn]['ion_label']} if cnum==len(combos)-1 else {}))
				y_err = postdat[sn]['ratios_err'][cnum]
				if extras.get('error_bars',True): ax.errorbar(xpos,ypos+baseline,yerr=y_err,
					alpha=1.0,lw=2.0,c='k')
				max_y,min_y = max((max_y,ypos+baseline+y_err)),min((min_y,ypos+baseline-y_err))
		ax.set_xlim(-half_width-half_width/2.,(len(sns))*len(combos)-
			half_width+(len(combos)-1)*combo_spacer+half_width/2.)
		ax.axhline(1.0,c='k',lw=2)
		dividers = np.array([len(sns)*(cnum+1)+cnum*half_width-half_width/2.
			for cnum in range(-1,len(combos))])
		for divider in dividers: ax.axvline(divider,c='k',lw=1)
		mids = (dividers[1:]+dividers[:-1])/2.
		ax.set_xticks(mids)
		ax.set_xticklabels(['\n'.join(c) for c in combos],fontsize=18)
		ax.xaxis.tick_top()
		labelsize = 14
		if extras.get('small_labels_ax3',False) and axnum==3: labelsize = 10
		ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='on',labelsize=14)
		ax.tick_params(axis='x',which='both',top='off',bottom='off',labeltop='on',labelsize=labelsize)
		#---! ax.set_ylim((min_y*0.9,max_y*1.1))
		if axnum==len(axes)-1 or extras.get('legend_everywhere',False):
			bar_formats = make_bar_formats(sns_reorder,work=work)
			comparison_spec = dict(ptdins_names=list(set([work.meta[sn]['ptdins_resname'] for sn in sns])),
				ion_names=list(set([work.meta[sn]['cation'] for sn in sns])))
			kwargs = dict(bbox=(1.05,0.0,1.,1.),loc='upper left')
			if extras.get('legend_below',False):
				kwargs = dict(bbox=(0.5,-0.05),loc='upper center',ncol=4)
			legend,patches = legend_maker_stylized(ax,work=work,
				sns_this=sns_reorder,bar_formats=bar_formats,comparison_spec=comparison_spec,
				**kwargs)
	#---make the unity line even between subplots (used algrebra in real life here)
	#---...decided to shift down the right axis y plot since it has a smaller range, to make things even
	#---! note that this is not perfect but we should just pick our battles
	if extras.get('special',False):
		spread_prop = np.array([[i for i in ax.get_ylim()] for ax in axes[:2]])
		s0,s1,s2,s3 = spread_prop.reshape(-1)
		shift_down = ((1.-s2)-(1.-s0)*(s3-1.)/(s1-1.))
		axes[1].set_ylim((axes[1].get_ylim()[0],axes[1].get_ylim()[1]+shift_down))
	axes[0].set_ylabel('observations relative to chance',fontsize=16)
	if extras.get('four_plots',False):
		axes[1].set_ylabel('observations relative to chance',fontsize=16)
	if extras.get('all_y_labels',False):
		for ax in axes: ax.set_ylabel('observations relative to chance',fontsize=16)
	picturesave('fig.lipid_mesh_partners.%s'%figname,work.plotdir,backup=False,version=True,meta={},extras=[legend],
		form='pdf')

def reload():
	"""Load everything for the plot only once."""
	#---canonical globals list is loaded systematically 
	#---...but you have to load it into globals manually below
	global variables
	for v in variables: exec('global %s'%v)
	#---custom reload sequence goes here
	#---instead of plotload we do a sweep
	globals()['collect'] = work.collect_upstream_calculations_over_loop('lipid_mesh_partners')
	globals()['sns'] = work.sns()

###---STANDARD

def printer():
	"""Load once per plot session."""
	global variables,routine,printers
	#---reload if not all of the globals in the variables
	if any([v not in globals() or globals()[v] is None for v in variables]): reload()
	#---after loading we run the printers
	printers = list(set(printers if printers else []))
	if routine is None: routine = list(printers)	
	#---routine items are function names
	for key in routine: 
		status('running routine %s'%key,tag='printer')
		globals()[key]()

if __name__=='__main__': printer()
