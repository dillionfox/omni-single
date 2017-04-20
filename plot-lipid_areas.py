#!/usr/bin/env python

"""
PLOT MESH OBJECTS
plots that derive from the Delaunay mesh objects generated by codes/mesh.py
"""

routine = ['review3d','cellplot','curvature'][-1:]

#---load everything
if 'data' not in globals(): 
	data,calcs = plotload(plotname,work)
	data_prot,calcs_prot = plotload('protein_abstractor',work)

if 'review3d' in routine:

	import mayavi
	from mayavi import mlab
	from codes.review3d import review3d,pbcbox

	sn = 'mk004'
	dat = data[sn]['data']
	mn,fr = 1,10
	pts = dat['%d.%d.%s'%(mn,fr,'points')]
	vec = dat['%d.%d.%s'%(mn,fr,'vec')]
	mean_curvs = dat['%d.%d.%s'%(mn,fr,'mean')]
	mean_curvs_normed = mean_curvs/np.abs(mean_curvs).max()+0.5
	review3d(points=[pts],tube=0.02,radius=0.2,noshow=True,colorset=mean_curvs_normed,cmap='seismic')
	review3d(points=data_prot[sn]['data']['points_all'][fr],radius=0.4,noshow=True)
	pbcbox(vec)
	mlab.show()
	
if 'cellplot' in routine:

	from codes.cellplot import cellplot

	sn = 'mk002'
	dat = data[sn]['data']
	mn,fr = 1,10
	simps = dat['%d.%d.simplices'%(mn,fr)]
	pts = dat['%d.%d.points'%(mn,fr)]

if 'curvature' in routine:

	"""
	???
	"""

	#---settings
	spacing = 0.5
	mn = 0

	sn = 'mk004'
	dat = data[sn]['data']
	nframes = int(dat['nframes'])
	nmol = int(dat['%d.1.nmol'%mn])
	def get(mn,fr,name): return dat['%d.%d.%s'%(mn,fr,name)]

	mvecs = np.mean([get(0,fr,'vec') for fr in range(nframes)],axis=0)
	ngrid = np.round(mvecs/spacing)

	#---curvature of each leaflet
	curvs_map = [np.zeros((ngrid[0],ngrid[1])) for mn in range(2)]
	for mn in range(2):
		curvs = np.zeros((ngrid[0],ngrid[1]))
		curvs_counts = np.zeros((ngrid[0],ngrid[1]))
		nmol = int(dat['%d.1.nmol'%mn])
		for fr in range(nframes):
			simps = get(mn,fr,'simplices')
			pts = get(mn,fr,'points')
			vec = get(mn,fr,'vec')
			curv = get(mn,fr,'mean')
			#---repack the points in the box
			pts_repack = (pts-np.floor(pts/vec)*vec)
			pts_rounded = (pts_repack/(vec/ngrid)).astype(int)[:,:2]
			curvs[pts_rounded[:nmol,0],pts_rounded[:nmol,1]] += curv[:nmol]
			curvs_counts[pts_rounded[:nmol,0],pts_rounded[:nmol,1]] += 1
		obs = np.where(curvs_counts>0)
		means = curvs[obs]/curvs_counts[obs]
		curvs_map[mn][obs] = means

	from mpl_toolkits.axes_grid1.inset_locator import inset_axes
	import scipy
	import scipy.ndimage

	#---average the leaflets
	curvs_map = np.mean(curvs_map,axis=0)
	#---ironically back to nan for the imshow
	curvs_map[np.where(curvs_counts==0)] = np.nan
	#---! these maps are noisy so we smooth them --- no theory here though ...
	curvs_map = scipy.ndimage.gaussian_filter(curvs_map,sigma=(4,4),order=0)
	vmax = max([abs(j) for j in [curvs_map.max(),curvs_map.min()]])
	ax = plt.gca()
	im = ax.imshow(curvs_map,cmap=mpl.cm.RdBu,vmax=vmax,vmin=-1*vmax,origin='lower',
		extent=[0,mvecs[0],0,mvecs[1]])
	axins = inset_axes(ax,width="5%",height="100%",loc=3,
		bbox_to_anchor=(1.05,0.,1.,1.),bbox_transform=ax.transAxes,borderpad=0)
	cbar = plt.colorbar(im,cax=axins,orientation="vertical")
	cbar.set_label(r'$\mathrm{C_0\,({nm}^{-1})$',rotation=270)
	ax.set_xlabel('x (nm)')
	ax.set_ylabel('y (nm)')
	ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='on')
	ax.tick_params(axis='x',which='both',top='off',bottom='off',labelbottom='on')
	axins.tick_params(axis='y',which='both',left='off',right='off',labelright='on')
	ax.set_title('curvature map')
	prot_pts = data_prot[sn]['data']['points_all'].mean(axis=0)
	#---only one protein
	prot_pts = data_prot[sn]['data']['points_all'].mean(axis=0)[0]
	ax.scatter(prot_pts[:,0],prot_pts[:,1])
	picturesave('fig.%s.%s'%('curvature','demo'),work.plotdir,backup=False,version=True,meta=meta)
	plt.close()