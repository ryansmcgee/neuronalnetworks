PlaneSurface(x=, y=)
CylinderSurface(r=, h=) -> CylinderSurface(x=, y=)
TorusSurface(r_minor=, r_major=) -> TorusSurface(x=, y=)
SpheroidSurface(r_xy=, r_z=)
CubeVolume(x=, y=, z=)
CylinderVolume(r=, h=)
AnnulusVolume(w=, h=, r=)
TorusVolume(r_minor=, r_major=)
EllipsoidVolume(r_x=, r_y=, r_z=)


					Geometry Constructor		Neuron Positions
PlaneSurface		(x,y), (y,z), (x,z)			(x,y), (y,z), (x,z)		cartesian
CylinderSurface		(r,h), (w,h)				(theta,h), (w,h)		cartesian, parametric, surfacePlane
TorusSurface		(r_minor, r_major), (w,h)	(theta, phi), (w,h)		cartesian, parametric, surfacePlane
SpheroidSurface		(r_xy, r_z)					(theta, phi)			cartesian, parametric
CubeVolume			(x, y, z)					(x,y,z)					cartesian,
CylinderVolume		(r, h)						(theta,r,h), (x,y,z)	cartesian, parametric
AnnulusVolume		(r, w, h)					(r,w,h)					cartesian, parametric
TorusSurface		(r_minor, r_major)

(for all geos: cartesian coords include taking origin shift into account)
(for all geos: parametric coords are relative to the geo dimensions alone (no origin consideration)
	- (this holds for x,y and x,y,z geos: parametric x,y,z coords are relative to the non-origin-shifted geometry)





network.geometry.position_neurons(self, positioning='random', bounds_theta=(0,10), bounds_h=(0,10), bounds_x=(0,10), bounds_y=(0,10), bounds_z=(0,10))
network.geometry.position_neurons(self, positioning='random', bounds={'theta':(0,0), 'h':(0,0), 'x':(0,0), 'y':(0,0), 'z':(0,0)})

network.geometry.position_neurons(self, positioning='random', bounds_theta=(0,10), bounds_h=(0,10))
network.geometry.position_neurons(self, positioning='random', bounds={'theta':(0,0), 'h':(0,0)})

network.geometry.position_neurons(self, positioning='random', bounds_x=(0,10), bounds_y=(0,10), bounds_z=(0,10))
network.geometry.position_neurons(self, positioning='random', bounds={'x':(0,0), 'y':(0,0), 'z':(0,0)})

network.geometry.position_neurons(self, positioning='random', bounds_h=(2,8))
network.geometry.position_neurons(self, positioning='random', bounds={'h':(2,8)})
