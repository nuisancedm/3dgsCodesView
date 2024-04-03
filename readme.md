it is a copy of orginal 3d gaussian splatting source code
used for learning

# Parameter Track
## gaussian created:
* _feature_dc [100000,1,3]
* _feature_rest [100000,15,3]
* _opacity [100000,1]
* _rotation [100000,4]
* _scaling [100000,3]
* _xyz [100000,3]

## send to rasterizer
* means3D = pc.get_xyz [100000,3]
* means2D = screenspace_points [100000,3]
* shs = pc.get_features [100000,16,3]
* opacities = pc.get_opacity [100000,1]
* scales = pc.get_scaling [100000,3]
* rotations = pc.get_rotation [100000,4]
