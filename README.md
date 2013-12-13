-------------------------------------------------------------------------------
Deferred Shader
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------
![fullscene](/renders/FullHallAllTextures.PNG "Finished Rendering")

Youtube Video of Rendering:
<dl>
<a href="http://www.youtube.com/watch?feature=player_embedded&v=kvQ3dNG4Mdg
" target="_blank"><img src="http://img.youtube.com/vi/kvQ3dNG4Mdg/0.jpg" 
alt="Youtube Video of Rendering Process" width="480" height="360" border="10" /></a>
</dl>

NOTE:
-------------------------------------------------------------------------------
This project requires a CUDA-capable graphics card, as well as OpenNI and an
Xbox Kinect to run a live demonstration.

-------------------------------------------------------------------------------
BACKGROUND:
-------------------------------------------------------------------------------
Previous work has demonstrated the diverse capabilities of RGBD cameras, from
generating highly accurate 3D surface models to reliable 3D pose estimation.
However, many algorithms attempt to store the generated environment as a RGB 3D
point cloud, which is not easily adaptable to dynamic environments, requires
very large quantities of memory to store large environments, and provides no
intuition to higher perception processes about distinct objects beyond a
volumetric approximation. Other approaches have been able to store and merge
the surface data more efficiently, but still regard the environement as a
unified whole rather than discrete objects. By extracting meaningful geometry
from the RGB-D in the form of triangle meshes instead, a large number of
advantages can be realized.

* High storage efficiency
* Natural low level object segmentation
* Easy to manipulate, modify, and render in real time
* Efficient and easy to process intuition of geometry that higher cognitive functions can use for object recognition and manipulation tasks.
* Straightforward tradeoff between simplicity and accuracy with mesh resolution

-------------------------------------------------------------------------------
CONTENTS:
-------------------------------------------------------------------------------
The Project6 root directory contains the following subdirectories:
	
* CUDA-Mesh/ is ...
* Recorder-Viewer/ is ...
* algorithm-python/ is ...
* shared/ contains glew

-------------------------------------------------------------------------------
CODE TOUR/CONTROLS
-------------------------------------------------------------------------------

Stage 1 samples model textures renders the scene geometry to the G-Buffer
* pass.vert
* pass.frag

Stage 2 renders the lighting passes and accumulates to the P-Buffer
* shade.vert
* ambient.frag
* point.frag
* diagnostic.frag

Stage 3 renders the post processing
* post.vert
* post.frag

Keyboard controls
[keyboard](https://github.com/cboots/Deferred-Shading/blob/master/base/src/main.cpp#L1178):
This is a good reference for the key mappings in the program. 
WASDQZ - Movement
X - Toggle scissor test
R - Reload shaders
1 - View depth
2 - View eye space normals
3 - View Diffuse color
4 - View eye space positions
5 - View lighting debug mode
6 - View Specular Mapping
7 - View Only Bloomed Geometry
0 - Standard view

x - Toggle Scissor Test
r - Reload Shaders
p - Print camera position to console
j - Toggle timing measurements to console (averaged since last reset)
SPACE - Reset timing averages

Shift-L - Toggle Bloom
Shift-T - Toggle Toon Shading
Shift-D - Toggle Diffuse Mapping
Shift-S - Toggle Specular Mapping
Shift-B - Toggle Bump Mapping
Shift-M - Toggle Transparency Masking

c - Reset  diffuse color to default
t - Change diffuse color to texture coordinate visualization
h - Overlay diffuse color with visualization of available textures
b - Change diffuse color to bump map visualization if available
m - Change diffuse color to white if mask texture is available

-------------------------------------------------------------------------------
Features:
-------------------------------------------------------------------------------

* Renders .obj files with support for .mtl files with
  * Diffuse Textures
  * Specular Textures
  * Height Maps (Bump Mapping)
  * Texture Masking
 
* Bloom (Not seperable, very inefficient)
![NoBloom](/renders/LampNoBloom.PNG "Without Bloom")
![Bloom](/renders/LampWithBloom.PNG "With Bloom")

* "Toon" Shading (with basic silhouetting and color quantization)
![Toon](/renders/ToonShadingNoColor.PNG "Toon Shading B/W")
![Toon](/renders/FullHallToon.PNG "Toon Shading")

* Point light sources with specular

![Point Specular](/renders/PointLightSpeculars.PNG "Point Light Speculars")
-------------------------------------------------------------------------------
SCREENSHOTS
-------------------------------------------------------------------------------

No special features enabled. Just Point lighting:
![Baseline](/renders/FullHallBaseline.PNG "Baseline")

Diffuse Texture:
![Diffuse Map](/renders/FullHallDiffuseOnly.PNG "Diffuse Texture")

Specular Texture:
![Specular Map](/renders/FullHallSpecularOnly.PNG "Specular Texture")

Bump Mapping:
![Bump Map](/renders/FullHallBumpOnly.PNG "Bump Texture")
![Bump Map](/renders/LionCloseNoBump.PNG "Without Bump")
![Bump Map](/renders/LionClose.PNG "With Bump")

Masking:
![Mask](/renders/FullHallMaskOnly.PNG "Mask Texture")
![No Mask](/renders/PlantsNoMask.PNG "No Mask")
![With Mask](/renders/PlantsWithMask.PNG "With Mask")


All Textures:
![All Textures Map](/renders/FullHallAllTextures.PNG "Specular Texture")

Scissor Test Specular Bug:
![ScissorNo](/renders/SpecularNoScissorTest.PNG "No Scissor Test")
![Scissor](/renders/SpecularWithScissorTest.PNG "With Scissor Test")

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
Global vs shared memory for point normals computation:

![KernelRuntime](/docs/performance/SharedVsGlobalRuntime.png "Kernel Runtime")
![FPS](/docs/performance/SharedVsGlobalFPS.png "FPS")

---
ACKNOWLEDGEMENTS
---
This project makes use of [tinyobjloader](http://syoyo.github.io/tinyobjloader/) and [SOIL](http://lonesock.net/soil.html)

REMEMBER TO ACKNOWLEDGE LIBRARIES

