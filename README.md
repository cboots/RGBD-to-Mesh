-------------------------------------------------------------------------------
Surface Mesh Reconstruction from RGBD Images
-------------------------------------------------------------------------------
Final Project for Patrick Cozzi's CIS 565, Fall 2013
-------------------------------------------------------------------------------

Click for video
<dl>
<a href="http://www.youtube.com/watch?feature=player_embedded&v=pg0YZ76ZZw4
" target="_blank">
<img src="/docs/screenshots/AllStages.PNG" 
alt="Project Overview (YouTube)" width="640" height="480" border="10" />
</a>

</dl>
<a href="/docs/report/latex/report.pdf?raw=true">
Full report can be found here.
</a>
</dl>

NOTE:
-------------------------------------------------------------------------------
This project requires a CUDA-capable graphics card, as well as Boost, OpenNI and an
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
	
* CUDA-Mesh/ is the main project directory.
* Recorder-Viewer/ is a tool for viewing the sensor input and recording/playing back logs.
* algorithm-python/ is a prototype algorithm used to test the math for normal estimation.
* shared/ contains glew

-------------------------------------------------------------------------------
CODE TOUR
-------------------------------------------------------------------------------

The overall image processing line is shown below. First, an RGB frame and a
depth frame are pulled from the Kinect and pushed to the GPU for processing. A
world-space point cloud is then generated from the RGBD data, and a
neighborhood-based estimate of the point normals is then extracted for later
processing. Finally, the point cloud is triangulated and the generated mesh is
passed to OpenGL where a variety of rendering options are implemented.

![Image Processing Pipeline](/docs/diagrams/ImageProcessingPipeline.png "Image Processing Pipeline")

The underlying architecture is very modular, and can be easily extended to
handle input RGBD streams other than the Kinect (as demonstrated in the
implementation of log streams). A generic RGBD frame format is used, allowing
computation and visualization to be performed without regard to how the data
was obtained.

![Framework Layout](/docs/diagrams/FrameworkLayout.png "Framework Layout")

A more detailed view of the program flow is shown below. Note that after the
RGB and depth frames are synchronized and shipped to the GPU (the purple arrow), all computation
and rendering is performed on the GPU, enhancing performance and allowing the
CPU to be free for other tasks. The ComputeNormalsFast kernel supplants an
earlier iteration, ComputeNormals, which was written for estimation quality at
the cost of a significant performance penalty. 
The new implementation is much faster thanks to shared memory optimization (see performance section below).

![Program Flow](/docs/diagrams/ProgramFlow.png "Program Flow")

Finally, the following is a more detailed view of the OpenGL rendering
pipeline. The rendering pipeline is also written in a very modular manner,
allowing both for rapid code modification to experiment with different
visualazation techniques, as well as hooks (note the black diamonds) for
keypresses to completely change the render output on-the-fly.

![OpenGL Pipeline](/docs/diagrams/OpenGLPipeline.png "OpenGL Pipeline")

-------------------------------------------------------------------------------
CONTROLS
-------------------------------------------------------------------------------

Click and drag the mouse to look around in 3D views

Keypress | Function
--- | ---
w a s d q z| Move Camera in 3D views Forward/Left/Back/Right/Up/Down
W A S D Q Z| Move Camera slowly in 3D views
x | Reset camera view
r | Reload GLSL shaders
p | Restart playback at beginning of log (only for LogDevice input)
= | Increase playback speed (only for LogDevice input)
- | Decrease playback speed (only for LogDevice input)
F | Increase camera FOV
f | Decrease camera FOV
ESC | Exit
h | Toggle normal hair display in 3D point cloud mode
v | Toggle wireframe mode for 3D mesh
M | Increase maximum triangle edge length by 1cm
m | Increase maximum triangle edge length by 1mm
N | Decrease maximum triangle edge length by 1cm
n | Decrease maximum triangle edge length by 1mm
b | Toggle fast/slow normal computation algorithm
B | Toggle normal computation on/off
1 | Display depth overlaid on color input image
2 | Display depth only
3 | Display color only
4 | Display depth, color, and overlay on same screen
5 | Display point cloud buffer debug views
6 | Display 3D point cloud rendered from VBO
7 | Display 3D mesh reconstruction
8 | Display side by side comparison of color input image and 3D mesh reconstruction
9 | Display side by side comparison of 3D point cloud and 3D mesh reconstruction
0 | Display both input images, 3D point, and 3D mesh


-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------

Our point normals kernel was implemented as follows. A window radius is first
specified as an algorithm parameter. For each point, we loop through its
neighboring points in screen space in the square window specified by the
radius, and pair it with a screen-space orthogonal point at the same radius. If
both points are within a specified radius from the center point in world space,
we take the cross product to compute the normal, which is then flipped if
pointing away from the camera. If sufficiently many valid normals are found, we
average them to produce the final normal estimate, otherwise we discard the
point.

To improve the runtime of the point normals kernel, we reimplemented the
algorithm using shared memory. In the shared memory implementation, all points
in given thread block are first loaded into shared memory, along with the
points lying within the specified neighborhood radius of the edges of the
thread block, and the distance and cross product calculations are then
performed using shared memory access. The results of the shared memory
optimization on kernel runtime are shown for a range of window radii using a
thread block size of 8x8.

![KernelRuntime](/docs/performance/SharedVsGlobalRuntime.png "Kernel Runtime")
![FPS](/docs/performance/SharedVsGlobalFPS.png "FPS")

As demonstrated, the shared memory optimization reduced the kernel runtime by
approximately a factor of 2. The impact on the overall FPS was less dramatic,
though still pronounced, due to the time spent in the rendering pipeline.

All testing for this project was conducted on the following hardware:
* CPU: Intel Core i5-2450M, 2.5GHz 6GB (Windows 8, 64-bit OS)
* GPU: NVIDIA GeForce GT 525M

-------------------------------------------------------------------------------
SCREENSHOTS
-------------------------------------------------------------------------------

View of 3D point cloud with normals visualized as color data.
![3DPointCloudNormalsAbove.PNG](/docs/screenshots/3DPointCloudNormalsAbove.PNG "3DPointCloudNormalsAbove.PNG")

Colorfully shaded depth data overlaid on color data using GLSL shaders.
![ColorfulOverlay.PNG](/docs/screenshots/ColorfulOverlay.PNG "ColorfulOverlay.PNG")

Comparision of valid point cloud data, normals, world space position, and depth buffer.
![ImprovedNormals.PNG](/docs/screenshots/ImprovedNormals.PNG "ImprovedNormals.PNG")

First successful attempt to render a reconstructed mesh. Normals as color again.
![MeshNormals.PNG](/docs/screenshots/MeshNormals.PNG "MeshNormals.PNG")

Point cloud normals.
![3DPointCloudNormals.PNG](/docs/screenshots/3DPointCloudNormals.PNG "3DPointCloudNormals.PNG")

Side by side depth, color and overlay.
![DepthDataGUI.PNG](/docs/screenshots/DepthDataGUI.PNG "DepthDataGUI.PNG")

Mesh reconstruction, wireframe mode.
![MeshHead.PNG](/docs/screenshots/MeshHead.PNG "MeshHead.PNG")
![Mesh.PNG](/docs/screenshots/Mesh.PNG "Mesh.PNG")

Same face, but full rendering of mesh.
![FaceFilled.PNG](/docs/screenshots/FaceFilled.PNG "FaceFilled.PNG")
![MeshFile.PNG](/docs/screenshots/MeshFile.PNG "MeshFile.PNG")

Same visualization as above, just a different view.
![Window.PNG](/docs/screenshots/Window.PNG "Window.PNG")

Early attempt at computing normals. Was slow and far too picky.
![ChairPointCloudNormals.PNG](/docs/screenshots/ChairPointCloudNormals.PNG "ChairPointCloudNormals.PNG")

Python implementation of normal estimation algorithm. 
![python-normals.png](/docs/screenshots/python-normals.png "python-normals.png")

Normals visualized as lines.
![ColorHairs.PNG](/docs/screenshots/ColorHairs.PNG "ColorHairs.PNG")

Normals visualizes as colored lines
![Hairs.PNG](/docs/screenshots/Hairs.PNG "Hairs.PNG")

Multiple views of the same data.
![AllStages.PNG](/docs/screenshots/AllStages.PNG "AllStages.PNG")

Another mesh rendering.
![MeshFace.PNG](/docs/screenshots/MeshFace.PNG "MeshFace.PNG")

Same view but as a point cloud.
![Points.PNG](/docs/screenshots/Points.PNG "Points.PNG")


-------------------------------------------------------------------------------
ACKNOWLEDGEMENTS
-------------------------------------------------------------------------------

This project makes use of the following 3rd party libraries
* Boost: http://www.boost.org/
* CUB: http://nvlabs.github.io/cub/
* Thrust: https://code.google.com/p/thrust/
* LZ4: https://code.google.com/p/lz4/
* RapidXML: http://rapidxml.sourceforge.net/

