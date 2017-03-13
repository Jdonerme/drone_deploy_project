# drone_deploy_project

Project to find the pattern QR code in a series of images taken by an iphone 6.
Uses the QR orientation to estimate camera location and position.

To use, run the find_camera_locations script. 
(i.e. python find_camera_locations.py)
In the visualization displayed, the dot represents the lens of the camera
and the arrow represents where the camera was pointed. The grid represents where
the QR code was. The QR code should always be situated such that the corner 
without the square in it is at the bottom left of the image (and thus near the
coordinate -4, -4).