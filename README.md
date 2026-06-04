# ROB 421 CTF

2 Vex Robots atempt to grab all their barrel's from the others side before the other.





##### To Install Updated Particle Filter:

The updated particle filter is the file particle.py. Copy this file, and go into your aim-fsm folder. Then paste the file, making sure to replace the particle.py file that is already present



If you want to, make a backup of the particle.py file before you replace it



This new particle file lets the particle filter add variance to the particles as it moves, letting it correct itself if it got a bad sensor reading early in the data collection.



To use the particle filter: Make sure to look at localization.fsm, which contains all the information to setup the particle filter and the map provided from the step files and aruco marker PDF. You can also boot up the particle filter to see the location of the Aruco Markers when building the map



##### Building the Map:

The map was created using a 3D printer with ABS filament, but as far as I know you could use any filament for this printing. You will also need to print out the Aruco Marker PDF pages, and place one of each marker from 1-8 within the depression on each wall piece. When building the map, start with 1 and go in a circle clockwise increasing the aruco marker's value. Make sure that the 1 marker is on the left side of the squares perimeter.





