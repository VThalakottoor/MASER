# MASER


Simulation: Maxwell - Bloch - Provotorov, Dipole Dipole Interaction and Bo Inhomogenety
Author: Vineeth Thalakottoor (vineethfrancis.physics@gmail.com)
Created: 12 November 2021
Modified: 4 November 2022
https://github.com/VThalakottoor/MASER.git

This program can solve Maxwell - Bloch - Provotorov, along with dipole-dipole interaction and Bo inkomogenety. 
This is a beta program, its not that user friendly.
Go to line 579 for setting the parameters
output files:
1) Mxi.txt save (t, np.sqrt((np.sum(M[0:nx*ny*nz*3:3]))**2 + (np.sum(M[1:nx*ny*nz*3:3]))**2), np.sum(M[2:nx*ny*nz*3:3])) in defined instant of time (line 72)
2) Bdip.txt save dipolar field and magnetization at defiend time # caution consume a lot of space
row 7*i + 0 : time, t
row 7*i + 1 : Bdipole_X (t)
row 7*i + 2 : Bdipole_Y (t)
row 7*i + 3 : Bdipole_X (t)
row 7*i + 4 : M_X (t)
row 7*i + 5 : M_Y (t)
row 7*i + 6 : M_Z (t)
use the the file splitBdip to split each into different files
3) dataMx.txt save average Mx 
   dataMy.txt save average My
   dataMz.txt save average Mz
   datat.txt save time
4) dataMxall.txt save individual Mx
   dataMyall.txt save individual My
   dataMzall.txt save individual Mz
5) When simulation complete it will plot average transverse and longitudinal magnetization   

For any question write to me

make two folder codes and data. save this program in codes and run it. Output will be in folder data.
