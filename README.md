# MASER
# Program: BMP_simulation_04112022.py
## Make two folder, codes and data. Save this program in codes and run it. Output will be in folder data. <br />

Simulation: Maxwell - Bloch - Provotorov, Dipole Dipole Interaction and Bo Inhomogenety <br />
Author: Vineeth Thalakottoor (vineethfrancis.physics@gmail.com) <br />
Created: 12 November 2021 <br />
Modified: 4 November 2022 <br />
https://github.com/VThalakottoor/MASER.git <br />

A) This program can solve Maxwell - Bloch - Provotorov, along with dipole-dipole interaction and Bo inhomogenety. <br />
B) This is a beta program, its not that user friendly. <br />
C) Go to line 579 for setting the parameters <br />

D) Output files:

1) Mxi.txt save (t, Transverse magnetization, Longitudinal magnetization) in defined instant of time (line 72) <br />

2) Bdip.txt save dipolar field and magnetization at defiend time # caution consume a lot of space <br />
row 7*i + 0 : time, t <br />
row 7*i + 1 : Bdipole_X (t) <br />
row 7*i + 2 : Bdipole_Y (t) <br />
row 7*i + 3 : Bdipole_X (t) <br />
row 7*i + 4 : M_X (t) <br />
row 7*i + 5 : M_Y (t) <br />
row 7*i + 6 : M_Z (t) <br />

use the the file **splitBdip.sh** to split each into different files <br />

3) dataMx.txt save average Mx  <br />
   dataMy.txt save average My <br />
   dataMz.txt save average Mz <br />
   datat.txt save time <br />
   
4) dataMxall.txt save individual Mx <br />
   dataMyall.txt save individual My <br />
   dataMzall.txt save individual Mz <br />
   
E) When simulation complete it will plot average transverse and longitudinal magnetization    <br />

F) If you want to polt data live, plot with BMP_simulation_liveplotter.ipynb, save this file in folder data
For any question write to me <br />
