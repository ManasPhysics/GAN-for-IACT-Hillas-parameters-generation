# GAN-for-IACT-Hillas-parameters-generation

The Python file "Hillas_GAN.py" trains a GAN.
The datafile "HillasGamma_Zen10_PE18.dat" is Hillas parameter file for Gamma
The datafile "HillasProton_Zen10_PE18.dat" is Hillas parameter file for Proton

#### How to evaluate training performance ######

Run only the portion of code(line no 192-204)

#### How to change datafile(proton or gamma datafile)
Change the filename in line number 27

#### How to see generated parameters distribution#######
Run the Hillas_GAN.py code first 
then Run Histogram_Real_and_Generated_parameters.py 
