REM This script builds the conda environment required for running DielectricDecomposition
REM
REM ---------------------- REM
REM Initialize Environment REM
REM ---------------------- REM
CALL conda remove --name=MMSRDD --all
CALL conda create --name=MMSRDD python==3.7.3
CALL conda activate MMSRDD
REM
REM -------------- REM
REM Setup Packages REM
REM -------------- REM
REM
CALL pip install numpy
CALL pip install scipy
CALL pip install matplotlib
CALL pip install scikit-image
CALL pip install tqdm
