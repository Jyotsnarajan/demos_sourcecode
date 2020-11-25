--------------------------------------------------------------------------------
This directory contains the source code associated to the following
IPOL article:
Title: Parameter-Free Fast Pixelwise Non-Local Means Denoising
Authors: Jacques Froment
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
About:

* Author       : Jacques Froment <Jacques.Froment@univ-ubs.fr>
* Version      : 1.2
* Release Date : October 27, 2014
* Licence      : GPL v3+ (see GPLv3.txt) but BSD (see BSD.txt) for src/nlmp.c

* File src/nlmp.c implements algorithms possibly linked to the patents

A. Buades, T. Coll and J.M. Morel, Image data processing method by
reducing image noise, and camera integrating means for implementing
said method, EP Patent 1,749,278 (Feb. 7, 2007).

This source code is made available for the exclusive aim of serving 
as scientific tool to verify the soundness and completeness of the
algorithm description. Compilation, execution and redistribution
of this source code may violate patents rights in certain countries.
The situation being different for every country and changing
over time, it is your responsibility to determine which patent
rights restrictions apply to you before you compile, use,
modify, or redistribute this source code. A patent lawyer is qualified
to make this determination.
If and only if they don't conflict with any patent terms, you can 
benefit from the following license terms attached to this source code.

--------------------------------------------------------------------------------
Overview:

This source code provides a fast and parameter-free implementation of the 
pixelwise Non-Local Means image denoising method (NLMP for short).

- Compilation. 
The compilation has been tested on Unix/Linux only.
Automated compilation requires the make program.

- Library. 
This code requires the libpng library. 
To enable parallel processing the OpenMP library is needed.

- Image format. 
Only the PNG format is supported. 
 
--------------------------------------------------------------------------------
Usage:

1. Download the code package and extract it. Go to that directory. 

2. Compile the source code (on Unix/Linux/Mac OS). 
There are two ways to compile the code. 
(1) RECOMMENDED, with Open Multi-Processing multithread parallelization 
(http://openmp.org/). Roughly speaking, it accelerates the program using the 
multiple processors in the computer. Run
make opti OMP=1

OR
(2) If the complier does not support OpenMp, run 
make opti

(compiler optimization may be disable by removing the word "opti").

3. Run the Fast Pixelwise Non-Local Means code by
./NLMeansP

and get the PSNR between original and denoised image by
./img_mse_ipol

4. Example (you should provide a PNG noise-free image named "U.png"):

make opti OMP=1
./NLMeansP U.png 20 V.png Vd.png
./img_mse_ipol U.PNG Vd.png

--------------------------------------------------------------------------------
List of files ([R] means Reviewed file):

BSD.txt         : BSD license for src/nlmp.c
GPLv3.txt       : GPL v3+ license for other files
Makefile        : compilation instructions for the make command
README.txt      : this README file

In the src subdirectory:

addgaussnoise.c : Add a gaussian noise to an image using Mersenne Twister 
                  pseudo-RNG coden (main file)
img_mse_ipol.c  : Compute the RMSE and the PSNR of two images in PNG format
img_diff_ipol.cpp : Compute the difference of two images in PNG format
                  (same code as in Antoni Buades, Bartomeu Coll, and Jean-Michel 
                   Morel, "Non-Local Means Denoising", IPOL 2011).
io_png.c        : PNG read/write simplified interface
io_png.h        : header (include file) for io_png.c
libauxiliary.c  : Contains the addgaussnoise() function
libauxiliary.h  : header (include file) for libauxiliary.
mt19937ar.c     : Mersenne Twister pseudo-RNG code
mt19937ar.h     : header (include file) for mt19937ar.c
NLMeansP.c [R]  : pixelwise NLMP (basic and SIL algorithms), main file
nlmp.c [R]      : pixelwise NLMP (basic and SIL algorithms), nlmp() function

--------------------------------------------------------------------------------
History of changes:

V1.2 - October 27, 2014: added img_diff_ipol.cpp to display image difference in 
                         the demo.
V1.1 - September 8, 2014: Second submitted version to reflect manuscript V. 1.02.
 changes regarding V. 1.0:
 - Minor change in NLMeansP.c, function SetParam_a (no change for integer values
   of sigma);
 - The exp() function called to compute NLM-weights is now tabulated using a LUT
   (for faster access);
 - Add -c (clock) option to print the CPU time of the denoising process only 
   (excluding I/O).
V1.0 - July 18, 2014 : First submitted version

--------------------------------------------------------------------------------
