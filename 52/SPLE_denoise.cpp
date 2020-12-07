/*
 * Copyright 2009-2012 Yi-Qing WANG 
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file SPLE_denoise.cpp
 * @brief command-line handler for S-PLE image denoising
 *
 * @author Yi-Qing WANG <yiqing.wang@polytechnique.edu>
 */

#ifndef STRING_H
#define STRING_H
#include <string>
#endif

#ifndef STD_H
#define STD_H
#include <iostream>
#include <iomanip>
#include <cstdio>
#endif

#ifndef SPLE_H
#define SPLE_H
#include "SPLE_lib.h" 		// routines dedicated to this implementation
#endif

using namespace std;


int main( int argc, char *const *argv ){


	if( !(argc == 5 || argc == 6) ){

		cerr << endl << endl << endl;
		cerr << "\t\t#######################################################################################################" << endl << endl;
		cerr << "\t\tBesides generating noise internally, this implementation can read in noise (add_gaussian_noise)." << endl << endl;
		cerr << "\t\tThe algorithm operates under two modes: the testing and demo mode controlled by Testing_Mode." << endl << endl;
		cerr << "\t\tThe slower testing mode produces more images to look at to help better understand the algorithm." << endl << endl;
		cerr << "\t\t#######################################################################################################" << endl << endl;
		cerr << "\t Usage : please specify the parameters for " << argv[0] << endl << endl; 
		cerr << "\t\t 1. input (clean) image name: please include the suffix .png" << endl;
		cerr << "\t\t 2. sigma: standard deviation of gaussian noise to apply to the clean image" << endl;
		cerr << "\t\t 3. iterations: an integer bigger than 1" << endl;
		cerr << "\t\t 4. output (denoised) image name: please include the suffix .png" << endl;
		cerr << "\t\t 5. operating mode: it defaults to the demo mode. Set it to enable the testing mode." << endl << endl;
		cerr << "\t For example: " << endl << endl;
		cerr << "\t\t ./denoiseSPLE clean.png 10 2 denoised.png" << endl << endl;
		cerr << "\t Or" << endl << endl;
		cerr << "\t\t ./denoiseSPLE clean.png 10 2 denoised.png 1" << endl << endl;

        	return EXIT_FAILURE;
	}		


	const char * input = argv[1];
	const char * output = argv[4];
	const char * expected_suffix = ".png";
//	input name as a string
	string si(input);
//	output name as a string
	string so(output);
//	input name's last 4 letters
	string suffix = si.substr(si.length()-4, 4);
	if( suffix.compare(expected_suffix) != 0 ){
		cerr << "ERROR: the implementation only supports PNG. Please include .png in the input name" << endl;
		return EXIT_FAILURE;
	}
	suffix = so.substr(so.length()-4, 4);	
	if( suffix.compare(expected_suffix) != 0 ){
        	cerr << "ERROR: the implementation only supports PNG. Please include .png in the output name" << endl;
                return EXIT_FAILURE;
        } 


	double sigma = atof(argv[2]);
	if( sigma <= 0 ){
		cerr << "ERROR: noise level must be strictly positive and not " << sigma << endl;
		return EXIT_FAILURE;
	}

	
	int n_iter = atoi(argv[3]);
	if( n_iter < 1 ){
		cerr << "ERROR: the algorithm has to iterate at least once" << endl;
		return EXIT_FAILURE;
	}


	bool Testing_Mode = argc == 6 ? true : false;	
			
	return SPLE_main_routine( 	/* input */ input,
					/* output */ output,
					/* polluting_sigma */ sigma,
					/* times */ n_iter,
					/* overlap */ 7, 
					/* patch_size */ 8,
					/* num_orientations */ 18,
					/* std_num_factors */ 32,
					/* only_Wiener */ false,
		       			/* both Wiener and Donoho*/ true,
					/* Testing Mode */ Testing_Mode );
}

