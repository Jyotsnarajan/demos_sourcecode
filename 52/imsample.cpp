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
 * @file imsample.cpp
 * @brief command-line handler for sampling patches 
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
#include "util.h" 		// routines dedicated to this implementation
#endif

using namespace std;


int main( int argc, char *const *argv ){


	//default parameters
	double flat_threshold = 1e4;
	double orient_threshold = 5;
	int minimal_num_samples = 5e3;
	string cpp_name("dataCXX.cpp");

	//message
	if( !(argc == 1 || argc == 5) ){

		cerr << endl << endl << endl;
		cerr << "\t\t#######################################################################################################" << endl << endl;
		cerr << "\t\t Before running this, make sure you've got a file named list.txt containing the paths to grayscale PNGs" << endl << endl;
		cerr << "\t\t#######################################################################################################" << endl << endl;
		cerr << "\t Usage : please specify the parameters for " << argv[0] << endl << endl; 
		cerr << "\t\t 1. flat_threshold: threshold for flat patches. It defaults to " << flat_threshold << endl;
		cerr << "\t\t 2. orient_threshold: threshold for oriented patches. It defaults to " << orient_threshold << endl;
		cerr << "\t\t 3. minimal_num_samples: minimal number of samples for each model. It defaults to " << minimal_num_samples << endl;
		cerr << "\t\t 4. data_file: output a .cpp file to use with DataProvider.h. It defaults to " << cpp_name << endl << endl;
		cerr << "\t For example: " << endl << endl;
		cerr << "\t\t ./imsample 10000 5 100 dataCXX.cpp" << endl << endl;
		cerr << "\t Or simply" << endl << endl;
		cerr << "\t\t ./imsample" << endl << endl;

        	return EXIT_FAILURE;
	}		


	//in this case, I need to check	
	if( argc == 5 ){
		//name handling
		string usr_filename(argv[4]);
		if( usr_filename.compare("DataProvider.cpp") == 0 ){
			cerr << "ERROR: you are not allowed to overwrite DataProvider.cpp. Please choose another name." << endl;
			exit(EXIT_FAILURE);
		}else{
			cpp_name.assign(usr_filename);			
		}

		//now to the numerical parameters	
		flat_threshold = atof(argv[1]);
		orient_threshold = atof(argv[2]);
		minimal_num_samples = atoi(argv[3]);
		if( min(flat_threshold, orient_threshold) < 0 ){
			cerr << "ERROR: neither flat_threshold nor orient_threshold should ever be negative." << endl;
			exit(EXIT_FAILURE);
		}
		if( minimal_num_samples < 1 ){
			cerr << "ERROR: "  << minimal_num_samples << " samples for each model? That doesn't make sense." << endl;
			exit(EXIT_FAILURE);
		}
	}

	//convert string to const char *
	char * chared = new char [cpp_name.size()+1];
	chared[cpp_name.size()] = 0;
	memcpy(chared, cpp_name.c_str(), cpp_name.size());
	const char * filename = chared;

	//main
	sample_images(	/*patch_size*/          8,
			/*flat_threshold*/	flat_threshold,
			/*orient_threshold*/ 	orient_threshold,
			/*num_orientations*/ 	18,
			/*minimal_num_samples*/ minimal_num_samples,
			/*data_file*/ filename );

	//clean up	
	delete [] chared;
}
