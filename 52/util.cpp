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
 * @file util.cpp
 * @brief utils for both PLE and SPLE 
 * @author Yi-Qing WANG <yiqing.wang@polytechnique.edu>
 */

#include <sys/time.h>		// time the routines
#include <fstream>		// read in noise data
#include "io_png.h" 		// image IO
#include <stack>

#ifndef STRING_H
#define STRING_H
#include <string>
#endif

#ifndef UTIL_H
#define UTIL_H
#include "util.h"
#endif


#ifndef RAND_H
#define RAND_H
extern "C"{
	#include "randmt.h"	//external random number generator
}
#endif

#ifndef STD_H
#define STD_H
#include <iomanip>
#include <iostream>
#include <cstdio>
#endif

#ifndef DATA_H
#define DATA_H
#include "DataProvider.h" 	// initial mixture
#endif

#ifndef SHORT_NEWMAT
#define SHORT_NEWMAT
#include "newmat10/newmatap.h"  // NEWMAT 
typedef NEWMAT::Matrix NMatrix;
typedef NEWMAT::SymmetricMatrix NSym;
typedef NEWMAT::DiagonalMatrix NDiag;
typedef NEWMAT::ColumnVector NColV;
typedef NEWMAT::RowVector NRowV;
#endif

typedef Matrix<double,Dynamic,Dynamic,RowMajor> RMatrixXd;
using namespace std;

void fail (const char *message) {
	cerr << message << endl;
	exit(EXIT_FAILURE);
}

//routine timer
void times (const char *which) {
	/* If which is not empty, print the times since the previous call. */
	static double last_wall = 0.0, last_cpu = 0.0;
	double wall, cpu;
	struct timeval tv;
	clock_t stamp;
	
	wall = last_wall;
	cpu = last_cpu;
	if (gettimeofday(&tv,NULL) != 0 || (stamp = clock()) == (clock_t)-1)
	    fail("Unable to get times");
	last_wall = tv.tv_sec+1.0e-6*tv.tv_usec;
	last_cpu = stamp/(double)CLOCKS_PER_SEC;
	if (strlen(which) > 0){
	    wall = last_wall-wall;
	    cpu = last_cpu-cpu;
	    printf("%s time = %.2f seconds, CPU = %.2f seconds\n",which,wall,cpu);
	}
}


//wrapper in Matlab style
void imread(	
		const char * file_name
,		int & image_rows
,		int & image_cols
, 		int & num_channels
,		MatrixXd *& image		
){
		size_t nx, ny, nc;
		float * pixel_stream = NULL;
		pixel_stream = read_png_f32( file_name, &nx, &ny, &nc );
		if( pixel_stream == NULL )
			fail("Unable to get the image");	
		
		//return these useful parameters 
		image_cols = (int)nx;
		image_rows = (int)ny;
		num_channels = (int)nc;
		
		//input stream assumes row-major while Eigen defaults to column-major
		Map<MatrixXf> parallel( pixel_stream, image_cols, image_rows * num_channels );
		image = new MatrixXd [ num_channels ];
		for( int ch = 0; ch < num_channels; ch++ )
			image[ ch ] = parallel.block( 0, ch*image_rows, image_cols, image_rows).transpose().cast<double>();

		//release
		free(pixel_stream);
		cout << "INFO: read in image " << file_name << " of dimension: (" << ny << " , " << nx << " , " << nc << ")" << endl;
}

//write image and release memory
void imwrite( 
		const char * file_name 
, 		MatrixXd * image
, 		int num_channels
){
	int image_cols = image[0].cols();
	int image_rows = image[0].rows();
	int pixels_per_channel = image_cols * image_rows; 
	float * output = new float [ pixels_per_channel * num_channels ];
	//this part should be straightforward but still be careful with the order
	#pragma omp parallel for schedule( static )
	for(int j = 0; j < image_cols; j++)
		for(int i = 0; i < image_rows; i++)
			for(int ch = 0; ch < num_channels; ch++)
				output[ ch*pixels_per_channel + i*image_cols + j ] = (float) image[ ch ](i,j);
	//release
	delete [] image;
	write_png_f32( file_name, output, (size_t) image_cols, (size_t) image_rows, (size_t) num_channels );
	delete [] output;
	cout << "INFO: write the image " << file_name << " to local folder." << endl;
}

//add noise to image
void add_gaussian_noise( 
		MatrixXd * image
, 		double sigma
,		int num_channels
,		bool generate_noise 
){
	int image_rows = image[0].rows();
	int image_cols = image[0].cols();
	int pixels_per_channel = image_rows * image_cols;

	//generate defaults to true
	//the other option allows you to read in noise
	//which is helpful in algo comparison and debugging
	if( generate_noise ){
		init_randmt_auto();
		double * noise = new double [ pixels_per_channel ];
		for(int ch = 0; ch < num_channels; ch++){
			for( int i = 0; i < pixels_per_channel; i++ )
				noise[ i ] = sigma * rand_normal();
			Map<MatrixXd> pure_noise( noise, image_rows, image_cols );
			image[ch] += pure_noise;
		}
		delete [] noise;
	}else{
		//noise_data is a binary file containing noise
		int num_data = pixels_per_channel * num_channels;
		double noise [num_data];
		FILE * noise_data = fopen( "noise_data", "rb" );
		if( noise_data == NULL )
			fail("Unable to get the noise data");
		size_t count = fread( noise, sizeof(double), num_data, noise_data );
		Map<MatrixXd> pure_noise( noise, image_rows, image_cols * num_channels );
		for( int ch = 0; ch < num_channels; ch++ )
			image[ch] += pure_noise.block( 0, ch * image_cols, image_rows, image_cols );
	}
	
	cout << "INFO: noise with sigma = " << sigma << " is added to the image." << endl;
	
}

//calc num of patches needed to cover the whole image
//according to a sliding window scheme specified by overlap
//which is the num of columns shared by horizontally neighboring patches
int num_patches( 
		int n_pixels
,		int patch_size
,		int overlap
){
	int step = patch_size - overlap;
//	it holds that for some k, n_pixels = patch_size + step * k + something
//	with something = 0 to k-1
	int something = (n_pixels - patch_size) % step;
	int correction;
	if( something == 0 )
		correction = 1;
	else
		correction = 2;
	int k = (n_pixels - something - patch_size)/step;
	return k + correction;
}

//reduce an image into patches
VectorXd ** image2patches( 
		MatrixXd const & image
,		int image_rows
,		int image_cols
,		int overlap
,		int patch_size 
){
	int map_rows = num_patches( image_rows, patch_size, overlap );
	int map_cols = num_patches( image_cols, patch_size, overlap );
	int data_size = pow( patch_size, 2 );
//	allocate some memory for patches
	VectorXd ** patch_at_coordinates = new VectorXd * [ map_rows ];
	for( int row = 0; row < map_rows; row++ )
		patch_at_coordinates[ row ] = new VectorXd [ map_cols ];
//	the patch upper left corner's coordinates
	int coordinate_j, coordinate_i = -1*patch_size;
//	coordinate cannot exceed max 
	int max_coordinate_i = image_rows - patch_size;
	int max_coordinate_j = image_cols - patch_size;
//	sliding window step
	int step = patch_size - overlap;
//	meat
	for( int i = 0; i < map_rows; i++ ){
		coordinate_i = max( 0, min( max_coordinate_i, coordinate_i + step ) );
		coordinate_j = -1*patch_size;
		for( int j = 0; j < map_cols; j++ ){
			coordinate_j = max( 0, min( max_coordinate_j, coordinate_j + step ) );
			MatrixXd patch = image.block( coordinate_i, coordinate_j, patch_size, patch_size ).transpose();
			patch_at_coordinates[i][j] = Map<MatrixXd>( patch.data(), data_size, 1 );
		}
	}
	cout << "INFO: the image is reduced to ( " << map_rows <<  " , " << map_cols << " ) patches." << endl;
	return patch_at_coordinates;
}

//assemble patches to form an image again
void patches2image(     
		VectorXd ** patch_at_coordinates
,		int overlap
, 		int patch_size
,		MatrixXd & image
,		bool normalize
){
	int image_rows = image.rows();
	int image_cols = image.cols();
//      reset the whole picture
        image.setZero( image_rows, image_cols );
//      mask counts for each pixel the number of patches covering it
        MatrixXd mask = image;
        int coordinate_j, coordinate_i = -1*patch_size;
	int max_coordinate_i = image_rows - patch_size;
	int max_coordinate_j = image_cols - patch_size;
	int step = patch_size - overlap;
//	block to mark the patch in the mask 
	MatrixXd block( patch_size, patch_size );
	block.setOnes( patch_size, patch_size );
        int map_rows = num_patches( image_rows, patch_size, overlap );
        int map_cols = num_patches( image_cols, patch_size, overlap );

        for( int i = 0; i < map_rows; i++ ){
                coordinate_i = max( 0, min( max_coordinate_i,  coordinate_i + step ) );
                coordinate_j = -1*patch_size;
                for( int j = 0; j < map_cols; j++ ){
                        coordinate_j = max( 0, min( max_coordinate_j, coordinate_j + step ) );
			//a transposition to reflect what has been done in image2patches 
                        //image.block(coordinate_i, coordinate_j, patch_size, patch_size) += Map<MatrixXd>( patch_at_coordinates[i][j].data(), patch_size, patch_size ).transpose();
			//equivalently, use RowMajor config 
                        image.block(coordinate_i, coordinate_j, patch_size, patch_size) += Map<RMatrixXd>( patch_at_coordinates[i][j].data(), patch_size, patch_size ); 
			if( normalize )
                        	mask.block(coordinate_i, coordinate_j, patch_size, patch_size) += block;
                }
        }
	if( normalize )
		image = image.cwiseQuotient( mask );
}

//color space transformation to enhance the 1st channel's SNR
void RGB_transform( 
		MatrixXd * image
,		int num_channels
,		bool inverse 
){
	//transformation is only defined for color image
	if( num_channels == 1 )
		return;
	MatrixXd * copy = new MatrixXd[ num_channels ];
	for( int ch = 0; ch < num_channels; ch++ )
		copy[ch] = image[ch];
	//meat
	if( !inverse ){
		// center signal rather than noise to be statistically consistent 
		image[0] = (copy[0] + copy[1] + copy[2])/3.;
		image[1] = (copy[0] - copy[2])/sqrt(2.);
		image[2] = (copy[0] - copy[1]*2. + copy[2])/sqrt(6.);
	}else{
		// copy[0] remains the same because of the change made above
		copy[1] /= sqrt(2.);
		copy[2] /= sqrt(6.);
		image[0] = copy[0] + copy[1] + copy[2]; 
		image[1] = copy[0] - 2.*copy[2];
		image[2] = copy[0] - copy[1] + copy[2];
	}
	delete [] copy;
} 


//calc image MSE and print it out
void show_image_MSE(
		MatrixXd & clean
,		MatrixXd & denoised
){
	int pixels_per_channel = clean.rows() * clean.cols();
	double MSE = (clean - denoised).squaredNorm()/pixels_per_channel;
	cout << " MSE = " << fixed << setprecision(5) << MSE << " RMSE = " << sqrt(MSE) << endl;
}

//display the colorful patch_map
void print_patch_map( 
		MatrixXi const & patch_map
,		bool is_patch_map
){
	//read in model to color mapping
	DataProvider mydata;
	double * cm = mydata.GetColormap();
	//in total 64 different colors
	Map<MatrixXd> colormap( cm, 64, 3 );


	//allocate memory for the patch_map
	int map_rows = patch_map.rows();
	int map_cols = patch_map.cols();
	MatrixXd temp( map_rows, map_cols );
	MatrixXd * patch_map_ptr = new MatrixXd [3];
	for( int ch = 0; ch < 3; ch++ )
		patch_map_ptr[ ch ] = temp;


	//range allows to print out other maps
	double range = 20;
	if( !is_patch_map )
		range = patch_map.maxCoeff() - patch_map.minCoeff() + 1;
	//fill in
	#pragma omp parallel for schedule( static )
	for( int row = 0; row < map_rows; row++ )
		for( int col = 0; col < map_cols; col++ ){
			int scale = patch_map( row, col );
			//+1 because scale ranges from 0 to 19
			scale = ceil( 64.0/range*(scale+1) ) - 1;
			for( int ch = 0; ch < 3; ch++ )
				patch_map_ptr[ch](row, col) = colormap(scale, ch)*255;
		}


	//print it out
	if( is_patch_map )
		imwrite( "patchmap.png", patch_map_ptr, 3 );	
	else
		imwrite( "othermap.png", patch_map_ptr, 3 );
}

void vectorArray2Matrix( 
		VectorXd ** varray
,		int row_id
,		MatrixXd & vmatrix
){
	int n_cols = vmatrix.cols();
	int n_rows = vmatrix.rows();
	for( int col = 0; col < n_cols; col++ )
		vmatrix.block( 0, col, n_rows, 1 ) = varray[row_id][col];
}

//tensor structure orientation detector
int tensorStructure(
		MatrixXd & image
,		int row
,		int col
,		SamplingParams & params	
){
	Matrix2d tensor = Matrix2d::Zero();
	static int patch_size = params.patch_size;
	//first I need to compute gradient at all the pixel sites in the patch
	for( int i = row; i < row + patch_size; i++ )	
		for( int j = col; j < col + patch_size; j++ ){
			//I'll always avoid image boundary, so relax
			//this scheme of differentiation is better as the second order error is gone
			double row_grad = image(i+1,j)	- image(i-1,j);
			double col_grad = image(i,j+1) - image(i,j-1);
			Vector2d grad;
			grad << row_grad, col_grad;	
			tensor += grad*grad.transpose();
		}
	//the traditional Ax = \lambda x thing didn't prove numerically reliable. Sorry, fall back to NewMat
	NMatrix cov_container(2, 2);
	cov_container << tensor.data(); 
	NSym sym_cov(2);
	sym_cov << cov_container;
	NMatrix U(2, 2);
	NDiag D(2);
	SVD( sym_cov, D, U  );
	double big_val = D(1,1);
	double small_val = D(2,2); 
	//now we are going to decide 
	int model;
	if( big_val/small_val < params.orient_threshold ){
		model = big_val < params.flat_threshold ? params.flat_model : params.textural_model;
	}else{
		//take the eigenvector associated with the larger eigenvalue
		double theta = atan(U(2,1)/U(1,1));
		theta = theta >= 0 ? theta : theta + M_PI;
		model = floor(theta*params.num_orientations/M_PI);
		//sometimes, if theta is a very small negative, shit does happen
		if( model == params.textural_model ){
			cout << "INFO: a tiny numerical approximation leads to a big quantification error. Corrected!" << endl;
			model = model - 1;
		}	
	}
	return model;
}

//do the sampling and output the result
void sample_images(	
		int patch_size
,		double flat_threshold
,		double orient_threshold 
,		int num_orientations
,		int minimal_num_samples
,		const char * filename
){
	//feed the parameters into a single struct
	SamplingParams params;
	params.patch_size = patch_size;
	int data_size = pow(patch_size, 2);
	params.num_orientations = num_orientations;
	int num_models = params.num_orientations + 2;
	//the model index starts from 0
	params.flat_model = num_models - 1;
	params.textural_model = num_models - 2;
	params.flat_threshold = flat_threshold;
	params.orient_threshold = orient_threshold;
	params.minimal_num_samples = minimal_num_samples;
	params.filename = filename;
	params.verbose = true;

	//we use the image list produced by the bash script, getPaths.sh
	char ** addr = NULL;
	int num_images = 0;
	for( int loop = 0; loop < 2; loop++ ){
		ifstream myfile ("list.txt");
		if(!myfile.is_open()){
			cerr << "ERROR: Unable to open list.txt!" << endl;
			exit(EXIT_FAILURE);	
		}
		string line;
		//an offset to make the line count right 
		int total = -1;
		while( myfile.good() ){
			getline( myfile, line );
			total += 1;
			if( loop == 1 && total < num_images ){
				//convert string to char array
				addr[total] = new char [line.size()+1];
				addr[total][line.size()] = 0;
				memcpy(addr[total], line.c_str(), line.size());
				//cout << addr[total] << endl;
			}
		}
		myfile.close();
		//first loop just count the number of lines in the file
		if( loop == 0 ){
			num_images = total;
			cout << "INFO: there are " << total << " grayscale PNGs." << endl;
			if( total < 1 ){
				cerr << "ERROR: No grayscale PNGs indicated in list.txt." << endl;
				exit(EXIT_FAILURE);
			}
			addr = new char * [total];
		}
	}

	cout << "INFO: total number of models in the mixture : " << num_models << endl;
	//get down to the main business	
	init_randmt_auto();
	//dynamic storage
	stack<VectorXd> stack[num_models];
	//model mean
	VectorXd mean[num_models];
	for(int model = 0; model < num_models; model++)
		mean[model].setZero(data_size);
	//count the number of samples collected for each model
	VectorXd counter(num_models);
	counter.setZero();
	
	//meat	
	while( counter.minCoeff() < params.minimal_num_samples ){
		//randomly choose an image	
		const char * file = addr[(int)floor(num_images*rand_unif())];
		MatrixXd * image = NULL;
		int num_channels, image_rows, image_cols;
		imread( file, image_rows, image_cols, num_channels, image );
		//an arbitrary margin so that differential can be defined in any way anywhere
		int margin = max(15, patch_size + 5);
		//randomly choose a coordinate
		int row = margin + (int)floor((image_rows - 2*margin)*rand_unif());
		int col = margin + (int)floor((image_cols - 2*margin)*rand_unif());
		//calculate tensor structure
		int model = tensorStructure(image[0], row, col, params);
		counter(model) += 1;
		if(params.verbose){
			cout << "INFO: so far we've collected for each model: " << endl;
			cout << counter.transpose() << endl;
		}
		//stored the patch
		MatrixXd patch = image[0].block(row, col, patch_size, patch_size).transpose();
		stack[model].push(Map<MatrixXd>(patch.data(), data_size, 1));
		mean[model] += Map<MatrixXd>(patch.data(), data_size, 1);
		//release
		delete [] image;
	}
	
	//all right, now compute all the statistics
	VectorXd mixing_weights = counter/counter.sum();	
	MatrixXd cov_mats[num_models];
	for(int model = 0; model < num_models; model++){
		mean[model] /= counter(model);
		cov_mats[model].setZero(data_size, data_size);
		while( !stack[model].empty() ){
			VectorXd container = stack[model].top() - mean[model];
			cov_mats[model] += container*container.transpose(); 
			stack[model].pop();
		}
		cov_mats[model] /= counter(model);
	}
	
	//produce a script	
	write_out_everything(params, cov_mats, mean, mixing_weights);

	//release
	for(int i = 0; i < num_images; i++ )	
		delete [] addr[i];
	delete [] addr;
}	

//write out an immediately usable cpp
void write_out_everything(
			SamplingParams & params
,			MatrixXd * cov_mats
,			VectorXd * mean
,			VectorXd mixing_weights 
){
	if(params.verbose)
		cout << "INFO: the initial mixing weights is " << mixing_weights.transpose() << endl;

	int num_models = params.num_orientations + 2;
	int data_size = pow(params.patch_size, 2);
	ofstream outputFile(params.filename);

	//get header and colormap from DataProvider.cpp
	ifstream readin("DataProvider.cpp");
	if(!readin.is_open()){
		cerr << "ERROR: Unable to open DataProvider.cpp!" << endl;
		exit(EXIT_FAILURE);	
	}
	string line;	
	//copy the first 25 lines
	for(int i = 0; i < 25; i++){
		getline(readin, line);
		outputFile << line << endl;
	}		
	readin.close();

	//first write out the mixing weights
	outputFile << "double DataProvider::prior_0" << params.patch_size << "[] = {";
	for(int model = 0; model < num_models - 1; model++)
		outputFile << mixing_weights(model) << ", ";
	outputFile << mixing_weights(num_models-1) << endl << "};" << endl;

	//now write out the model means
	outputFile << "double DataProvider::mu_0" << params.patch_size << "[" << num_models << "]";
	outputFile << "[" << data_size << "] = {";
	for(int model = 0; model < num_models; model++){
		outputFile << "{";
		for(int i = 0; i < data_size; i++){
			outputFile << mean[model](i);
			if(i < data_size - 1)
				outputFile << ", ";
		}
		outputFile << endl << "}";
		if(model < num_models - 1)
			outputFile << "," << endl;
	}
	outputFile << "};" << endl;
	
	//the model covariance	
	outputFile << "double DataProvider::cov_0" << params.patch_size << "[" << num_models << "]";
	outputFile << "[" << pow(data_size, 2) << "] = {";
	for(int model = 0; model < num_models; model++){
		outputFile << "{";
		int data_size_square = pow(data_size, 2);
		for(int i = 0; i < data_size_square; i++){
			outputFile << cov_mats[model].data()[i];
			if(i < data_size_square - 1)
				outputFile << ", ";
		}
		outputFile << endl << "}";
		if(model < num_models - 1)
			outputFile << "," << endl;
	}
	outputFile << "};" << endl;

	outputFile.close();	
}
