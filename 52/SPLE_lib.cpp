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
 * @file SPLE_lib.cpp
 * @brief library for SPLE denoising algorithm
 * @brief SPLE is implemented mainly with Eigen (Newmat for SVD) 
 * @brief PLE is implemented mainly with Newmat 
 * @author Yi-Qing WANG <yiqing.wang@polytechnique.edu>
 */


#include <functional>		// connected component labeling
#include "connected.h"		// connected component labeling

#ifndef SPLE_H
#define SPLE_H
#include "SPLE_lib.h" 		// routines dedicated to this implementation
#endif

#ifndef RAND_H
#define RAND_H
extern "C"{
	#include "randmt.h"	//external random number generator
}
#endif

#ifndef STD_H
#define STD_H
#include <iostream>
#include <iomanip>
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

#ifndef UTIL_H
#define UTIL_H
#include "util.h"
#endif

#ifdef _OPENMP
#include <omp.h>		// OpenMP
#endif

using namespace std;
using namespace Eigen;

 

//return num of factors given model and standard setting
int ret_num_factors(
		int std_num_factors
,		int data_size
, 		int model
,		int num_models
){
	//remove static if there is model culling
	static int flat_model = num_models - 1;
	static int textural_model = num_models - 2;
	if( model == flat_model )
		return 1;
	else if( model == textural_model )
		return data_size - 1;
	else
		return std_num_factors;
}


//read in the initial config
//the default setting is 18 orientation + 2 mixture 
//patch size can be either 8 or 6
void read_config(	
		int data_size
,		int std_num_factors
,		int num_models
,		VectorXd *& prior
,		VectorXd *& mus
,		MatrixXd *& factors
 ){
	bool regularize_cov = true;
	if( regularize_cov )
		cout << "INFO: covariance regularization is planned." << endl;
	else
		cout << "INFO: covariance regularization is NOT planned." << endl;;
	int patch_size = (int)sqrt(data_size);
	//initial GMM setup
	DataProvider my_data;
	//mixing weights
	prior = new VectorXd [ 1 ];
	prior[0] = Map<MatrixXd>( my_data.GetPrior( patch_size ), num_models, 1 );
	//model mean
	mus = new VectorXd [ num_models ];
	//model factor loading
	factors = new MatrixXd [ num_models ];
	for( int model = 0; model < num_models; model++ ){
		mus[ model ] = Map<MatrixXd>( my_data.GetMu( patch_size, model ), data_size, 1 );
		MatrixXd cov_mat = Map<MatrixXd>( my_data.GetCov( patch_size, model ), data_size, data_size );
		//all the rest depends on svd
		MatrixXd eig_vecs( data_size, data_size );
		VectorXd eig_vals( data_size );
		deduce_eigens( cov_mat, std_num_factors, data_size, model, num_models, 0, false, false, eig_vecs, eig_vals );
		//determine the number of factors for this model
		int num_factors = ret_num_factors( std_num_factors, data_size, model, num_models );
		//regularize the covariance matrices
		double average_noise_energy = 0;
		int redundancy = data_size - num_factors;
		if( regularize_cov && redundancy > 0 )
			average_noise_energy = eig_vals.tail( redundancy ).sum()/(double)redundancy;
		VectorXd cst;
		cst.setConstant( num_factors, average_noise_energy );
		VectorXd sqrt_eig_vals = (eig_vals.head( num_factors ) - cst).cwiseSqrt();
		
		//the normalized factors
		factors[ model ] = eig_vecs.block( 0, 0, data_size, num_factors );
		//do not confuse eig vectors with factors
		factors[ model ] *= sqrt_eig_vals.asDiagonal();
		//for testing 
		if( model == 0 && false ){
			cout << " model  : " << model << " : " << num_factors << endl;
			//Map<MatrixXd> showp( factors[model].block(0, 0, data_size, 1 ).data(), patch_size, patch_size );
			//cout << factors[model].block(0, 0, data_size, 1 )<< endl;
		}
	}
	//for testing
	if( false ){
		cout << "INFO: here comes the initial prior" << endl;
		cout << fixed << setprecision(5) << prior[0] << endl;
		cout << "INFO: and they sum to " << prior[0].sum() << endl;
	}
}


//deduce factors and their eigen-values from a covariance matrix
//this routine is needed as several places in this implementation
void deduce_eigens(
		MatrixXd & cov_mat 
,		int std_num_factors
,		int data_size
,		int model
,		int num_models
, 		double squared_sigma
,		bool inverse_eig_vals
,		bool trim
,		MatrixXd & eig_vecs
,		VectorXd & eig_vals
){
		//standard SVD: Methods provided in Eigen can be flaky
		//Revert to what I know is reliable: NEWMAT
		NMatrix cov_container( data_size, data_size );
		cov_container << cov_mat.data(); 
		NSym sym_cov( data_size );
		sym_cov << cov_container;
		NMatrix U( data_size, data_size );
		NDiag D( data_size );
		SVD( sym_cov, D, U  );
		//convert to the other world
		//the index in NEWMAT library starts at 1 instead of 0
		MatrixXd U_matrix( data_size, data_size );
		VectorXd D_vector( data_size );
		for( int i = 0; i < data_size; i++ ){
			D_vector(i) = D(i+1);
			for( int j = 0; j < data_size; j++ )
				U_matrix(i, j) = U(i+1,j+1);
		}
		//determine the number of factors for this model
		if( trim ){
			int num_factors = ret_num_factors( std_num_factors, data_size, model, num_models );
			eig_vecs = U_matrix.block( 0, 0, data_size, num_factors ).transpose();
			if( inverse_eig_vals ){
				VectorXd ss;
				ss.setConstant( num_factors, squared_sigma );
				ss += D_vector.head( num_factors ); 
				eig_vals = ss.cwiseInverse();
			}else
				eig_vals = D_vector.head( num_factors );
		}else{
			eig_vals = D_vector; 
			eig_vecs = U_matrix; 
		}
}

//one of the most time consuming part of the algorithm
//calc for all (patch, model) couple their probability (scaled) 
//and form the patch map for SURE-aided adaptive filtering
void calc_responsibilities(	
		VectorXd ** noisy_patches_at_coordinates
, 		MatrixXd * factors
, 		VectorXd * mus
,		VectorXd * prior
,		int map_rows
,		int map_cols
,		int data_size
,		int std_num_factors	
,	 	double sigma
, 		VectorXd **& responsibility 
, 		MatrixXi & patch_map
){

	cout << "INFO: calculating model responsibilities..." << endl;
//	allocate memory just once for later responsibility will not be NULL any more
	if( responsibility == NULL ){
		responsibility = new VectorXd * [ map_rows ];	
		for( int row = 0; row < map_rows; row++ )
			responsibility[ row ] = new VectorXd [ map_cols ];
	}
		
//	compute the models' eigenvalues and eigenvectors to avoid repeated operations 
	int num_models = prior[0].rows();
//	inverse eigenvalues for density calculation	
	VectorXd * inv_eig_vals = new VectorXd [ num_models ];
//	eigenvectors
	MatrixXd * eig_vecs = new MatrixXd[ num_models ];

	double squared_sigma = sigma * sigma;

//	the first set of factors are orthogonal while it might not be the case in later iterations
//	as EM is supposed to approximate, rather than solve the density maximization problem
//	this part is taken out because the values in it will be repeatedly used later in the function
	#pragma omp parallel for schedule( static ) 
	for( int model = 0; model < num_models; model++ ){
		MatrixXd cov = factors[ model ] * factors[ model ].transpose();
		int num_factors = ret_num_factors( std_num_factors, data_size, model, num_models );
		MatrixXd temp_eig_vecs( num_factors, data_size );
		VectorXd temp_eig_vals( num_factors );
		deduce_eigens( cov, std_num_factors, data_size, model, num_models, squared_sigma, true, true, temp_eig_vecs, temp_eig_vals );
		eig_vecs[ model ] = temp_eig_vecs;
		inv_eig_vals[ model ] = temp_eig_vals;
	}

//	now, we can compute the patch responsibilities, up to a multiplicative constant
//	if for one particular patch, no model yields a positive density, let us default
//	the unknown model to the textural_model at the risk of producing artifacts
//	or to the flat one at the risk of producing equally undesirable blur
	int flat_model = num_models - 1;
	int textural_model = num_models - 2;
	int default_model = flat_model;
	VectorXd fake;
	fake.setZero( num_models );
	fake( default_model ) = 1;


//	get the business started here: calc responsibility as well as the patch map 
	#pragma omp parallel for schedule( static ) 
	for( int row_id = 0; row_id < map_rows; row_id++ )
		for( int col_id = 0; col_id < map_cols; col_id++ ){
			double highest;
			int best_model;
			VectorXd densities( num_models );
			for( int model = 0; model < num_models; model++ ){
				int num_factors = ret_num_factors( std_num_factors, data_size, model, num_models );
				double val = prior[0](model)*gaussian_density_EM( noisy_patches_at_coordinates[row_id][col_id], mus[model], inv_eig_vals[model], eig_vecs[model], squared_sigma, num_factors);
				densities( model ) = val;
				if( model == 0 || val > highest ){
					highest = val;
					best_model = model;
				}
			}
			double test = densities.sum();
			if( test == 0 || isinf(test) ){
				//only report serious error
				if( isinf(test) ){
					cerr << "#######################################################################" << endl;
					cerr << "Error: wrong density sum : " << test << " The program could break"  << endl;
					cerr << "problem patch: row_id : " << row_id << " col_id : " << col_id << endl;
					cerr << "#######################################################################" << endl;
				}
				responsibility[ row_id ][ col_id ] = fake;
				patch_map( row_id, col_id ) = default_model; 	
			}else{
				//normalize the density so they become MAP probability
				responsibility[ row_id ][ col_id ] = densities/test;
				patch_map( row_id, col_id ) = best_model;
			}
		}
	
	delete [] eig_vecs;
	delete [] inv_eig_vals;
}


//calc gaussian density for EM up to a multiplicative factor 
//where x is the variable that samples the density
//mu is the gaussian vector's expectation
double gaussian_density_EM( 
			VectorXd const & x
,			VectorXd const & mu
, 			VectorXd const & inv_eig_vals
,			MatrixXd const & eig_vecs
,			double squared_sigma
,			int num_factors 
){
	//projection
	VectorXd centered = x - mu;
	VectorXd coordinates = eig_vecs * centered;
	//logNumerator
	double logNumerator = coordinates.cwiseAbs2().dot(inv_eig_vals)+(centered.squaredNorm()-coordinates.squaredNorm())/squared_sigma;
	//static because the density is only computed for the first channel, otherwise should be removed 
	static double logSSigma = log(squared_sigma); 
	static int data_size = x.rows();
	//300 helps ensure that density values remain in an acceptable numerical range
	double result = exp(300 + 0.5*(logSSigma*(num_factors - data_size) - logNumerator));
	//remaining terms
	double inv_product = 1e200;
	for( int i = 0; i < num_factors; i++ )
		inv_product *= inv_eig_vals(i);
	//ok
	result *= sqrt(inv_product);
	if( isinf(result) )
		cerr << "ERROR: find inf density. Please adjust the multiplicative constants" << endl;
	return result;		
}

void filter_patches(	
		VectorXd ** noisy_patches
, 		MatrixXd * factors
,		VectorXd * mus
,		VectorXd ** responsibility
,		MatrixXi const & patch_map
,		int image_rows
,		int image_cols
,		int overlap
,		int ch
,		double sigma
, 		bool only_Wiener
,		bool both_filters
,		MatrixXd & model_MSE
,		double & SURE_mean
,		VectorXd **& restored_patches
,		bool Testing_Mode
){
	int map_rows = patch_map.rows();
	int map_cols = patch_map.cols();
	int num_models = model_MSE.rows();
	//a fresh start
	model_MSE.setZero( num_models, 2 );

 
	if( ch > 0 ){
		//for color images, I do what I please for an algorithm speedup
		//only do Donoho Minimax in this case
		both_filters = false;
		only_Wiener = false;
	}
	
	if( both_filters ){

		//first let Wiener process the patches
		filter( noisy_patches, factors, mus, responsibility, patch_map, model_MSE, ch, sigma, true, restored_patches );
 
		//now it is Donoho's turn
		MatrixXd Donoho_model_MSE;
		Donoho_model_MSE.setZero( num_models, 2 );
		VectorXd ** Donoho_restored_patches = NULL;	
		filter( noisy_patches, factors, mus, responsibility, patch_map, Donoho_model_MSE, ch, sigma, false, Donoho_restored_patches );
 
		//evaluate relative performance of these two filters
		bool switch2Donoho[ num_models ];
		for( int model = 0; model < num_models; model++ )
			if( model_MSE( model, 0 ) > Donoho_model_MSE( model, 0 ) ){
				switch2Donoho[ model ] = true;
				model_MSE( model, 0 ) = Donoho_model_MSE( model, 0 );
			}
			else
				switch2Donoho[ model ] = false;
	
		//based on SURE, I switch between Wiener and Donoho 
		#pragma omp parallel for schedule( static )
		for( int row = 0; row < map_rows; row++ )
			for( int col = 0; col < map_cols; col++ )
				if( switch2Donoho[patch_map(row, col)] )
					restored_patches[row][col] = Donoho_restored_patches[row][col];

		//mission accomplished
		for( int row = 0; row < map_rows; row++ )
			delete [] Donoho_restored_patches[row];
		delete [] Donoho_restored_patches;	
	
	}else{
		if( only_Wiener )
			filter( noisy_patches, factors, mus, responsibility, patch_map, model_MSE, ch, sigma, true, restored_patches ); 
		else	
			filter( noisy_patches, factors, mus, responsibility, patch_map, model_MSE, ch, sigma, false, restored_patches ); 
	}
//	calc filter real time performance indicator: actually I don't need the second column
	SURE_mean = model_MSE.col(0).sum();
	static int num_patches = map_rows * map_cols; 
	SURE_mean /= num_patches;
//	an additional flat patch expansion designed to make the sky look bluer
	diffusion_denoise( noisy_patches, patch_map, num_models, sigma, overlap, image_rows, image_cols, restored_patches, Testing_Mode );
}


//the non-linear minimax coordinate shrinkage filter of Donoho and Johnstone
//or Wiener filter (or even SureShrinkage implemented in the first version, now deleted)
//because the latter does not offer observable improvement due to limited coordinate number I guess
void filter(	
		VectorXd ** noisy_patches 
,		MatrixXd * factors
,		VectorXd * mus
,		VectorXd ** responsibility
,		MatrixXi const & patch_map
,		MatrixXd & model_MSE 	//record for each cluster the (L) total MSE (R) number of patches received
,		int ch
,		double sigma
,		bool is_Wiener 	    	//Donoho or Wiener
,		VectorXd **& restored_patches
){
	if( is_Wiener )
		cout << "INFO: apply Wiener L2 filter" << endl;
	else
		cout << "INFO: apply Donoho minimax filter" << endl;

//	general parameters
	int map_rows = patch_map.rows();
	int map_cols = patch_map.cols();	
	int data_size = noisy_patches[0][0].rows();
	int num_models = model_MSE.rows();
	int std_num_factors = factors[0].cols();
	double squared_sigma = sigma * sigma;

//	prepare the filter paramters
//	the classical Wiener ratio	
	VectorXd * Wiener_shrinkage = NULL;
	if( is_Wiener )
		Wiener_shrinkage = new VectorXd [ num_models ];


//	sum of these ratios for computing SURE, well an approximation 
	VectorXd sum_shrink( num_models ); 
	MatrixXd * filter_basis = new MatrixXd [ num_models ];
//	prepare filtering bases for both Wiener and Donoho and Wiener Shrinkage if required
	#pragma omp parallel for schedule( static ) 
	for( int model = 0; model < num_models; model++ ){
		MatrixXd cov = factors[model]*factors[model].transpose();
		MatrixXd eig_vecs( data_size, data_size );
		VectorXd eig_vals( data_size );
		deduce_eigens( cov, std_num_factors, data_size, model, num_models, 0, false, false, eig_vecs, eig_vals );
		int num_factors = ret_num_factors( std_num_factors, data_size, model, num_models );
		//now I have the optimal basis for patch representation and thus denoising
		filter_basis[model] = eig_vecs.block(0, 0, data_size, num_factors).transpose();	
		if( is_Wiener ){
			//but for Wiener filtering, filtering ratio is still needed
			//ss = signal strength
			VectorXd ss = eig_vals.head(num_factors);
			//sn = signal + noise
			VectorXd sn;
			sn.setConstant(num_factors, squared_sigma);
			sn += ss;
			Wiener_shrinkage[model] = ss.cwiseQuotient(sn);
			//this is intended for SURE 
			sum_shrink(model) = Wiener_shrinkage[model].sum();
		}
	}

//	allocate memory for 
	restored_patches = new VectorXd * [ map_rows ];
	for( int row = 0; row < map_rows; row++ )
		restored_patches[ row ] = new VectorXd [ map_cols ];

//	denoising starts
	#pragma omp parallel for schedule( static ) if( is_Wiener ) 
	for( int row_id = 0; row_id < map_rows; row_id++ )
		for( int col_id = 0; col_id < map_cols; col_id++ ){
			VectorXd noisy_patch = noisy_patches[row_id][col_id];
			VectorXd estimated_patch;
			estimated_patch.setZero(data_size);
			int model = patch_map( row_id, col_id );
			//for SURE
			double correcting_term = 0;

			if( is_Wiener ){
				//it is only an approximation as explained in the paper
				correcting_term = responsibility[row_id][col_id].dot( sum_shrink );
				Wiener_Conditional_Expectation( noisy_patch, responsibility[row_id][col_id], mus, filter_basis, Wiener_shrinkage, num_models, data_size, estimated_patch );
			}else{
				// Donoho minimax filtering
				
				//projection
				//for the 2nd and 3rd transformed channels, it would be better to assume that their means vanish
				int num_factors = ret_num_factors( std_num_factors, data_size, model, num_models );
				VectorXd coordinates( num_factors );
				if( ch == 0 )
					coordinates = filter_basis[model]*(noisy_patch - mus[model]);		
				else
					coordinates = filter_basis[model]*noisy_patch;	

				//thresholding	
				//this threshold might be too conservative according to Mallat
				double threshold =  sigma * sqrt( 2 * log( num_factors ) );
				bool hard_shrinkage = true;
				if( hard_shrinkage ){
					//hard shrinkage works better, though mathematically more difficult to analyze
					for( int i = 0; i < num_factors; i++ )
						if( abs(coordinates(i)) < threshold )
							coordinates(i) = 0;
						else	
							correcting_term += 1;	
				}else{
					for( int i = 0; i < num_factors; i++ )
						if( abs(coordinates(i)) < threshold )	
							coordinates(i) = 0;
						else{
							double sign = 1;
							if( coordinates(i) < 0 )
  								sign = -1;
							coordinates(i) = sign * ( abs(coordinates(i)) - threshold );	
							correcting_term += 1;	
						}
				}
			
				//reconstruct
				estimated_patch = (coordinates.transpose() * filter_basis[model]).transpose();
				if( ch == 0 )
					estimated_patch += mus[model];
			}

			restored_patches[row_id][col_id] = estimated_patch;

			//compute SURE
			double sum = (estimated_patch - noisy_patch).squaredNorm();
			double increment = sum/data_size - squared_sigma*(1 - 2*correcting_term/data_size);
			#pragma omp critical
			{
			//	the total number of patches can be obtained without this
			//	model_MSE( model, 1 ) += 1;
				model_MSE( model, 0 ) += increment;
			}
		}

	//release
	delete [] Wiener_shrinkage;
	delete [] filter_basis;
}


//use prior to calculate the conditional expectation
void Wiener_Conditional_Expectation( 	
		VectorXd const & noisy_patch
,		VectorXd const & proba_weight
,		VectorXd * mus
,		MatrixXd * filter_basis
,		VectorXd * Wiener_shrinkage
,		int num_models
,		int data_size
,		VectorXd & estimated_patch
){
	//all models participate in the restoration
	VectorXd * model_estimate = new VectorXd [num_models];
	//proba_threshold helps accelerate the algorithm and won't change anything numerically 
	static double proba_threshold = 1e-3;
	//proba_sum is used as a normalizer 
	double proba_sum = 0;
	//removed because it helps reduce execution cost
	//#pragma omp parallel for schedule( static ) reduction(+:proba_sum) 
	for( int model = 0; model < num_models; model++ ){
		double proba = proba_weight(model);
		if( proba < proba_threshold )
			continue;
		proba_sum += proba;
		//project and Wiener filtering
		VectorXd coef = Wiener_shrinkage[model].cwiseProduct( filter_basis[model] * (noisy_patch - mus[model]) );
		//reconstruct
		model_estimate[model] = (coef.transpose()*filter_basis[model]).transpose()+mus[model];
	}

	//combine linearly all the estimates
	for( int model = 0; model < num_models; model++ ){
		double weight = proba_weight( model );
		if( weight >= proba_threshold )
			estimated_patch += model_estimate[model]*weight/proba_sum;
	}
	//release
	delete [] model_estimate;
}

//run a two pass connected component algorithm on a binary image 
void connected_components(	
		MatrixXi const & binary
,		int map_rows
,		int map_cols
,		int *& regions
,		bool Testing_Mode
){
	cout << "INFO: TWO-PASS Connected Components Algorithm.." << endl;
	if( Testing_Mode ){
		//see the binary map the algorithm is operating on
		MatrixXd * op = new MatrixXd [1];
		op[0] = binary.cast<double>() * 255;
		imwrite( "binary.png", op, 1 );
	}
	//the return
   	regions = new int [ map_rows * map_cols ];
    	ConnectedComponents cc( 50 );
	//the boolean false here to enforce 4-connectivity rather than 8-connectivity
      	cc.connected( binary.data(), regions, map_rows, map_cols, std::equal_to<int>(), false );
}

//test null hypothesis that both patches share the same signal
//show is a parameter left over for testing purpose
bool test_null( 
	VectorXd const & patch_1
, 	VectorXd const & patch_2
,	double sigma
,	bool show  
){
	double statistic = (patch_1-patch_2).squaredNorm();
	double double_squared_sigma = pow( sigma, 2 ) * 2;
	statistic /= double_squared_sigma;
	return statistic < 65 ? true : false;
}

void diffusion_denoise(	
		VectorXd  ** noisy_patches
,		MatrixXi const & patch_map
,		int num_models
,		double sigma
,		int overlap
,		int image_rows
,		int image_cols
,		VectorXd ** & restored_patches
,		bool Testing_Mode
){

//	general parameters
	int map_rows = patch_map.rows();
	int map_cols = patch_map.cols();
	int data_size = noisy_patches[0][0].rows();
	int patch_size = sqrt( data_size );

	//the leap one makes from one patch to another to have independently generated noise
	double step = patch_size - overlap;
	int leap = ceil( patch_size/step );

	int flat_model = num_models - 1;
	//get a binary map to run the connected component algorithm 
	MatrixXi all_flat;
	all_flat.setConstant( map_rows, map_cols, flat_model );
	MatrixXi is_flat = patch_map.cwiseEqual( all_flat ).cast<int>();
	int * region_indices = NULL;
	connected_components( is_flat, map_rows, map_cols, region_indices, Testing_Mode );
	Map<MatrixXi> regions( region_indices, map_rows, map_cols );
	//maybe you like to see the region map as well
	if( Testing_Mode )
		print_patch_map( regions, false );



	//find friends	
	//allocate memory to avoid use critical clause in the next loop
	//an entry in helper_coordinates is a similar patch which helps denoise the current one
	std::vector< Vector2i > ** helper_coordinates = new std::vector< Vector2i > * [ map_rows ];
	for( int row = 0; row < map_rows; row++ )
		helper_coordinates[ row ] = new vector< Vector2i > [map_cols];
	//friends are found based on two criteria: 
	//1. they must stay in the same region
	//2. they must look quite similar
	#pragma omp parallel for schedule( static )
	for( int row_id = 0; row_id < map_rows; row_id++ )
		for( int col_id = 0; col_id < map_cols; col_id++ )
			if( is_flat( row_id, col_id ) == 1 ){  
				VectorXd myself = noisy_patches[row_id][col_id];	
				Vector2i center;
				center << row_id, col_id;
				//count myself in
				helper_coordinates[row_id][col_id].push_back( center );
				int region_index = regions( row_id, col_id );
				//smoothing window size is a bit arbitrary
				int window = 10;
				int min_row = max( row_id - window, 0 );
				int min_col = max( col_id - window, 0 );
				int max_row = min( row_id + window, map_rows - 1 );
				int max_col = min( col_id + window, map_cols - 1 );
				for( int row = min_row; row <= max_row; row += leap )
					for( int col = min_col; col <= max_col; col += leap )
						if( regions( row, col ) == region_index && !( row == row_id && col == col_id ) ){	
							VectorXd comrade = noisy_patches[row][col];
							if( test_null( comrade, myself, sigma ) ){
								Vector2i coordinates;
								coordinates << row, col; 
								helper_coordinates[row_id][col_id].push_back( coordinates );
							}
					}
			}

	//release
	delete [] region_indices;
	

	//even a statistical rule won't work if noise is too strong: because the statistic will have too big a variance to be reliable 
	//in this case, let me decide for you. !Testing_Mode = DEMO mode in which we do not compare to save time
	if( sigma > 20 || !Testing_Mode ){
		//patch expansion starts there
		//counter counts the number of times each patch is denoised so that we could recover each patch correctly
		//sum_averages together with counter will determine the denoised value for a flat patch
		MatrixXi counter;
		counter.setZero( map_rows, map_cols );
		MatrixXd sum_averages;
		sum_averages.setZero( map_rows, map_cols );
		int num_helpers;
		for( int row = 0; row < map_rows; row++ )
			for( int col = 0; col < map_cols; col++ )
				if( num_helpers = helper_coordinates[row][col].size() ){
					double denoised = 0;
					for( int m = 0; m < num_helpers; m++ ){
						Vector2i coordinates = helper_coordinates[ row ][ col ][ m ];
						int row_id = coordinates(0);
						int col_id = coordinates(1);
						denoised += noisy_patches[row_id][col_id].sum();
					}
					//average might be a better name
					denoised /= num_helpers * data_size;
					for( int m = 0; m < num_helpers; m++ ){
						Vector2i coordinates = helper_coordinates[ row ][ col ][ m ];
						int row_id = coordinates(0);
						int col_id = coordinates(1);
						sum_averages( row_id, col_id ) += denoised;
						counter( row_id, col_id ) += 1;
					}
				}

		//patch expansion done now let's translate it into pixel values
		#pragma omp parallel for schedule( static ) 
		for( int row = 0; row < map_rows; row++ )
			for( int col = 0; col < map_cols; col++ )
				if( helper_coordinates[ row ][ col ].size() )
					restored_patches[row][col].setConstant(data_size, sum_averages(row, col)/counter(row, col));		
	}else
		flat_SURE( noisy_patches, patch_map, helper_coordinates, image_rows, image_cols, num_models, overlap, sigma, restored_patches );

	//release
	for( int row = 0; row < map_rows; row++ )
		delete [] helper_coordinates[ row ];
	delete [] helper_coordinates;
}

//calculate flat region SURE and see if patch expansion can be
//justified in a statistically sound way. If so, return the better denoised version
//this part is not optimized because it isn't used in the demo, but it is correct
//you can check that by turning on the demo_mode: algo_testing_mode = false
void flat_SURE(	
		VectorXd ** noisy_patches
,		MatrixXi const & patch_map
,		std::vector<Vector2i> ** helper_coordinates
,		int image_rows
,		int image_cols
,		int num_models
,		int overlap 
,		double sigma
,		VectorXd **& restored_patches  
){
	int map_rows = patch_map.rows();
	int map_cols = patch_map.cols();
	int data_size = restored_patches[0][0].rows();
	int patch_size = sqrt( data_size );
	double squared_sigma = sigma * sigma;
	//first have the noisy image and first denoised version
	MatrixXd noisy_image( image_rows, image_cols ), restored_image( image_rows, image_cols ); 
	patches2image( noisy_patches, overlap, patch_size, noisy_image );
	patches2image( restored_patches, overlap, patch_size, restored_image );
	//next identify the flat region 
	//for that, allocate memory	
	VectorXd ** flat_marker = new VectorXd * [map_rows];
	for( int row = 0; row < map_rows; row++ )
		flat_marker[row] = new VectorXd [map_cols];
	int flat_model = num_models - 1;
	for( int row = 0; row < map_rows; row++ )
		for( int col = 0; col < map_cols; col++ )
			if( patch_map( row, col ) == flat_model )	
				flat_marker[row][col].setOnes(data_size);
			else
				flat_marker[row][col].setZero(data_size);
	MatrixXd flat_indicator( image_rows, image_cols );
	//I want those pixels fully belonging to 64 flat patches, not less 
	patches2image( flat_marker, overlap, patch_size, flat_indicator, false );
	//release
	for( int row = 0; row < map_rows; row++ )
		delete [] flat_marker[row];
	delete [] flat_marker;
	//proceed to calculate their SURE
	double actual_MSE = 0;
	double num_flat_pixels = 0;
	//compute SURE: due to the choice made on the flat pixels, the total correction factor is exactly num_flat_pixels/data_size
	//#pragma omp parallel for schedule( static ) reduction( + : actual_MSE, num_flat_pixels )
	for( int row = 0; row < image_rows; row++ )
		for( int col = 0; col < image_cols; col++ )
			if( (int)flat_indicator( row, col ) == data_size ){
				flat_indicator( row, col ) = 1;
				actual_MSE += pow( noisy_image( row, col ) - restored_image( row, col ), 2 ); 
				num_flat_pixels += 1;
			}else
				flat_indicator( row, col ) = 0;
	//for testing purpose: so you can see what are the so-called flat pixels
	if( true ){
		MatrixXd * container = new MatrixXd [1];
		container[0] = flat_indicator * 255;
		imwrite( "flat_pixels.png", container, 1 );
	}
	//SURE of these flat pixels without patch expansion
	double SURE = actual_MSE/num_flat_pixels - squared_sigma + 2*squared_sigma/data_size;
	cout << "INFO: Patch Expansion is based on gauging " << num_flat_pixels << " pixels." << endl; 
	//expand patches to see what happens
	//the job is obviously to calculate the trace of the linear denoising operator
	//num_estimates records how many estimates each flat pixel has
	//since all the pixels in a patch undergo the same operation, the container's dimension is justified
	MatrixXd num_estimates, self_weight, denoised_averages;
	num_estimates.setZero( map_rows, map_cols );
	//the weight of each pixel at denoising itself
	self_weight.setZero( map_rows, map_cols );
	//denoised_average records noisy pixel average
	denoised_averages.setZero( map_rows, map_cols );
	//#pragma omp parallel for schedule( static )
	for( int row = 0; row < map_rows; row++ )
		for( int col = 0; col < map_cols; col++ ){
			int num_helpers = helper_coordinates[ row ][ col ].size();
			if( num_helpers == 0 )
				continue;
			//ok, a flat patch, first denoise
			double denoised = 0;
			for( int m = 0; m < num_helpers; m++ ){
				Vector2i coordinates = helper_coordinates[row][col][m];
				int row_id = coordinates(0);
				int col_id = coordinates(1);
				denoised += noisy_patches[row_id][col_id].sum();
			}
			//self weight received because of this round of noisy pixel average
			double weight = 1./(num_helpers*data_size);
			denoised *= weight;
			//record the statistics
			for( int m = 0; m < num_helpers; m++ ){
				Vector2i coordinates = helper_coordinates[row][col][m];
				int row_id = coordinates(0);
				int col_id = coordinates(1);
				#pragma omp critical
				{
					denoised_averages( row_id, col_id ) += denoised;
					num_estimates( row_id, col_id ) += 1;
					self_weight( row_id, col_id ) += weight;
				}
			}
		}
	//translate that information into images
	VectorXd ** Mself_weights = new VectorXd * [ map_rows ];
	VectorXd ** Mnum_estimates = new VectorXd * [ map_rows ];
	VectorXd ** Mdenoised_averages = new VectorXd * [ map_rows ];
	for( int row = 0; row < map_rows; row++ ){
		Mself_weights[ row ] = new VectorXd [ map_cols ];
		Mnum_estimates[ row ] = new VectorXd [ map_cols ];
		Mdenoised_averages[ row ] = new VectorXd [ map_cols ];
	}
	//I learned, the hard way, that memory assignment shall never be parallelized
	for( int row = 0; row < map_rows; row++ )
		for( int col = 0; col < map_cols; col++ )
			if( helper_coordinates[ row ][ col ].size() > 0 ){
				Mdenoised_averages[row][col].setConstant(data_size, denoised_averages(row, col)/num_estimates(row, col));
				Mnum_estimates[row][col].setConstant(data_size, num_estimates(row, col));
				Mself_weights[row][col].setConstant(data_size, self_weight(row, col));
			}else{
				Mdenoised_averages[row][col].setZero(data_size);
				Mnum_estimates[row][col].setZero(data_size);
				Mself_weights[row][col].setZero(data_size);
			}
	//Mdenoised_averages records for all chosen flat pixels their expanded filtered values
	MatrixXd Iself_weights( image_rows, image_cols ), Inum_estimates( image_rows, image_cols ), Idenoised_averages( image_rows, image_cols );
	patches2image( Mdenoised_averages, overlap, patch_size, Idenoised_averages );
	//the other two matrices have what it takes to calculate the new SURE value
	patches2image( Mself_weights, overlap, patch_size, Iself_weights, false );
	patches2image( Mnum_estimates, overlap, patch_size, Inum_estimates, false );
	double new_MSE = 0;
	double new_correction = 0; 
	//#pragma omp parallel for schedule( static ) reduction( +: new_MSE, new_correction )
	for( int row = 0; row < image_rows; row++ )	
		for( int col = 0; col < image_cols; col++ )
			if( (int)flat_indicator( row, col ) ==  1 ){
				new_MSE += pow( noisy_image( row, col ) - Idenoised_averages( row, col ), 2 );
				new_correction += Iself_weights( row, col )/Inum_estimates( row, col );	
			}
	double new_SURE = new_MSE/num_flat_pixels - squared_sigma + 2 * squared_sigma /num_flat_pixels * new_correction;
	cout << "INFO: plain SURE : " << SURE << " and expanded SURE : " << new_SURE << endl;
	if( SURE > new_SURE ){
		cout << "INFO: render the flat region aggressively." << endl;
		#pragma omp parallel for schedule( static )
		for( int row = 0; row < map_rows; row++ )
			for( int col = 0; col < map_cols; col++ )
				if( helper_coordinates[ row ][ col ].size() )
					restored_patches[ row ][ col ] = Mdenoised_averages[row][col];
	}
	//release
	for( int row = 0; row < map_rows; row++ ){
		delete [] Mdenoised_averages[row];
		delete [] Mself_weights[row];
		delete [] Mnum_estimates[row]; 
	}
	delete [] Mdenoised_averages;
	delete [] Mself_weights;
	delete [] Mnum_estimates; 
}

//update the probabilistic PCA mixture with the two stage GEM
void update_PPCA(	
		VectorXd ** noisy_patches
,		VectorXd ** responsibility
,		VectorXd * prior
,		int num_models
,		int map_rows
,		int map_cols
,		double sigma
,		VectorXd *& mus
,		MatrixXd *& factors
){
	cout << "INFO: updating the mixture ..." << endl;
	int data_size = noisy_patches[0][0].rows();


//	update the prior probability and the model expectations at the same time
//	updating rule is very very intuitive, basically it says that given a mixture model
//	you can never be sure which model a patch belongs to, no matter where you find it
//	because a Gaussian multivariate distribution has an infinite support 

	MatrixXd updated_means;
	updated_means.setZero(data_size, num_models);
	VectorXd total_weights;
	total_weights.setZero(num_models);
	for( int row = 0; row < map_rows; row++ ){
		MatrixXd patch_matrix( data_size, map_cols );
		MatrixXd resp_matrix( num_models, map_cols );
		vectorArray2Matrix( noisy_patches, row, patch_matrix );
		vectorArray2Matrix( responsibility, row, resp_matrix );
		updated_means += patch_matrix * resp_matrix.transpose();
		total_weights += resp_matrix.rowwise().sum();	
	}
	updated_means *= total_weights.cwiseInverse().asDiagonal();
	double num_patches = map_rows * map_cols;
	prior[0] = total_weights/num_patches;


//	now update the factors, memory allocation first
//	OmegaF refers to the product of Omega (estimated model covariance matrix up to a multiplicative constant) and Factors (see doc)
	MatrixXd * OmegaFs = new MatrixXd [num_models];
	MatrixXd * centers = new MatrixXd [num_models];
	int std_num_factors = factors[0].cols();
//	#pragma omp parallel for
	for( int model = 0; model < num_models; model++ ){
		mus[model] = updated_means.block(0, model, data_size, 1);
		centers[model] = mus[model].rowwise().replicate(map_cols);
		int num_factors = ret_num_factors( std_num_factors, data_size, model, num_models );
		OmegaFs[model].setZero(data_size, num_factors);
	}
//	continue. first calculate OmegaFs 
//	it is the next chunk that consumes the most time in this routine
	#pragma omp parallel for schedule( static ) 
	for( int row = 0; row < map_rows; row++ ){
		MatrixXd patch_matrix( data_size, map_cols );
		MatrixXd resp_matrix( num_models, map_cols );
		vectorArray2Matrix( noisy_patches, row, patch_matrix );
		vectorArray2Matrix( responsibility, row, resp_matrix );
		for( int model = 0; model < num_models; model++ ){
			MatrixXd centered = patch_matrix - centers[model];
			MatrixXd tmp = centered*(resp_matrix.row(model).asDiagonal() * (factors[model].transpose() * centered).transpose());		
			#pragma omp critical
			{
				OmegaFs[model] += tmp;
			}
		}
	}
//	release
	delete [] centers;


//	the final two linear equations to solve
	double squared_sigma = sigma * sigma;	
	#pragma omp parallel for schedule( static ) 
	for( int model = 0; model < num_models; model++ ){
//		to understand the rest, please refer to the doc
		int num_factors = ret_num_factors( std_num_factors, data_size, model, num_models );
		MatrixXd identity;
		identity.setIdentity(num_factors, num_factors);
		MatrixXd M = factors[model].transpose()*factors[model] + identity*squared_sigma;
//		note that OmegaFs haven't been normalized
		double normalizer = squared_sigma * prior[0](model) * num_patches;
		MatrixXd intermediate = M.ldlt().solve(factors[model].transpose())*OmegaFs[model] + identity*normalizer;
		factors[model] = intermediate.transpose().colPivHouseholderQr().solve(OmegaFs[model].transpose()).transpose();
	}
//	release
	delete [] OmegaFs;
}


//main_routine adds noise and stores the clean image for comparison purpose
//to have the best performance, turns on both_filters = true
//otherwise, both_filters = false only_Wiener = true: only Wiener filtering
//both_filters = false only_Wiener = false: only Donoho filtering
int SPLE_main_routine( 	
			const char * input
,			const char * output 
,			double sigma
,			int num_iterations
,			int overlap 	
,			int patch_size  
,			int num_orientations 
,			double std_num_factors 
,			bool only_Wiener 
,	       		bool both_filters 
,			bool Testing_Mode  //output_after_each_iteration  
){     
	int num_channels, image_rows, image_cols;
	MatrixXd * clean_image = NULL;
	imread( input, image_rows, image_cols, num_channels, clean_image );
	if( clean_image == NULL )
		fail("Unable to get the image");

	//compare true MSE and SURE
	//I stick with MSE rather than RMSE because SURE can turn negative	
	bool compare = true; 
	if( compare )
		cout << "INFO: true MSE will be printed out along the way" << endl;
	else
		cout << "INFO: turn on compare to see how RMSE behaves at each iteration" << endl;

	MatrixXd * clean_copy = NULL;
	if( compare ){
		clean_copy = new MatrixXd[num_channels];
		for( int ch = 0; ch < num_channels; ch ++ )
			clean_copy[ch] = clean_image[ch];
	}

	//add noise and write out a noisy version
	add_gaussian_noise( clean_image, sigma, num_channels );
	MatrixXd * noisy = new MatrixXd [num_channels];
	for( int ch = 0; ch < num_channels; ch++ )
		noisy[ch] = clean_image[ch];
	//clean image's no longer there
	imwrite( "noisy.png", clean_image, num_channels );


	//get down to the business
	MatrixXd * denoised = SPLE_sub_routine( clean_copy, noisy, num_channels, sigma, patch_size, overlap, num_orientations, std_num_factors, num_iterations, only_Wiener, both_filters, compare, output, Testing_Mode ); 
	imwrite( output, denoised, num_channels );
	//PLE_sub_routine( clean_copy, noisy, num_channels, sigma, patch_size, overlap, num_orientations, num_iterations, output, compare );

	return EXIT_SUCCESS;
}



//SPLE routine
MatrixXd * SPLE_sub_routine( 
			MatrixXd * clean
,			MatrixXd * noisy
,			int num_channels 
,			double noise_sigma
,			int patch_size
,			int overlap
,			int num_orientations
,			int std_num_factors
,			int num_iterations 
,			bool only_Wiener
,			bool both_filters 
,			bool compare
,			char const * output
,			bool Testing_Mode
){ 
//	improve SNR on the first transformed channel
	RGB_transform( noisy, num_channels, false );
	if( compare )
		RGB_transform( clean, num_channels, false );
	
	
	cout << "INFO: SPLE with " << num_orientations << " orientations + 2 will iterate " << num_iterations << " times." << endl;
	if( Testing_Mode )
		cout << "INFO: I'm in the testing mode and will write out all interesting stuff. " << endl;
	else
		cout << "INFO: I'm in the demo mode and will only write out the final output and its patch map." << endl;
	
//	the parameters in the GFMM setup for SPLE
	VectorXd * prior = NULL;
	MatrixXd * factors = NULL;
	VectorXd * mus = NULL;
	VectorXd ** responsibility = NULL;
// 	two additional models to account for textural (multi-oriented) and flat patches
	int num_models = num_orientations + 2;
	int data_size = patch_size * patch_size;
//	setup GFMM	
	read_config( data_size, std_num_factors, num_models, prior, mus, factors );
//	history records for each iteration the denoised image, in testing mode
	MatrixXd ** history = new MatrixXd * [num_channels];
	int image_rows = noisy[0].rows();
	int image_cols = noisy[0].cols();
	int map_rows = num_patches( image_rows, patch_size, overlap );
	int map_cols = num_patches( image_cols, patch_size, overlap );
//	patch_map takes integer values
	MatrixXi patch_map( map_rows, map_cols );
//	SPLE starts: I am so excited!
	for( int ch = 0; ch < num_channels; ch++ ){
		cout << endl << "INFO: CH " << ch << endl;
		double sigma = noise_sigma;
//		I prefer to normalize signal to noise for it is better described by the collected prior
//		the sigma here reflects the lumninance-chrominance transformation in RGB_transform() 
		if( ch == 0 )
			sigma /= sqrt((double)num_channels);
		
		if(compare){
			cout << "INFO: Initial True MSE at CH " << ch << ": ";
			show_image_MSE( clean[ch], noisy[ch] );
		}

		//these two parameters help track the filter's real time performance
		//the intention was to stop the whole thing if SURE goes up for the first time
		//certainly you can fiddle with them to make a rule of you own
		//here the algo always presses on regardless because 
		//SURE is liable to go back if allowed enough iterations
		double fmr_best_SURE, SURE_now;

		//raw data and container
		history[ch] = new MatrixXd [num_iterations];
		VectorXd ** noisy_patches = image2patches( noisy[ch], image_rows, image_cols, overlap, patch_size );

		//CAREFUL: this part should be carried out sequentially 
		for( int iter = 0; iter < num_iterations; iter++ ){
			//only perform EM on the transformed channel with the highest SNR
			if( ch == 0 ){	
				times("");
				calc_responsibilities( noisy_patches, factors, mus, prior, map_rows, map_cols, data_size, std_num_factors, sigma, responsibility, patch_map );		
				times("RESPONSIBILITY: ");
			}

			//tolerate some randomness and make SURE slightly higher than MSE initially 
			if( iter == 0 )
				fmr_best_SURE = pow( sigma + min( 1., sigma/4. ), 2 );

			//filter only if necessary
			//Testing_Mode on: filter at each iteration 
			//Testing_Mode off: filter at the last iteration 
			bool apply_filter = Testing_Mode || (iter == num_iterations - 1);

			VectorXd ** restored_patches = NULL;
			//model_MSE tracks the model-wide MSE and can be made as a vector
			//though for testing purpose, let's leave it in a matrix
			MatrixXd model_MSE(num_models, 2);

			//set Testing_Mode to see how SURE behaves over time
			if( apply_filter ){
				times("");
				filter_patches( noisy_patches, factors, mus, responsibility, patch_map, image_rows, image_cols, overlap, ch, sigma, only_Wiener, both_filters, model_MSE, SURE_now, restored_patches, Testing_Mode );
				times("FILTER: ");
	
				cout << "INFO: here comes the patch map" << endl;
				print_patch_map( patch_map );

				//let's decide based on SURE whether to accept these restored patches
				//see, SURE really guides SPLE
				//I could compare the previous filtered patches with the current ones using the model-wide SURE
				//but for simplicity, do it in an easy way 
				if( SURE_now <= fmr_best_SURE ){
					fmr_best_SURE = SURE_now;
					MatrixXd filtered( image_rows, image_cols );
					patches2image(restored_patches, overlap, patch_size, filtered);
					history[ch][iter] = filtered;
					if(compare){
						cout << endl << "INFO: Iteration " << iter << " : "; 
						show_image_MSE( clean[ch], filtered );
					}
				}else{
					//maybe you can do some more thoughtful stuff here
					cout << "INFO: Iteration " << iter << " is lost according to SURE : (" << endl;
					if( Testing_Mode ){
						//let's see whether we shall trust SURE, if it does not work out the right way
						//stopping at the first increase in SURE might not be a wise thing to do
						MatrixXd filtered( image_rows, image_cols );
						patches2image(restored_patches, overlap, patch_size, filtered);
						cout << "INFO: Check the real thing: ";
						show_image_MSE( clean[ch], filtered );
					}
					//only under Testing_Mode, otherwise, there's no previous version to fall back on
					if(iter > 0 && Testing_Mode)
						history[ch][iter] = history[ch][iter-1];
					else
						history[ch][iter] = noisy[ch];
				}
				cout << fixed << setprecision(5) << "INFO: SURE at this channel : "  << SURE_now << endl;
			}

			//update the mixture parameters	
			if( iter < num_iterations - 1 && ch == 0 ){
				times("");
				update_PPCA( noisy_patches, responsibility, prior, num_models, map_rows, map_cols, sigma, mus, factors );
				times("UPDATE: ");
			}
			
			//release the container if used
			if( apply_filter ){
				for( int row = 0; row < map_rows; row++ )
					delete [] restored_patches[ row ];
				delete [] restored_patches;
			}
		}

		//this channel has been processed
		for( int row = 0; row < map_rows; row++ )
			delete [] noisy_patches[row];
		delete [] noisy_patches;
	}

//	over! the image has been denoised		
	delete [] factors;
	delete [] mus;
	delete [] prior;
	for( int row = 0; row < map_rows; row++ )
		delete [] responsibility[row];
	delete [] responsibility;


	//write out the images with the required names
	MatrixXd * denoised_image_ptr = NULL;
	int start = 0;
	if(!Testing_Mode)
		start = num_iterations - 1;
	for(int iter = start; iter < num_iterations; iter++){
		denoised_image_ptr = new MatrixXd [num_channels];
		for( int ch = 0; ch < num_channels; ch++ )
			denoised_image_ptr[ch] = history[ch][iter];		
		RGB_transform( denoised_image_ptr, num_channels, true );
		if( Testing_Mode && iter < num_iterations - 1 ){	
			int num_words = (int)strlen(output);
			char image_name[ num_words-3 ];
			strncpy( image_name, output, num_words-4 );
			image_name[ num_words-4 ] = '\0';
			char buffer[100];
			sprintf( buffer, "%s_%02d.png", image_name, iter );	
			const char * fname = buffer;
			imwrite( fname, denoised_image_ptr, num_channels );
		}
	}

	//release
	for(int ch = 0; ch < num_channels; ch++)
		delete [] history[ch];
	delete [] history;	
	delete [] noisy;
	if( compare )
		delete [] clean;

	//return
	return denoised_image_ptr;
}



