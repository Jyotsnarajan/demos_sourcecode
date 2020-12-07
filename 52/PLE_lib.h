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
 * @file PLE_lib.h
 * @brief header file for PLE denoising algorithm
 * @author Yi-Qing WANG <yiqing.wang@polytechnique.edu>
 */

/**
 * @brief blur an image with a Gaussian kernel 
 *
 * @param image sharp image 
 * @param kernel blurring kernel 
 * @param result blurred image 
 */

#ifndef VECTOR_H
#define VECTOR_H
#include <vector>		
#endif 

#ifndef EIGEN_H
#define EIGEN_H
#include <Eigen/Dense>  	// Eigen
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

using namespace Eigen;

void convolute( 
		NMatrix const & image
,		NMatrix const & kernel
,		NMatrix & result 
);

/**
 * @brief create a Gaussian blurring kernel 
 *
 * @param sigma Gaussian kernel standard deviation 
 * @param kernel returned blurring kernel 
 */
void Gaussian_kernel( 
			double sigma
,			NMatrix & kernel 
);

/**
 * @brief Gaussian density for blurring kernels
 *
 * @param x observation 
 * @param mu Gaussian mean 
 * @param cov Gaussian covariance 
 */
double gaussian_density(	
		NColV const & x
,		NColV const & mu
,		NSym const & cov
);

/**
 * @brief create feature oriented synthetic images 
 *
 * @param theta feature orientation 
 * @param image returned synthetic images 
 */
void get_synthesized_image( 	
			double theta
,			NMatrix & image
);

/**
 * @brief sample patches from synthetic images 
 *
 * @param theta feature orientation 
 * @param n_samples number of samples 
 * @param patch_size patch dimension = patch_size * patch_size 
 * @param result return the vectorized patch samples 
 */
void get_synthesized_patches( 	
				double theta
,				int n_samples
,		 		int patch_size
,		 		NMatrix & result
);	

/**
 * @brief Principal Component Analysis by SVD 
 *
 * @param theta feature orientation 
 * @param n_samples number of samples 
 * @param patch_size patch dimension = patch_size * patch_size 
 * @param U return the oriented basis by SVD
 */
void PCA( 	
		double theta
,		int n_samples
, 		int patch_size
,		NMatrix & U 
);

/** @brief return DCT in output */
void get_DCT_basis( NMatrix & output );

/** @brief a wrapper of patch sampling routines */
NSym * initialize_covs( 	
		int num_models
,		int patch_size
);

/**
 * @brief main routine
 *
 * @param input clean image name 
 * @param output denoised image name 
 * @param sigma the standard deviation of corrupting noise 
 * @param num_iterations how many times SPLE or PLE shall iterate 
 * @param overlap the neighboring patch overlap used by the filtering scheme to draw patches
 * @param patch_size patch size 
 * @param num_orientations how many oriented components the mixture should include 
 */
int PLE_main_routine( 	
			const char * input
,			const char * output 
,			double sigma
,			int num_iterations
,			int overlap 
,			int patch_size 
,			int num_orientations 
);     
/**
 * @brief a wrapper for PLE_denoise 
 *
 * @param clean clean image 
 * @param noisy noisy image 
 * @param num_channels RGB or Gray
 * @param noise_sigma the standard deviation of corrupting noise 
 * @param patch_size patch size 
 * @param overlap the neighboring patch overlap used by the filtering scheme to draw patches
 * @param num_orientations how many oriented components the mixture should include 
 * @param num_iterations how many times SPLE or PLE shall iterate 
 * @param output denoised image name 
 * @param compare whether to see the difference between the denoised images and the clean one in MSE
 */
void PLE_sub_routine(
			MatrixXd * clean
,			MatrixXd * noisy
,			int num_channels 
,			double noise_sigma
,			int patch_size
,			int overlap
,			int num_orientations
,			int num_iterations 
,			char const * output
,			bool compare
);

/**
 * @brief PLE (Yu et al. 2012) 
 *
 * @param noisy noisy image 
 * @param clean clean image 
 * @param num_iterations how many times SPLE or PLE shall iterate 
 * @param overlap the neighboring patch overlap used by the filtering scheme to draw patches
 * @param patch_size patch size 
 * @param sigma the standard deviation of corrupting noise 
 * @param num_models the number of oriented models plus one DCT 
 * @param covs initialized model covariances drawn from synthetic images 
 * @param mus initialized model means (all set to zero)
 * @param compare whether to see the difference between the denoised images and the clean one in MSE
 * @return denoised images from all the PLE iterations
 */
MatrixXd * PLE_denoise( 	
		MatrixXd & noisy
,		MatrixXd & clean
,		int num_iterations
,		int overlap
,		int patch_size
,		double sigma
,		int num_models
,		NSym * covs
,		NColV * mus
,		bool compare 
);

/**
 * @brief PLE clustering 
 *
 * @param mus model means 
 * @param covs model covariances
 * @param noisy_patches noisy patches
 * @param cluster_map how are the patches are clustered as an integer matrix
 * @param sigma the standard deviation of corrupting noise 
 * @param num_models the number of oriented models plus one DCT 
 * @param restored_patches patches restored along with clustering 
 */
void make_clusters( 	
		NColV const * mus
,		NSym const * covs
,		NMatrix const * noisy_patches
,		MatrixXi & cluster_map
,		double sigma
,		int num_models
,		NMatrix *& restored_patches 
);

/**
 * @brief compare patch likelihoods under different model assumptions 
 *
 * @param mus model means 
 * @param cov_determinant determinants of model covariances  
 * @param inv_cov inverted model covariances 
 * @param Wiener_shrinkage Wiener ratios S/(S+N) 
 * @param sigma the standard deviation of corrupting noise 
 * @param num_models the number of oriented models plus one DCT 
 * @param observation observed noisy patch 
 * @param guessed_signal estimated patch by likelihood comparison 
 */
int most_probable_generator( 	
		NColV const * mus
,		double const * cov_determinant
,		NSym const * inv_cov
,		NSym const * Wiener_shrinkage
,		double sigma 
,		int num_models
,		NColV const & observation
,		NColV & guessed_signal 
);

/**
 * @brief Wiener filtering 
 *
 * @param mu model mean 
 * @param cov_determinant the determinant of the model covariance
 * @param inv_cov inverted model covariance matrix 
 * @param shrinkage model Wiener ratios S/(S+N) 
 * @param sigma the standard deviation of corrupting noise 
 * @param observation observed noisy patch 
 * @param best_guess Wiener filtered patch 
 */
double guess_signal( 	
		NColV const & mu
,		double cov_determinant
,		NSym const & inv_cov
,		NSym const & shrinkage
,		double sigma
,		NColV const & observation
,		NColV & best_guess 
);

/**
 * @brief update the cluster parameters 
 *
 * @param cluster_map how are the patches are clustered as an integer matrix
 * @param restored_patches patches restored along with clustering 
 * @param num_models the number of oriented models plus one DCT 
 * @param mus updated model means 
 * @param covs updated model covariances
 */
void update_parameters(	
		MatrixXi const & cluster_map
,		NMatrix const * restored_patches_at_row
,		int num_models
,		NColV *& mus
,		NSym *& covs 
);

/** @brief an interface between newmat and eigen library */
void convertFormat( 
			VectorXd **& eigen
,			NMatrix *& newmat
,			int map_rows 
,			int map_cols
,			int data_size
,			bool inverse
);
