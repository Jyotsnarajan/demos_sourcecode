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
 * @file SPLE_lib.h
 * @brief header file for SPLE denoising algorithm
 * @brief SPLE is implemented mainly with Eigen (Newmat for SVD) 
 * @brief PLE is implemented mainly with Newmat 
 * @author Yi-Qing WANG <yiqing.wang@polytechnique.edu>
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


/**
 * @brief return the number of factors in each mixture component 
 *
 * @param std_num_factors number of factors in an oriented model 
 * @param data_size the number of pixels in a patch 
 * @param model the current model 
 * @param num_models the number of mixture components 
 */
int ret_num_factors(
		int std_num_factors
,		int data_size
, 		int model
,		int num_models
);

/**
 * @brief read in the initial GFM setup 
 *
 * @param prior the model priors (mixing weights) 
 * @param mus model means 
 * @param factors model factors
 */
void read_config(	
		int data_size
,		int std_num_factors
,		int num_models
,		VectorXd *& prior
,		VectorXd *& mus
,		MatrixXd *& factors
);

/**
 * @brief a multivariate Gaussian density value
 *
 * @param x the observation 
 * @param mu the Gaussian model mean
 * @param inv_eig_vals the inverse of the eigenvalues of the sum of factor + noise variance 
 * @param eig_vecs the principal components of the subspace spanned by the factors 
 * @param squared_sigma squared sigma
 * @param num_factors number of factors in the current model
 * @return density up to an absolute constant 
 */
double gaussian_density_EM( 
			VectorXd const & x
,			VectorXd const & mu
, 			VectorXd const & inv_eig_vals
,			MatrixXd const & eig_vecs
,			double squared_sigma
,			int num_factors 
);

/**
 * @brief calculate the posterior probability for a patch to belong to a mixture component 
 *
 * @param noisy_patches_at_coordinates noisy patch array 
 * @param factors mixture component factors 
 * @param mus model means 
 * @param prior model priors 
 * @param map_rows the number of patches covering the image by row 
 * @param map_cols the number of patches covering the image by column 
 * @param data_size the number of pixels in a patch 
 * @param std_num_factors the number of factors in an oriented model 
 * @param sigma the standard deviation of corrupting noise 
 * @param responsibility the posterior probability for each patch in the array 
 * @param patch_map a patch to model mapping in the form of an integer matrix 
 */
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
);

/**
 * @brief a wrapper of SVD implemented in NEWMAT 
 *
 * @param cov_mat the matrix to SVD 
 * @param std_num_factors the number of factors contained in an oriented mixture component 
 * @param data_size the number of pixels in a patch 
 * @param model the current component 
 * @param num_models the number of mixture components 
 * @param squared_sigma variance of corrupting noise 
 * @param inverse_eig_vals whether to compute the inverse of eigenvalues for Wiener filtering for instance
 * @param trim whether to remove the factors attrbuted to noise rather than signal 
 * @param eig_vecs returned eigenvectors of cov_mat 
 * @param eig_vecs returned eigenvalues of cov_mat
 */
void deduce_eigens(
		MatrixXd & cov_mat 
,		int std_num_factors
,		int data_size
,		int model
,		int num_models
,		double squared_sigma
,		bool inverse_eig_vals
,		bool trim
,		MatrixXd & eig_vecs
,		VectorXd & eig_vals
);

/**
 * @brief filter noisy patches with Donoho Minimax, Wiener L2 or both 
 *
 * @param noisy_patches noisy patch array 
 * @param factors the mixture component factors 
 * @param mus the model means
 * @param responsibility the posterior probability for each noisy patch 
 * @param patch_map the patch map 
 * @param image_rows the number of pixels in each image column 
 * @param image_cols the number of pixels in each image row
 * @param overlap the neighboring patch overlap used by the filtering scheme to draw patches 
 * @param ch which channel am I looking at 
 * @param sigma the standard deviation of corrupting noise 
 * @param only_Wiener apply only Wiener filter 
 * @param both_filters apply both Donoho and Wiener 
 * @param model_MSE the estimated MSE of each mixture component
 * @param SURE_mean a filter performance tracker as well as image MSE asymptotitc upper bound 
 * @param restored_patches filtered patches
 * @param Testing_Mode whether I'm in Testing_Mode or Demo_Mode 
 */
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
);

/**
 * @brief Wiener filtering 
 *
 * @param noisy_patch noisy patch
 * @param proba_weight the conditional posterior probability given the noisy patch
 * @param mus the model means
 * @param filter_basis eigenvectors of the mixture components 
 * @param Wiener_shrinkage Wiener filtering ratio 
 * @param num_models the number of mixture components 
 * @param data_size the number of pixels in a patch 
 * @param estimated_patch returned Wiener filtered patch 
 */
void Wiener_Conditional_Expectation( 	
		VectorXd const & noisy_patch
,		VectorXd const & proba_weight
,		VectorXd * mus
,		MatrixXd * filter_basis
,		VectorXd * Wiener_shrinkage
,		int num_models
,		int data_size
,		VectorXd & estimated_patch
);

/**
 * @brief a wrapper of both Wiener and Donoho filter 
 *
 * @param noisy_patches noisy patch array 
 * @param factors the mixture component factors 
 * @param mus the model means
 * @param responsibility the posterior probability for each noisy patch 
 * @param patch_map the patch map 
 * @param model_MSE the estimated MSE of each mixture component
 * @param ch which channel am I looking at 
 * @param sigma the standard deviation of corrupting noise 
 * @param is_Wiener apply Wiener filter or Donoho hard shrinkage 
 * @param restored_patches filtered patches
 */
void filter(	
		VectorXd ** noisy_patches 
,		MatrixXd * factors
,		VectorXd * mus
,		VectorXd ** responsibility
,		MatrixXi const & patch_map
,		MatrixXd & model_MSE 
,		int ch
,		double sigma
,		bool is_Wiener 	    
,		VectorXd **& restored_patches
);

/**
 * @brief a device that makes the sky look bluer
 *
 * @param noisy_patches noisy patch array 
 * @param patch_map the patch map
 * @param num_models the number of components in the mixture 
 * @param sigma the standard deviation of corrupting noise 
 * @param overlap the neighboring patch overlap used by the filtering scheme to draw patches 
 * @param image_rows the number of pixels in each image column 
 * @param image_cols the number of pixels in each image row
 * @param restored_patches filtered patches
 * @param Testing_Mode whether I'm in Testing_Mode or Demo_Mode 
 */
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
);

/** @brief chi squared test */
bool test_null( 
	VectorXd const & patch_1
, 	VectorXd const & patch_2
,	double sigma
,	bool = false 
);

/**
 * @brief whether should we expand flat patches? If so, do it
 *
 * @param noisy_patches noisy patch array 
 * @param patch_map the patch map
 * @param helper_coordinates where are similar patches given one flat patch 
 * @param image_rows the number of pixels in each image column 
 * @param image_cols the number of pixels in each image row
 * @param overlap the neighboring patch overlap used by the filtering scheme to draw patches 
 * @param sigma the standard deviation of corrupting noise 
 * @param restored_patches filtered patches
 */
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
);

/**
 * @brief update the mixture model means and factors
 *
 * @param noisy_patches noisy patch array 
 * @param responsibility the posterior probability for each noisy patch 
 * @param prior model priors (or mixing weights)
 * @param num_models number of mixture components 
 * @param map_rows the number of patches to cover the image by row
 * @param map_cols the number of patches to cover the image by column 
 * @param sigma the standard deviation of corrupting noise 
 * @param mus the model means
 * @param factors the mixture component factors 
 */
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
 * @param std_num_factors how many factors an oriented component should have 
 * @param only_Wiener apply only Wiener filter (SPLE) 
 * @param both_filters apply both Donoho and Wiener (SPLE)
 * @param Testing_Mode whether I'm in Testing_Mode or Demo_Mode (SPLE) 
 */
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
,			bool Testing_Mode 
);     

/**
 * @brief SPLE 
 *
 * @param clean clean image 
 * @param noisy noisy image 
 * @param num_channels RGB or Gray
 * @param noise_sigma the standard deviation of corrupting noise 
 * @param patch_size patch size 
 * @param overlap the neighboring patch overlap used by the filtering scheme to draw patches
 * @param num_iterations how many times SPLE or PLE shall iterate 
 * @param num_orientations how many oriented components the mixture should include 
 * @param std_num_factors how many factors an oriented component should have 
 * @param only_Wiener apply only Wiener filter (SPLE) 
 * @param both_filters apply both Donoho and Wiener (SPLE)
 * @param compare whether to see the difference between the denoised images and the clean one in MSE
 * @param output denoised image name 
 * @param Testing_Mode whether I'm in Testing_Mode or Demo_Mode (SPLE) 
 */
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
); 

/**
 * @brief two pass connected component labelling algorithm 
 *
 * @param binary the binary image 
 * @param map_rows the number of pixel rows in the binary image 
 * @param map_cols the number of pixel columns in the binary image
 * @param regions the output of CCL as a pixel stream 
 * @param Testing_Mode whether I'm in Testing_Mode or Demo_Mode (SPLE) 
 */
void connected_components(	
		MatrixXi const & binary
,		int map_rows
,		int map_cols
,		int *& regions
,		bool Testing_Mode
);

