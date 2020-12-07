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
 * @file util.h
 * @brief util header 
 * @author Yi-Qing WANG <yiqing.wang@polytechnique.edu>
 */

#ifndef EIGEN_H
#define EIGEN_H
#include <Eigen/Dense>  	// Eigen
#endif
using namespace Eigen;

/** @brief coordinate of a pixel in an image */
struct SamplingParams{
	int patch_size;
	double flat_threshold;
	double orient_threshold;
	int num_orientations;
	int flat_model;
	int textural_model;
	int minimal_num_samples;
	bool verbose;
	const char * filename;
};

/** @brief exception messaging */
void fail (const char * message );

/** @brief routine timer */
void times (const char * which );

/**
 * @brief read in a PNG and return its dimension 
 *
 * @param file_name the image name
 * @param image the image as a MatrixXd array 
 */
void imread(	
		const char * file_name
,		int & image_rows
,		int & image_cols
, 		int & num_channels
,		MatrixXd *& image		
);

/**
 * @brief write out as a PNG
 *
 * @param file_name the image name
 * @param num_channels RGB or Gray
 */
void imwrite( 
		const char * file_name 
, 		MatrixXd * image
, 		int num_channels
);

/**
 * @brief add i.i.d. Gaussian noise to the image
 *
 * @param sigma noise standard deviation
 * @param num_channels RGB or Gray
 * @param generate_noise read in noise or generate it internally
 */
void add_gaussian_noise( 
		MatrixXd * image
, 		double sigma
,		int num_channels
,		bool generate_noise = true 
);

/**
 * @brief reduce a gray image into patches 
 *
 * @param image a matrix representing one image
 * @param overlap number of columns (resp. rows) two horizontal (resp. vectical) neighbor patches share
 * @param patch_size patch dimension: patch_size * patch_size 
 * @return an array of vectorized patches 
 */
VectorXd ** image2patches( 
		MatrixXd const & image
,		int image_rows
,		int image_cols
,		int overlap
,		int patch_size 
);

/** @brief number of patches required to cover the whole image by row or by column */
int num_patches( 
		int n_pixels
,		int patch_size
,		int overlap
);


/**
 * @brief the inverse of image2patches 
 *
 * @param patch_at_coordinates an array of vectorized patches 
 * @param image the image made of the patches in the patch array 
 * @param normalize whether to average the estimates on each pixel 
 */
void patches2image(     
		VectorXd ** patch_at_coordinates
,		int overlap
, 		int patch_size
,		MatrixXd & image
,		bool normalize = true
);


/**
 * @brief switch between RBG and a slightly different YUV 
 *
 * @param image image array. In case of a gray image, do nothing 
 * @param inverse RGB to YUV or YUV to RGB 
 */
void RGB_transform( 
		MatrixXd * image
,		int num_channels
,		bool inverse 
);


/** @brief compute the MSE of clean - denoised*/ 
void show_image_MSE(
		MatrixXd & clean
,		MatrixXd & denoised
);

/** @brief print out an integer matrix as a color image */ 
void print_patch_map( 
		MatrixXi const & patch_map
,		bool is_patch_map = true
);

/**@brief present the patch array with matrices*/
void vectorArray2Matrix( 
		VectorXd ** varray
,		int row_id
,		MatrixXd & vmatrix
);

/**
 * @brief tensor structure orientation detector 
 *
 * @param image a gray image for patch sampling 
 * @param row row coordinate of the upper left corner of the patch   
 * @param col col coordinate of the upper left corner of the patch
 * @param SamplingParams see its definition
 * @return the model to which the patch should belong
 */
int tensorStructure(
		MatrixXd & image
,		int row
,		int col
,		SamplingParams & params	
);

/**
 * @brief sample patches from a set of gray PNG and output the result 
 *
 * @param patch_size the size of sample patches 
 * @param flat_threshold 
 * @param orient_threshold 
 * @param num_orientations the number of directional models needed
 * @param minimal_num_samples the minimal number of samples required for each model  
 * @return a formatted data file filename
 */
void sample_images(	
		int patch_size
,		double flat_threshold
,		double orient_threshold 
,		int num_orientations
,		int minimal_num_samples	
,		const char * filename
);

/**
 * @brief create a DataProvider.cpp 
 *
 * @param params sampling parameters 
 * @param cov_mats model covariance  
 * @param mean model mean 
 * @param mixing_weights mixing weights 
 */
void write_out_everything( 
			SamplingParams & params
,			MatrixXd * cov_mats
,			VectorXd * mean
,			VectorXd mixing_weights 
);
