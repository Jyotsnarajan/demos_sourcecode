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
 * @file DataProvider.h
 * @brief class definition for handling various mixture parameters
 * @author Yi-Qing WANG <yiqing.wang@polytechnique.edu>
 */
class DataProvider
{
    	static double prior_08[];

	static double mu_08[][64];

	static double cov_08[][4096];
	
	static double colormap[];
public:	
	DataProvider(){}
	~DataProvider(){}

	 double *  GetColormap(){
		return colormap;
	}
	 double *  GetPrior( int patch_side ){
      			return prior_08;
	}
	 double *  GetMu( int patch_side, int model ){
      			return mu_08[model];
	}
	 double *  GetCov( int patch_side, int model ){
      			return cov_08[model];
	}
};
