/*
 * Copyright (C) 2014 Jacques Froment <Jacques.Froment@univ-ubs.fr>
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
 * @file addgaussnoise.c
 * @brief Add a gaussian noise to an image using Mersenne Twister pseudo-RNG 
 *        code (see mt19937ar.c file)
 *
 * @author Jacques Froment <Jacques.Froment@univ-ubs.fr>
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "io_png.h"
#include "libauxiliary.h"

/**
 * @brief Main function
 * WARNING : if calling addgaussnoise from the executable (command) rather from 
 *           the function library, input/output are managed using the PNG file
 *           format that DOES perform a thresholding outside [0,255].
 *           A a result of, the effective standard deviation of the noise in
 *           the noised image may be lower that the one asked for.
*/

int main(int argc, char **argv)
{
  
  size_t M,N;     /* M=number of rows (dy) and N=number of columns (dx) */
  size_t NC;      /* Number of channels */
  size_t MNC;     /* Total size of the image: MxNxNC */
  float *U=NULL;  /* Input image */
  float *V=NULL;  /* Output image (noisy U) */
  unsigned long s;/* Seed for the generator */
  double sigma;   /* Standard deviation of the gaussian noise */
  int optind;     /* Index to the first argument in the command line */

  if ( ((argc!=4)&&(argc!=6)) || ((argc==6)&&(strcmp(argv[1],"-s")!=0)) ) {
    fprintf(stderr,
	    "Usage: addgaussnoise [-s seed] <image_in> <sigma> <image_out>\n");
    return EXIT_FAILURE;
  }

  if (argc==6)
    /* -s seed option has been selected */
    {
      optind=3;
      /* Fixed seed allows to get same noise among all calls */
      s=atol(argv[2]);
    }
  else
    /* Default */
    {
      optind=1;
      /* By default noise should be different among all calls */
      s=time(NULL) + getpid();
    }

  /* Read image_in U */
  U = io_png_read_f32(argv[optind], &N, &M, &NC);
  if (!U) 
    {
      fprintf(stderr, "Error: <image_in> '%s' not found \
or not a correct png image \n", 
	      argv[optind]);
      exit(EXIT_FAILURE);       
    }

  sigma=atof(argv[optind+1]);
  if (sigma <= 0)
    {
      fprintf(stderr,"Error: invalid sigma=%f value\n",sigma);
      exit(EXIT_FAILURE);       
    }

  /* Memory allocation of image_in V */
  MNC=M*N*NC;
  V =(float *)malloc(MNC*sizeof(float)); 
  if (!V) 
    {
      fprintf(stderr,"Not enough memory (V)\n");
      exit(EXIT_FAILURE);       
    }

  /* Compute V = U + noise */
  addgaussnoise(U, V, sigma, s, MNC);

  /* Write output image in PNG format */
  if (io_png_write_f32(argv[optind+2], V, N, M, NC) != 0)
    {
      fprintf(stderr,"Couldn't write output image '%s'\n",argv[optind+2]);
      exit(EXIT_FAILURE);       
    }    

  /* Free the memory and exit with success */
  free(V);
  free(U);
  exit(EXIT_SUCCESS);
}
