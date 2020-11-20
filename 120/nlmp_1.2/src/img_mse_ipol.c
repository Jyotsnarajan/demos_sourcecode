/*
 * Copyright (c) 2009-2011, A. Buades <toni.buades@uib.es>,
 * Copyright (c) 2014 Jacques Froment <Jacques.Froment@univ-ubs.fr>
 * All rights reserved.
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
 * @file img_mse_ipol.c
 * @brief Compute the RMSE and the PSNR of two images in PNG format, 
 *        C89 version.
 *
 * @author Jacques Froment <Jacques.Froment@univ-ubs.fr>, partially based on
 *         the C++ version img_mse_ipol.cpp, author A. Buades
 *         <toni.buades@uib.es>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "io_png.h"

int main(int argc, char **argv)
{
  size_t nx,ny,nc,nx2,ny2,nc2, k, nxy, nxyc;
  float *U = NULL,*U2 = NULL, *ptr, *ptr2;
  double dif, Dist = 0.0, PSNR;
  

  if (argc < 3) 
    {
      fprintf(stderr,"Usage: img_mse_ipol <image1> <image2>\n");
      return EXIT_FAILURE;
    }
  
  /* Read input1 */
  U = io_png_read_f32(argv[1], &nx, &ny, &nc);
  if (!U) 
    {
      fprintf(stderr, "Error: <image1> '%s' not found \
or not a correct png image \n", 
	      argv[1]);
      return EXIT_FAILURE;
    }

  if (nc == 2) 
    {
      nc = 1;    /* We do not use the alpha channel */
      fprintf(stderr,
	      "Warning: number of channels in <image1> is 2, \
using only the first one.\n");
    }
  else
    if (nc > 3) 
      { 
	fprintf(stderr,
		"Warning: number of channels in <image1> is %d, \
using only the first 3 ones.\n", (int) nc);
	nc = 3;    /* We do not use the alpha channel */
      }
  
  /* Test if image is really a color image even if it has more than one 
     channel 
  */
  nxy = nx * ny;
  if (nc > 1) 
    {
      /* nc equals 3 */
      k=0;
      while (k < nxy && U[k] == U[nxy + k] && U[k] == U[2 * nxy + k ])  k++;
      if (k == nxy) 
	{
	fprintf(stderr,
		"Warning: color image <image1> is a gray level one, \
using only the first channel.\n");
	  nc = 1;
	}
    }

  
  /* Read input2 */
  U2 = io_png_read_f32(argv[2], &nx2, &ny2, &nc2);
  if (!U2) {
    printf("Error: <image2> '%s' not found or not a correct png image \n", 
	   argv[2]);
    return EXIT_FAILURE;
  }

  /* Test if same size */
  if (nx != nx2 || ny != ny2) 
    {
      fprintf(stderr, "Error: input images of different size\n");
      return EXIT_FAILURE;
    }
  
  if (nc2 == 2) 
    {
      nc2 = 1;     /* We do not use the alpha channel */
      fprintf(stderr,
	      "Warning: number of channels in <image2> is 2, \
using only the first one.\n");
    }
  else
    if (nc2 > 3) 
      {
	fprintf(stderr,
		"Warning: number of channels in <image2> is %d, \
using only the first 3 ones.\n", (int) nc2);
	nc2 = 3;    /* we do not use the alpha channel */
      }

  /* Test if image is really a color image even if it has more than one
     channel
  */
  if (nc2 > 1) 
    {
      /* nc2 equals 3 */
      k=0;
      while (k < nxy && U[k] == U[nxy + k] && U[k] == U[2 * nxy + k ]) k++;
      if (k == nxy) 
	{
	  fprintf(stderr,
		  "Warning: color image <image2> is a gray level one, \
using only the first channel.\n");
	  nc2 = 1;
	}
    }
  
  /* Test if same size */
  if (nc != nc2) 
    {
      fprintf(stderr,
	      "Error: input images of different number of channels\n");
      return EXIT_FAILURE;
    }
  
  /* Total number of values */
  nxyc=nc*nxy;
  
  /* Compute error and image difference */
  for (k=0, ptr=U, ptr2=U2; k < nxyc; k++, ptr++,ptr2++)
    {
      dif=*ptr-*ptr2;
      Dist += dif * dif;
    }
  
  Dist = sqrt(Dist/(double) nxyc);
  PSNR = 10.0 * log10(255.0 * 255.0 / (Dist * Dist));
  
  printf("RMSE = %2.2f\n", Dist);
  printf("PSNR = %2.2f\n", PSNR);
  
  return EXIT_SUCCESS;
}
