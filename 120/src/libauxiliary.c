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
 * @file libauxiliary.c
 * @brief Some auxiliary functions 
 *
 * @author Jacques Froment <Jacques.Froment@univ-ubs.fr>
 */

#include <stdlib.h>
#include <math.h>

#include "mt19937ar.h"

/* M_PI constant is not defined in strict ISO and/or POSIX compliance */
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

/**
 * @brief Add a gaussian noise to an array of floats using Mersenne Twister 
 *         pseudo-RNG code (see mt19937ar.c file)
 *
 * @param
 * U      Input floats array
 * V      Output floats array - has to be allocated before the call
 * sigma  Standard deviation of the Gaussian noise
 * seed   Seed to initialize the random numbers generator
 * size   Size of the U and V arrays (number of samples)  
*/

void addgaussnoise(float *U, float *V, double sigma, unsigned long seed, 
		   size_t size)

{
  float *pU,*pV;  /* Pointers to U and V */
  size_t k;       /* Index to address image's values */

  /* Initialize the generator with the seed */
  mt_init_genrand(seed);

  /* Compute the noisy float array V */
  for (k=0, pU=U, pV=V; k < size; k++, pU++,pV++)
    *pV = *pU + sigma * sqrt(-2.0 * log(mt_genrand_res53())) 
      * cos(2.0 * M_PI * mt_genrand_res53());
  
}

