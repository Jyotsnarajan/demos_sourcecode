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
 * @file NLMeansP.c
 * @brief Main file for image denoising by standard Pixelwise Non-Local Means.
 * parse command-line, read input image, run algorithm and write output image.
 *
 * @author  Jacques Froment <Jacques.Froment@univ-ubs.fr>
 */

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <time.h>
#include <unistd.h>

#include "io_png.h"
#include "libauxiliary.h"

/* Call NLMeansP denoising algorithm: prototype for function in nlmp.c */
extern float *nlmp(float *V,int ds,int Ds,int N1,int N2,int Nc,float a,
		   float h,char BasicAlgo);

/*
  To mark a parameter as not set
 */
#define PARAM_NOT_SET -1

/* Image size limit. This is the maximum of (image size * number of channels).
   Estimate this limit to obtain a computational time in the demo environment 
   below 30s.
 */
#define NNcmax 6560000

/**
 * @brief print usage and exit
 */

void PrintUsage(void)
{
  fprintf(stderr, "usage: NLMeansP [-b] [-c] [-s seed] [-d ds] [-D Ds] [-a a] \
[-h h] U sigma V Vd\n\n");
  fprintf(stderr,"\t-b: basic algorithm (default: SIL algorithm)\n");
  fprintf(stderr,"\t-c: clock (print CPU time of the denoising step)\n");
  fprintf(stderr,"\t-s seed:Seed for the random noise generator (default:\
pseudo-random seed)\n");
  fprintf(stderr, "\t-d ds:\tSet patch side length to d=2*ds+1\n");
  fprintf(stderr, "\t-D Ds:\tSet search window side length to D=2*Ds+1\n");
  fprintf(stderr, "\t-a a:\tSet parameter of the Gaussian Euclidean norm\n");
  fprintf(stderr, "\t     \tor 0 to select the plain Euclidean norm\n");
  fprintf(stderr, "\t-h h:\tSet the filtering parameter\n");
  fprintf(stderr, "\tU:\tInput original image (PNG format)\n");
  fprintf(stderr, "\tsigma:\tStandard deviation of the noise to be applied\n");
  fprintf(stderr, "\tV:\tOutput noisy image (PNG format)\n");
  fprintf(stderr, "\tVd:\tOutput denoised image (PNG format)\n");
  exit(EXIT_FAILURE); 
}

/**
 * @brief set default parameter 'a' for PSNR optimality (on average).
 *        See manuscript, Table "Optimal parameters for NLM-Pa".
 */

float SetParam_a(size_t Nc, double sigma)

{
  float a10;        /* Parameter of the Gaussian Euclidean norm, multiplied
                       by a factor 10 to follow the chosen quantization 
		       (see manuscript, Table "Range of the NLM-P and NLM-Pa
		       parameter exploration for the optimization process").
		       Set 0 for plain Euclidean norm (NLM-P).*/

  switch (Nc) {
    
  case 1:
    if (sigma<=1) a10=7;
    else
      if (sigma<3) a10=8;
      else
	if (sigma<=4) a10=9;
	else
	  if (sigma<=5) a10=10;
	  else
	    if (sigma<=7) a10=11;
	    else
	      if (sigma<=9) a10=13;
	      else
		if (sigma<=13) a10=14;
		else
		  if (sigma<=18) a10=16;
		  else
		    if (sigma<=19) a10=17;
		    else
		      a10=sigma;
    break;

  case 3:
    if (sigma<=3) a10=sigma+2;
    else
      if (sigma<=6) a10=sigma+1;
      else
	if (sigma<=9) a10=7;
	else
	  if (sigma<=13) a10=10;
	  else
	    if (sigma<=19) a10=11;
	    else
	      a10=sigma;
    break;

  default:
    fprintf(stderr,"Invalid Nc=%d value\n",(int)Nc);
    exit(EXIT_FAILURE);       
    break;
  }
  return(a10/10.0);
}


/**
 * @brief set default parameter 'ds' for PSNR optimality (on average).
 *        See manuscript, Table "Optimal parameters for NLM-Pa".
 */

int SetParam_ds(size_t Nc, double sigma, float a)

{
  switch (Nc) {
    
  case 1: /* Gray-level image */
    if (a==0) /* NLM-P */
      {
	if (sigma<=19) return(1);
	if (sigma<=28) return(2);
	if (sigma<=87) return(3);
	return(4);
      }
    else /* NLM-Pa */
      {
	if (sigma<=83) return(3);
	return(4);
      }
    break;

  case 3: /* Color image */
    /* NLM-P & NLM-Pa */
    if (sigma<=46) return(1);
    return(2);
    break;

  default:
    fprintf(stderr,"Invalid Nc=%d value\n",(int)Nc);
    exit(EXIT_FAILURE);       
    break;
  }
}

/**
 * @brief set default parameter 'Ds' for PSNR optimality (on average).
 *        See manuscript, Table "Optimal parameters for NLM-Pa".
 */

int SetParam_Ds(size_t Nc, double sigma, float a)

{
  switch (Nc) {
    
  case 1: /* Gray-level image */
    if (a==0) /* NLM-P */
      {
	if (sigma<=7)  return(3);
	if (sigma<=9)  return(4);
	if (sigma<=19) return(5);
	if (sigma<=47) return(6);
	if (sigma<=70) return(7);
	return(8);
      }
    else /* NLM-Pa */
      {
	if (sigma<=5)  return(3);
	if (sigma<=9)  return(4);
	if (sigma<=20) return(5);
	if (sigma<=28) return(6);
	if (sigma<=67) return(7);
	return(8);
      }
    break;

  case 3: /* Color image */
    if (a==0) /* NLM-P */
      {
	if (sigma<=3)  return(2);
	if (sigma<=8)  return(3);
	if (sigma<=9)  return(4);
	if (sigma<=17) return(5);
	if (sigma<=24) return(6);
	if (sigma<=46) return(8);
	if (sigma<=75) return(9);
	return(10);
      }
    else /* NLM-Pa */
      {
	if (sigma<=9)  return(5);
	if (sigma<=24) return(6);
	if (sigma<=45) return(8);
	if (sigma<=79) return(9);
	return(10);
      }
    break;

  default:
    fprintf(stderr,"Invalid Nc=%d value\n",(int)Nc);
    exit(EXIT_FAILURE);       
    break;
  }
}

/**
 * @brief set default parameter 'h' for PSNR optimality (on average).
 *        See manuscript, Table "Optimal parameters for NLM-Pa".
 */

float SetParam_h(size_t Nc, double sigma, float a)

{
  float hf;         /* Factor to get the filtering parameter h=hf*sigma/10
                      (see manuscript, Table "Range of the NLM-P and NLM-Pa
		       parameter exploration for the optimization process"). */

  switch (Nc) {
    
  case 1: /* Gray-level image */
    if (a==0) /* NLM-P */
      {
	if (sigma<=7)  hf=15;
	else
	  if (sigma<=9)  hf=14;
	  else
	    if (sigma<=19) hf=13;
	    else
	      if (sigma<=28) hf=11;
	      else
		hf=10;
      }
    else /* NLM-Pa */
      {
	if (sigma<=5)  hf=17;
	else
	  if (sigma<=7)  hf=16;
	  else
	    if (sigma<=9)  hf=14;
	    else
	      if (sigma<=19) hf=13;
	      else
		if (sigma<=20) hf=12;
		else
		  if (sigma<=28) hf=11;
		  else
		    hf=10;
      }
    break;

  case 3: /* Color image */
    if (a==0) /* NLM-P */
      {
	if (sigma<=3)  hf=15;
	else
	  if (sigma<=8)  hf=14;
	  else
	    if (sigma<=9)  hf=13;
	    else
	      if (sigma<=17) hf=12;
	      else
		if (sigma<=24) hf=11;
		else
		  if (sigma<=46) hf=10;
		  else
		    hf=9;
      }
    else /* NLM-Pa */
      {
	if (sigma<=4)  hf=16;
	else
	  if (sigma<=5)  hf=15;
	  else
	    if (sigma<=9)  hf=14;
	    else
	      if (sigma<=19) hf=12;
	      else
		if (sigma<=24) hf=11;
		else
		  if (sigma<=46) hf=10;
		  else
		    hf=9;
      }
    break;

  default:
    fprintf(stderr,"Invalid Nc=%d value\n",(int)Nc);
    exit(EXIT_FAILURE);       
    break;
  }
  return(hf*sigma/10.0);
}



/**
 * @brief main function call
 */

int main(int argc, char *argv[])
{
 
  /* Parameters of NLM-P & NLM-Pa  */
  int ds=PARAM_NOT_SET; /* Window side length is d=2*ds+1 */
  int Ds=PARAM_NOT_SET; /* Search window side length is D=2*Ds+1 */
  float a=PARAM_NOT_SET;/* Gaussian Euclidean norm (0 for plain NLM-P) */
  float h=PARAM_NOT_SET;/* Filtering parameter */
  
  /* Needed arguments */
  double sigma;            /* Standard deviation of the noise to be applied */

  /* Input/output images */
  float *U=NULL;           /* Input noiseless image  */
  float *V=NULL;           /* Output noisy image  */
  float *Vd=NULL;          /* Output denoised image */

  /* Size of the input image: N2xN1 matrix */
  size_t N2,N1;/* N2=number of rows (dx2) and N1=number of columns (dx1) */
  size_t Nc;   /* Number of channels */
  size_t NNc;  /* Total size of the image: N2xN1xNc */

  /* Others */
  int opt;              /* For command-line sparsing */
  unsigned long seed=0; /* Seed for the random generator */
  char sopt=0;          /* Flag for -s option */
  char BasicAlgo=0;     /* Flag for -b option */

  /* Clock option -c */
  char copt=0;          /* Flag for -c option */
  clock_t t0=0,t1=0;    /* Clock values at start and end */
  double cputime;       /* Sum of processors (threads) time in seconds */

  /* Parse options */
  while ((opt=getopt(argc, argv, "bcs:d:D:a:h:")) != -1) 
    {
      switch (opt) {
      case 'b':
	BasicAlgo=1;
	break;
      case 'c':
	copt=1;
	break;
     case 's' :
	seed = atol(optarg);
	sopt = 1;
	break;
      case 'd' :
	ds=atoi(optarg);
	break;
      case 'D' :
	Ds=atoi(optarg);
	break;
      case 'a':
	a =atof(optarg);
	break;
      case 'h':
	h =atof(optarg);
	break;
      default :
	PrintUsage();
	break;
      }
    }
  
  /* Command has 4 needed arguments */
  if (optind+4 != argc) 
    {
      fprintf(stderr,"Please give the arguments U, sigma, V and Vd.\n");
      PrintUsage();
    }  

  sigma=atof(argv[optind+1]);
  if (sigma <= 0) 
    {
      fprintf(stderr,"Illegal value sigma=%f\n",sigma);
      exit(EXIT_FAILURE);       
    }

  /* Read input image in PNG format */
  U = io_png_read_f32(argv[optind], &N1, &N2, &Nc);
  if (!U) 
    {
      fprintf(stderr,"Couldn't read input PNG image '%s'\n",
	      argv[optind]);
      exit(EXIT_FAILURE); 
    }

  /* Check image validity */
  if ((Nc != 1)&&(Nc != 3))
    {
      fprintf(stderr,
	      "Neither a gray-level nor a 3-channels color input image \
(number of channels is %d)\n", (int) Nc);
      exit(EXIT_FAILURE);       
    }
  NNc=N2*N1*Nc;
  if (NNc > NNcmax)
    {
      fprintf(stderr,
	      "To maintain a reasonable computation time, the pixels' number \
is limited to %ld.\n",
	      (long int) NNcmax);
      fprintf(stderr,
	      "The input image of size (%ld,%ld) and %ld color plane(s) \
contains %ld pixels.\n",
	      (long int) N1,(long int) N2,(long int) Nc,(long int) NNc);
      fprintf(stderr,"Please decrease the image size or the number of \
planes.\n");
      exit(EXIT_FAILURE);       
    }    


  /* Set parameters that have not been selected in the command line.
     Beware: selecting only some of these 4 parameters may lead to
     very sub-optimal results since, in the following default setting,
     values are NOT independent from each other.
  */
  if (a==PARAM_NOT_SET)  a=SetParam_a(Nc,sigma);
  /* Setting a=0 means using plain Euclidean norm (NLM-P) instead of Gaussian 
     Euclidean one (NLM-Pa).
     Beware: optimal ds, Ds & h parameters depend whether or not a=0.
  */
  if (ds==PARAM_NOT_SET) ds=SetParam_ds(Nc,sigma,a);
  if (Ds==PARAM_NOT_SET) Ds=SetParam_Ds(Nc,sigma,a);
  if (h==PARAM_NOT_SET)  h=SetParam_h(Nc,sigma,a);

  /* Memory allocation of noisy image V */
  V =(float *)malloc(NNc*sizeof(float)); 
  if (!V) 
    {
      fprintf(stderr,"Not enough memory (V)\n");
      exit(EXIT_FAILURE);       
    }

  /* Compute V = U + noise, using given or pseudo random seed */
  if (sopt==0) seed=time(NULL) + getpid();
  addgaussnoise(U, V, sigma, seed, NNc);

  /* Call NLM-P/NLM-Pa denoising scheme, clocking the call if requested */
  if (copt==1) t0=clock();
  Vd=nlmp(V,ds,Ds,(int)N1,(int)N2,(int)Nc,a,h,BasicAlgo);
  if (copt==1) t1=clock();

  /* Write output noisy image in PNG format 
     WARNING: as PNG format thresholds values outside [0,255], the output
     does not exactly matches the noisy image on which computation has
     been done. In particular, the noise standard deviation is lower.
   */
  if (io_png_write_f32(argv[optind+2], V, N1, N2, Nc) != 0)
    {
      fprintf(stderr,"Couldn't write output noisy image V '%s'\n",
	      argv[optind+1]);
      exit(EXIT_FAILURE);       
    }    

  /* Write output denoised image in PNG format */
  if (io_png_write_f32(argv[optind+3], Vd, N1, N2, Nc) != 0) 
    {
      fprintf(stderr,"Couldn't write output denoised image Vd'%s'\n",
	      argv[optind+1]);
      exit(EXIT_FAILURE);       
    }    

  /* Free the memory */
  free(Vd);
  free(V);
  free(U);

  if (copt==1)  /* Clock option selected: print the CPU time */
    {
      cputime=(t1-t0)/(double) CLOCKS_PER_SEC;
      printf("CPU time=%.4f",cputime);
    }
  
  exit(EXIT_SUCCESS);
}
