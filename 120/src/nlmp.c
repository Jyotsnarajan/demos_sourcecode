/*
 * Copyright (C) 2014 Jacques Froment <Jacques.Froment@univ-ubs.fr>
 *
 * --------------------------------------------------------------------
 * Patent warning
 * --------------------------------------------------------------------
 * This file implements algorithms possibly linked to the patents
 *
 * # A. Buades, T. Coll and J.M. Morel, Image data processing method by
 * reducing image noise, and camera integrating means for implementing
 * said method, EP Patent 1,749,278 (Feb. 7, 2007).
 *
 * This file is made available for the exclusive aim of serving as
 * scientific tool to verify the soundness and completeness of the
 * algorithm description. Compilation, execution and redistribution
 * of this file may violate patents rights in certain countries.
 * The situation being different for every country and changing
 * over time, it is your responsibility to determine which patent
 * rights restrictions apply to you before you compile, use,
 * modify, or redistribute this file. A patent lawyer is qualified
 * to make this determination.
 * If and only if they don't conflict with any patent terms, you
 * can benefit from the following license terms attached to this
 * file.
 * --------------------------------------------------------------------
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

/**
 * @file nlmp.c
 * @brief Pixelwise Non-Local Means (NLM-P and NLM-Pa), basic and SIL algorithms
 * @author Jacques Froment <Jacques.Froment@univ-ubs.fr>
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
 #include <omp.h>
#endif

/* Usual min and max functions defined as macros */
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

/* The following is related to the LUT (Look-Up Table) defined to get a fast 
   exp() by tabulating exp(-x) for x in [0, MaxExpLUT] with step 1/SampExpLUT:
*/

/* Maximum value of x when tabulating exp(-x), x>=0.
   The LUT returns 0 if x is greater than this threshold
*/
#define MaxExpLUT 30.0
/* Sampling frequency for x when tabulating exp(-x), x>=0.
   The sampling period (interval between two adjacent samples) is 1/SampExpLUT.
*/
#define SampExpLUT 1000
/* Minimum distance to an integer to not to be considered as an integer */
#define EpsExpLUT 1e-10

/**
 * @brief Return the value of exp(-x) as tabulated by the given LUT, with
 *        linear interpolation between samples.
 *
 * @param
 * LUT       Precomputed LUT for exp(-x)
 * x         x in exp(-x), must be >= 0
*/

double explut(float *LUT, double x)

{
  int ix;       /* Lowest index for x in the LUT   */
  double xs;    /* x*SampExpLUT : if it is an integer, no need for interpolation */
  double xsmix; /* xs - ix as a double */
  float y0;     /* Value of the LUT for index ix   */
  float y1;     /* Value of the LUT for index ix+1 */

  if (x > MaxExpLUT) return(0.0);
  xs=x*SampExpLUT;
  ix=(int)xs;
  xsmix=xs-ix;
  y0=LUT[ix]; 
  if (xsmix<EpsExpLUT) return(y0);
  /* xs is not an integer, perform linear interpolation */
  y1=LUT[ix+1]; 
  return((y1-y0)*xsmix+y0);  
}

/**
 * @brief Main NLM-P and NLM-Pa function
 *
 * @param
 * V         Input noisy image
 * sigma     Noise standard deviation
 * ds        Patch side length is d=2*ds+1
 * Ds        Search window side length is D=2*Ds+1 
 * N1,N2     Size of the input image: N2xN1 matrix with 
 *           N2=number of rows (dx2) and N1=number of columns (dx1) 
 * Nc        Number of channels in the input image (1 or 3)
 * a         Parameter of the Gaussian Euclidean norm (NLP-Pa) or 0 for the 
 *           plain Euclidean norm (NLM-P) 
 * h         Filtering parameter in the Gaussian weights NLM-P and NLM-Pa
 * BasicAlgo 1 to run the basic algorithm, 0 to run fast SIL algorithm
 *
*/

float *nlmp(float *V, int ds, int Ds, int N1, int N2, int Nc, float a, float h, 
	    char BasicAlgo)
     
{
  /* Declaration of shared local variables.
   * By shared variables we mean variables that are shared between threads
   * (if inside a parallel section). 
   * Inside a parallel section, this is safe only for read-only variables. 
   */
  float *Vd;      /* Output denoised image (return value) */
  float *Vsym;    /* Symmetrized noisy image */
  float *ExpLUT;  /* LUT (Look-Up Table) for a fast exp(): tabulate exp(-x) */
  int lExpLUT;    /* Length of ExpLUT (number of samples) */
  double *K;      /* Kernel used to compute patchs similarity */
  int adr;        /* Pixel address in V or in Vd */
  int symadr;     /* Pixel address in Vsym */
  int l,lsym;     /* Shift due to channel c in V/Vd and in Vsym */
  int *padr;      /* Patch pixels addresses */
  int N;          /* Number of pixels of the input image */
  int NNc;        /* Size of the input image (N * Nc) */
  int border;     /* Border size Ds+ds to be added to the input image */
  int N2sym,N1sym;/* Size of the symmetrized image */
  int NS;         /* N for  the symmetrized image */
  int d;          /* Patch side length */
  int d2;         /* Patch side size */
  int D;          /* Search window side length */
  int D2;         /* Search window side size */
  int Dd;         /* D*d */
  int D2d;        /* D2*d */
  int i,j;        /* Various indexes */
  int x1,x2,c;    /* Pixel (x1,x2) and color channel c */
  double s;       /* Multi-purpose variable */

  /* Initialization */
  
  N=N2*N1;NNc=Nc*N; /* Image size (one plane;all planes) */
  border=Ds+ds;     /* Border size to be added to perform symmetrization */
  d=2*ds+1; d2=d*d; /* Patch side length and size */
  D=2*Ds+1; D2=D*D; /* Search window side length and size */
  Dd=D*d; D2d=D2*d; 

  if ((N1<=border)||(N2<=border))
    {
      fprintf(stderr,
	      "Image of size (%d,%d) too small regarding border size %d\n",
	      N1,N2,border);
      fprintf(stderr,"Please decrease ds and/or Ds parameter.\n");
      exit(EXIT_FAILURE);       
    }

  /* Memory allocation of output image */
  Vd =(float *)malloc(NNc*sizeof(float)); 
  if (!Vd) 
    {
      fprintf(stderr,"Not enough memory (Vd)\n");
      exit(EXIT_FAILURE);       
    }

  /* Memory allocation of symmetrized noisy image Vsym */
  N1sym=N1+2*border; N2sym=N2+2*border; NS=N2sym*N1sym;
  Vsym =(float *)malloc(Nc*NS*sizeof(float)); 
  if (!Vsym) 
    {
      fprintf(stderr,"Not enough memory (Vsym)\n");
      exit(EXIT_FAILURE);       
    }
  
  /* Memory allocation of the LUT (Look-Up Table) for fast exp() access */
  lExpLUT=1+ceil((double)MaxExpLUT*SampExpLUT);
  ExpLUT=(float *)malloc(lExpLUT*sizeof(float));
  if (!ExpLUT)
    {
      fprintf(stderr,"Not enough memory (ExpLUT)\n");
      exit(EXIT_FAILURE);       
    }
  /* Fill the LUT */
  ExpLUT[0]=1.0;
  for (i=1; i<lExpLUT; i++) ExpLUT[i]=exp(-(double)i/SampExpLUT);

  /* Compute the symmetrized image Vsym */
  for (j=0;j<N2sym;j++)
    {
      if (j<border ) 
	x2=border-j;
      else 
	if (j>N2+border-1) 
	  x2=2*N2+border-j-2;
	else 
	  x2=j-border;
      
      for (i=0;i<N1sym;i++)
	{
	  if (i<border) 
	    x1=border-i;
	  else 
	    if (i>N1+border-1)
	      x1=2*N1+border-i-2;
	    else 
	      x1=i-border;

	  /* Pixel address in V and in Vsym */
	  adr=x2*N1+x1;
	  symadr=j*N1sym+i;
	  for (l=lsym=c=0; c<Nc; c++, l+=N, lsym+=NS)
	    Vsym[lsym+symadr]=V[l+adr];
	}
    }

  /* Basic Dist2 computation is the fastest way when ds=1 */
  if (ds<=1) BasicAlgo=1;

  /* Allocation of the kernel and padr tabs */
  if (BasicAlgo) K=(double*)malloc(d2*sizeof(double));
  else K=(double*)malloc(d*sizeof(double));
  if (!K) 
    {
      fprintf(stderr,"Not enough memory (K)\n");
      exit(EXIT_FAILURE);       
    }
  padr  = (int*)malloc(d2*sizeof(int));
  if (!padr) 
    {
      fprintf(stderr,"Not enough memory (padr)\n");
      exit(EXIT_FAILURE);       
    }

  /* Computation of the kernel K */

  a*=2*a;
  s=0;
  if (BasicAlgo)
    /* Without fast SIL computation, use standard 2D kernel form */
    {
      for(i=0,x2=-ds; x2<=ds;x2++)
	for(x1=-ds;x1<=ds;x1++,i++)
	  {
	    padr[i]=x2*N1sym+x1;
	    if (a > 0) /* NLM-Pa: K = Gaussien kernel of standard deviation a */
	      K[i]=exp(-((double)x1*x1+x2*x2)/a);
	    else /* NLM-P: K = constant kernel */
	      K[i]=1;
	    s+=K[i];
	  }
      /* Normalization of the kernel.
	 To optimize computations, factor 1/h^2 is put in the kernel.
	 Beware: h^2 here, 2h^2 in some other NLM implementations 
      */
      for(i=d2;i--;) K[i]/=s*h*h;
    }
  else
    /* With fast SIL computation, use separable kernel form */
    {
      for(i=0,x2=-ds; x2<=ds;x2++,i++)
	{
	  if (a > 0) /* NLM-Pa: K = Gaussien kernel of standard deviation a */
	    K[i]=exp(-((double)x2*x2)/a);
	  else /* NLM-P: K = constant kernel */
	    K[i]=1;
	  s+=K[i];
	  padr[i]=x2*N1sym;
	}
      /* Normalization of the kernel.
	 To optimize computations, factor 1/h^2 is put in the kernel.
	 Beware: h^2 here, 2h^2 in some other NLM implementations 
      */
      for(i=d;i--;) K[i]/=s*h;
    }

  /*     ***** PARALLEL SECTION *****
   */
#pragma omp parallel 
  {
    /* Declaration of private local variables */
    int nthrds;   /* Number of threads running in parallel */
    int tid;      /* Current thread id */
    int x1,x2,c;  /* Pixel x=(x1,x2) and color channel c */
    int x1s;      /* Step x1 in parallelized main loop */
    int y1,y2;    /* Pixel y=(y1,y2) within a patch */
    int adr;      /* Pixel address in V or in Vd */
    int symadr;   /* Pixel address in Vsym */
    int symadrp;  /* Pixel address in Vsym inside a patch */
    int l,lsym;   /* Shift due to channel c in V/Vd and in Vsym */
    int i,j,k;    /* Some indexes */
    int *dd;      /* Pointer to padr */
    double Dist2; /* Weighted MSE between patches x=(x1,x2) and y=(y1,y2) */
    double di;    /* Difference between two pixel values */
    double edi;   /* Euclidean difference between two pixel values */
    double r;     /* Multi-purpose variable */
    double s;     /* Multi-purpose variable */
    double w;     /* NLM weight w(x,y) */
    double *W;    /* NLM weights tab for a search window */
    double *Wptr; /* Pointer to the NLM weights tab */

    /* The following variables are used for the fast SIL computation only */

    double *L2;   /* Dist2 for a given line j, between patches (x1,x2) and
		     (y1,y2). Corresponding address is
		     (x2%2)*D2d + (y2+Ds-x2)*Dd + (y1+Ds-x1)*d + j
		  */
    double l2;    /* Current computation of L2[] */
    int L2adr;    /* Pixel address in L2 (line 0) */
    int L2sadr;   /* Shifted pixel address in L2 (line 0) */
    int llsymadr; /* Last line relative to symadr */
    int llsymadrp;/* Last line relative to symadrp */

    W=(double*)malloc(D2*sizeof(double));
    if (!W) 
      {
	fprintf(stderr,"Not enough memory (W)\n");
	exit(EXIT_FAILURE);       
      }

    if (!BasicAlgo)
      {
	L2=(double*)malloc(2*D2d*sizeof(double));  
	if (!L2) 
	  {
	    fprintf(stderr,"Not enough memory (L2)\n");
	    exit(EXIT_FAILURE);       
	  }
      }
    else L2=NULL;

#ifdef _OPENMP
    /* Obtain thread id and number of threads */
    tid=omp_get_thread_num();
    nthrds=omp_get_num_threads();
#else
    tid=0; nthrds=1;
#endif

    /* MAIN LOOP : denoise pixel x=(x1,x2)
       The first loop follows x1 axis and is dispatched on all 
       available threads (therefore N1 should be greater than nthrds).
     */
    for(x1s=tid;x1s<N1;x1s+=nthrds)
      for(x2=border;x2<N2+border;x2++)
	{
	  x1=x1s+border;
	  /* x=(x1,x2) is the pixel to denoise, center of the 1st patch
	   */
	  /* Pixel address in V and in Vsym */
	  adr=(x2-border)*N1+x1s;
	  symadr=x2*N1sym+x1;
	  /* FIRST PASS
	     Scan the 2d patch to compute weighted MSE between patches 
	     centered at x=(x1,x2) and at y=(y1,y2) as well as NLM-weigths tab.
	     Note that these data do NOT depend on a color plane.
	  */
	  for(Wptr=W,y2=x2-Ds;y2<=x2+Ds;y2++)
	    for(y1=x1-Ds;y1<=x1+Ds;y1++)
	      {
		/* y=(y1,y2) is the center of the 2d patch */
		symadrp=y2*N1sym+y1;
		Dist2=0;
		if (BasicAlgo)
		  /* Basic computation of Dist2 */
		  for(j=d2,dd=padr;j--;dd++)
		    /* Color plane #c */
		    for (c=lsym=0; c<Nc; c++, lsym+=NS)
		      {
			di=Vsym[lsym+symadr+*dd]-Vsym[lsym+symadrp+*dd];
			Dist2+=K[j]*di*di;   
		      }
		else
		  /* Fast computation of Dist2 (SIL algorithm) */
		  {
		    L2adr=(y2+Ds-x2)*Dd+(y1+Ds-x1)*d;
		    L2sadr=L2adr+((x2-1)%2)*D2d;
		    L2adr+=(x2%2)*D2d;
		    if (x2==border)
		      /* Patch V(x) is on the upper side: 
			 compute and record Dist2 for all lines */
		      for (j=0,dd=padr; j<d; j++,dd++) 
			/* line j-ds */
			{
			  llsymadr=symadr+*dd-ds;
			  llsymadrp=symadrp+*dd-ds;
			  l2=0;
			  for (k=0,i=-ds; i<=ds; i++)
			    /* column i */
			    {
			      edi=0;
			      for (c=lsym=0; c<Nc; c++, lsym+=NS)
				/*  Color plane #c */
				{
				  di=Vsym[lsym+llsymadr] -
				    Vsym[lsym+llsymadrp];
				  edi+=di*di;
				}
			      l2+=K[k++]*edi;
			      llsymadr++;
			      llsymadrp++;
			    }
			  L2[L2adr+j]=l2;
			  Dist2+=K[j]*l2;
			}
		    else
		      /* Patch not on upper side: use previously computed 
			 lines with a shift and compute last line */
		      {
			/* Shift 1 line down all previously computed
			   lines but the last one */
			for (j=0,dd=padr; j<d-1; j++,dd++) 
			  {
			    l2=L2[L2sadr+j+1]; /* Shift 1 line down */
			    Dist2+=K[j]*l2;
			    L2[L2adr+j]=l2;
			  }
			/* Compute last line (j=d-1) */
			dd=&padr[j];
			llsymadr=symadr+*dd-ds;
			llsymadrp=symadrp+*dd-ds;
			l2=0;
			for (k=0,i=-ds;i<=ds; i++)
			  /* Column i */
			  {
			    edi=0;
			    for (c=lsym=0; c<Nc; c++, lsym+=NS)
			      /*  Color plane #c */
			      {
				di=Vsym[lsym+llsymadr] -
				  Vsym[lsym+llsymadrp];
				edi+=di*di;
			      }
			    l2+=K[k++]*edi;   
			    llsymadr++;
			    llsymadrp++;
			  }
			L2[L2adr+j]=l2;
			Dist2+=K[j]*l2;
		      }
		  }  /* End of Fast computation of Dist2 (SIL algorithm) */
		*Wptr++=explut(ExpLUT,Dist2/Nc);
	      } /* End of FIRST PASS (for y1, for y2) */
		  
	  /* SECOND PASS
	     For each color plane, scan the 2d patch to compute the pixel's 
	     estimate at x 
	  */
	  for (c=l=lsym=0; c<Nc; c++, l+=N, lsym+=NS)
	    {
	      /* Color plane #c */
	      r=s=0;
	      for(Wptr=W,y2=x2-Ds;y2<=x2+Ds;y2++)
		for(y1=x1-Ds;y1<=x1+Ds;y1++)
		  {
		    /* y=(y1,y2) is the center of the 2d patch */
		    symadrp=lsym+y2*N1sym+y1;
		    w=*Wptr++;
		    r+=w*Vsym[symadrp];
		    s+=w;
		  } /* End of for y1, for y2 */
	      Vd[l+adr] = MIN( MAX( (r/s),0 ),255 );
	    } /* End of for (c=l=lsym=lL=0...) */
	} /* End of for x1s, x2 */

    /* Memory deallocation of private variables */
    if (L2) free(L2);
    free(W);

  } /*  ***** END OF PARALLEL SECTION ***** */
 
  /* Memory deallocation of shared variables */
  free(padr);
  free(K);
  free(ExpLUT);
  free(Vsym);

  return(Vd);
}
