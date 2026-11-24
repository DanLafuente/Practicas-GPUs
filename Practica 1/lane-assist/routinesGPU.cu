#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "routinesGPU.h"



#define CUDA_CALL(x)                                                            \
	do                                                                          \
	{                                                                           \
		cudaError_t result = x;                                                 \
		if (result != cudaSuccess)                                              \
		{                                                                       \
			printf("CUDA error at %s:%d code=%d(%s) \"%s\"\n",                  \
				   __FILE__, __LINE__, result, cudaGetErrorString(result), #x); \
			exit(EXIT_FAILURE);                                                 \
		}                                                                       \
	} while (0)


//------------------------------- Kernels Canny -------------------------------//
	__global__ void canny_NR_kernel(uint8_t *im,
									float *NR,
									int height, int width);

	__global__ void canny_Gx_kernel(
									float *NR, float *Gx,
									int height, int width);

	__global__ void canny_Gy_kernel(
									float *NR, float *Gy,
									int height, int width);

	__global__ void canny_G_kernel(
									float *G, float *Gx, float *Gy,
									int height, int width);

	__global__ void canny_phi_kernel(
									float *phi, float *Gx, float *Gy,
									int height, int width);

	__global__ void canny_pedge_kernel(
									uint8_t *pedge, float *phi, float *G,
									int height, int width);

	__global__ void canny_Umbralimg_kernel(uint8_t *image_out,
									uint8_t *pedge, float *G, 
									float level, 
									int height, int width);	
//------------------------------- Kernels Canny -------------------------------//


//------------------------------- Kernels Hough -------------------------------//
	__global__ void init_cos_sin_table_kernel(float *sin_table_GPU, float *cos_table_GPU, 
									int n);
	__global__ void houghtransform_kernel(uint8_t *im, int width, int height, 
									uint32_t *accumulators, int accu_width, int accu_height,
									float *sin_table, float *cos_table);
//------------------------------- Kernels Hough -------------------------------//


//-------------------------------- CPU Pintar lineas --------------------------------//
void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines);
//-------------------------------- CPU Pintar lineas --------------------------------//

void lane_assist_GPU(uint8_t *im, int height, int width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{
	/*
	* 1. Detencion de Bordes
	*/

	/*Reservar espacio en memoria & Definiciones Utiles*/
	uint8_t *im_GPU, *imEdge_GPU, *pedge_GPU;
	float *NR_GPU, *G_GPU, *Gx_GPU, *Gy_GPU, *phi_GPU, level = 1000.0f;

	const int size_matrix_U8 = sizeof(uint8_t) * width * height;
	const int size_matrix_Fl = sizeof(float) * width * height;

	/*Cuda Malloc, utilizando cudaCall para comprobar que lo haga bien*/
	CUDA_CALL(cudaMalloc((void **)&im_GPU, size_matrix_U8));
	CUDA_CALL(cudaMemcpy(im_GPU, im, size_matrix_U8, cudaMemcpyHostToDevice));//Vuelco la img en GPU
	CUDA_CALL(cudaMalloc((void **)&imEdge_GPU, size_matrix_U8));
	CUDA_CALL(cudaMalloc((void **)&NR_GPU, size_matrix_Fl);
	CUDA_CALL(cudaMalloc((void **)&G_GPU, size_matrix_Fl)));
	CUDA_CALL(cudaMalloc((void **)&phi_GPU, size_matrix_Fl));
	CUDA_CALL(cudaMalloc((void **)&Gx_GPU, size_matrix_Fl));
	CUDA_CALL(cudaMalloc((void **)&Gy_GPU, size_matrix_Fl));
	CUDA_CALL(cudaMalloc((void **)&pedge_GPU, size_matrix_U8));

	/*Inicializo el nº de bloques y el nº de hilos*/
	dim3 blockDim(16, 16);
	int ejeX = ceil(width/blockDim.x), ejeY= ceil(width/blockDim.y);
	dim3 gridDim(ejeX, ejeY);

	/*Canny en GPU (detencion de bordes)*/
	canny_NR_kernel<<<gridDim, blockDim>>>(im_GPU, NR_GPU, height, width);
	canny_Gx_kernel<<<gridDim, blockDim>>>(NR_GPU, Gx_GPU, height, width);
	canny_Gy_kernel<<<gridDim, blockDim>>>(NR_GPU, Gy_GPU, height, width);
	canny_G_kernel<<<gridDim, blockDim>>>(G_GPU, Gx_GPU, Gy_GPU, height, width);
	canny_phi_kernel<<<gridDim, blockDim>>>(phi_GPU, Gx_GPU, Gy_GPU, height, width);
	canny_pedge_kernel<<<gridDim, blockDim>>>(pedge_GPU, phi_GPU, G_GPU, height, width);
	canny_Umbralimg_kernel<<<gridDim, blockDim>>>(imEdge_GPU, pedge_GPU, G_GPU, level, height, width);
	//Aqui ya tenemos todos lo bordes pintados en imEdge_GPU, (si pusiera un Synchronize)

	/*
	* 2. Ahora hay que hacer la detencion de lineas
	*/

	/*Valores que necesitamos table & accum & imEdge_GPU*/
	float hough_h = ((sqrt(2.0) * (float)(height > width ? height : width)) / 2.0);
	int accu_height = hough_h * 2.0; // -rho -> +rho
	int accu_width = 180;
	const int size_matrix_accum = width * height * sizeof(uint32_t);
	uint32_t *accum_GPU;
	CUDA_CALL(cudaMalloc((void **)&accum_GPU, size_matrix_accum));
	const int size_accum_wh = accu_width * accu_height * sizeof(uint32_t);
	CUDA_CALL(cudaMemset(accum_GPU, 0, size_accum_wh));

	float *sin_table_GPU, *cos_table_GPU;
	const int size_table = 180 * sizeof(float);
	CUDA_CALL(cudaMalloc((void**)&sin_table_GPU, size_table));
	CUDA_CALL(cudaMalloc((void**)&cos_table_GPU, size_table));
	

	/*Hough en GPU*/
	init_cos_sin_table_kernel<<<dim3(1), dim3(180)>>>(sin_table_GPU, cos_table_GPU, 180);//en 1 dimension
	houghtransform_kernel<<<gridDim, blockDim>>>(imEdge_GPU, width, height, accum_GPU, accu_height, accu_width, sin_table_GPU, cos_table_GPU);

	/*
	* 3. Mostrar la solucion
	*/

	/*Primero volcamos en memoria CPU lo de la GPU (GPU -> CPU)*/
	uint32_t *accum_CPU = (uint32_t *)malloc(size_matrix_accum);
	float *sin_table_CPU = (float *)malloc(size_table), *cos_table_CPU = (float *)malloc(size_table);

	cudaMemcpy(accum_CPU,accum_GPU,size_matrix_accum,cudaMemcpyDeviceToHost);
	cudaMemcpy(sin_table_CPU,sin_table_GPU, size_table,cudaMemcpyDeviceToHost);
	cudaMemcpy(cos_table_CPU,cos_table_GPU, size_table,cudaMemcpyDeviceToHost);

	/*Ahora pintar las lineas*/
	CUDA_CALL(cudaDeviceSynchronize());//Antes hay que tenerlo todo terminado

	int threshold;
	if (width > height)
		threshold = width/6;
	else 
		threshold = height/6;

	getlines(threshold, accum_CPU, accu_width, accu_height, width, height, sin_table_CPU, cos_table_CPU, x1, y1, x2, y2, nlines);

	/*
	* 4. Liberar Memoria
	*/

	/*GPU*/
	CUDA_CALL(cudaFree(im_GPU));
	CUDA_CALL(cudaFree(imEdge_GPU));
	CUDA_CALL(cudaFree(NR_GPU));
	CUDA_CALL(cudaFree(G_GPU));
	CUDA_CALL(cudaFree(phi_GPU));
	CUDA_CALL(cudaFree(Gx_GPU));
	CUDA_CALL(cudaFree(Gy_GPU));
	CUDA_CALL(cudaFree(pedge_GPU));
	CUDA_CALL(cudaFree(accum_GPU));
	CUDA_CALL(cudaFree(sin_table_GPU));
	CUDA_CALL(cudaFree(cos_table_GPU));

	/*CPU*/
	free(accum_CPU);
	free(sin_table_CPU);
	free(cos_table_CPU);
}



//-------------------------------- Kernels del Canny --------------------------------//
__global__ void canny_NR_kernel(uint8_t *im,
								float *NR,
								int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 2 && i < height - 2 && j >= 2 && j < width - 2) 
	{
		// Noise reduction
			NR[i*width+j] =
				 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
				+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
				+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
				+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
				+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
				/159.0;
	}
}

__global__ void canny_Gx_kernel(
	float *NR, float *Gx,
	int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 2 && i < height - 2 && j >= 2 && j < width - 2)
	{
		// Intensity gradient of the image
		Gx[i * width + j] =
			(1.0 * NR[(i - 2) * width + (j - 2)] + 2.0 * NR[(i - 2) * width + (j - 1)] + (-2.0) * NR[(i - 2) * width + (j + 1)] + (-1.0) * NR[(i - 2) * width + (j + 2)] + 4.0 * NR[(i - 1) * width + (j - 2)] + 8.0 * NR[(i - 1) * width + (j - 1)] + (-8.0) * NR[(i - 1) * width + (j + 1)] + (-4.0) * NR[(i - 1) * width + (j + 2)] + 6.0 * NR[(i)*width + (j - 2)] + 12.0 * NR[(i)*width + (j - 1)] + (-12.0) * NR[(i)*width + (j + 1)] + (-6.0) * NR[(i)*width + (j + 2)] + 4.0 * NR[(i + 1) * width + (j - 2)] + 8.0 * NR[(i + 1) * width + (j - 1)] + (-8.0) * NR[(i + 1) * width + (j + 1)] + (-4.0) * NR[(i + 1) * width + (j + 2)] + 1.0 * NR[(i + 2) * width + (j - 2)] + 2.0 * NR[(i + 2) * width + (j - 1)] + (-2.0) * NR[(i + 2) * width + (j + 1)] + (-1.0) * NR[(i + 2) * width + (j + 2)]);
	}
}

__global__ void canny_Gy_kernel(
	float *NR, float *Gy,
	int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 2 && i < height - 2 && j >= 2 && j < width - 2)
	{
		// Intensity gradient of the image
		Gy[i * width + j] =
			((-1.0) * NR[(i - 2) * width + (j - 2)] + (-4.0) * NR[(i - 2) * width + (j - 1)] + (-6.0) * NR[(i - 2) * width + (j)] + (-4.0) * NR[(i - 2) * width + (j + 1)] + (-1.0) * NR[(i - 2) * width + (j + 2)] + (-2.0) * NR[(i - 1) * width + (j - 2)] + (-8.0) * NR[(i - 1) * width + (j - 1)] + (-12.0) * NR[(i - 1) * width + (j)] + (-8.0) * NR[(i - 1) * width + (j + 1)] + (-2.0) * NR[(i - 1) * width + (j + 2)] + 2.0 * NR[(i + 1) * width + (j - 2)] + 8.0 * NR[(i + 1) * width + (j - 1)] + 12.0 * NR[(i + 1) * width + (j)] + 8.0 * NR[(i + 1) * width + (j + 1)] + 2.0 * NR[(i + 1) * width + (j + 2)] + 1.0 * NR[(i + 2) * width + (j - 2)] + 4.0 * NR[(i + 2) * width + (j - 1)] + 6.0 * NR[(i + 2) * width + (j)] + 4.0 * NR[(i + 2) * width + (j + 1)] + 1.0 * NR[(i + 2) * width + (j + 2)]);
	}
}

__global__ void canny_G_kernel(
	float *G, float *Gx, float *Gy,
	int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 2 && i < height - 2 && j >= 2 && j < width - 2)
	{
		// Intensity gradient of the image
		G[i * width + j] = sqrtf((Gx[i * width + j] * Gx[i * width + j]) + (Gy[i * width + j] * Gy[i * width + j])); // G = √Gx²+Gy²
	}
}

__global__ void canny_phi_kernel(
	float *phi, float *Gx, float *Gy,
	int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 2 && i < height - 2 && j >= 2 && j < width - 2)
	{
		// Intensity gradient of the image
		phi[i * width + j] = atan2f(fabs(Gy[i * width + j]), fabs(Gx[i * width + j]));

		float PI = 3.141593;
		if (fabs(phi[i * width + j]) <= PI / 8)
			phi[i * width + j] = 0;
		else if (fabs(phi[i * width + j]) <= 3 * (PI / 8))
			phi[i * width + j] = 45;
		else if (fabs(phi[i * width + j]) <= 5 * (PI / 8))
			phi[i * width + j] = 90;
		else if (fabs(phi[i * width + j]) <= 7 * (PI / 8))
			phi[i * width + j] = 135;
		else
			phi[i * width + j] = 0;
	}
}

__global__ void canny_pedge_kernel(
	uint8_t *pedge, float *phi, float *G,
	int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 2 && i < height - 2 && j >= 2 && j < width - 2)
	{
		// Edge
		pedge[i * width + j] = 0;
		if (phi[i * width + j] == 0)
		{
			if (G[i * width + j] > G[i * width + j + 1] && G[i * width + j] > G[i * width + j - 1]) // edge is in N-S
				pedge[i * width + j] = 1;
		}
		else if (phi[i * width + j] == 45)
		{
			if (G[i * width + j] > G[(i + 1) * width + j + 1] && G[i * width + j] > G[(i - 1) * width + j - 1]) // edge is in NW-SE
				pedge[i * width + j] = 1;
		}
		else if (phi[i * width + j] == 90)
		{
			if (G[i * width + j] > G[(i + 1) * width + j] && G[i * width + j] > G[(i - 1) * width + j]) // edge is in E-W
				pedge[i * width + j] = 1;
		}
		else if (phi[i * width + j] == 135)
		{
			if (G[i * width + j] > G[(i + 1) * width + j - 1] && G[i * width + j] > G[(i - 1) * width + j + 1]) // edge is in NE-SW
				pedge[i * width + j] = 1;
		}
	}
}

__global__ void canny_Umbralimg_kernel(uint8_t *image_out, uint8_t *pedge, float *G, float level, int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 3 && i < height - 3 && j >= 3 && j < width - 3)
	{
		int idx = i * width + j;
		float lowthres = level / 2.0f;
		float hithres = 2.0f * level;

		image_out[idx] = 0;

		if (G[idx] > hithres && pedge[idx])
		{
			image_out[idx] = 255;
		}
		else if (pedge[idx] && G[idx] >= lowthres && G[idx] < hithres)
		{
			// Check 3x3 neighbors
			for (int ii = -1; ii <= 1; ii++)
			{
				for (int jj = -1; jj <= 1; jj++)
				{
					if (G[(i + ii) * width + j + jj] > hithres)
					{
						image_out[idx] = 255;
						break;
					}
				}
				if (image_out[idx] == 255)
				{
					break;
				}
			}
		}
	}
}

//-------------------------------- Kernels del Canny --------------------------------//


//-------------------------------- Kernels del Hough --------------------------------//
#define DEG2RAD 0.017453f
__global__ void init_cos_sin_table_kernel(float *sin_table_GPU, float *cos_table_GPU, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		sin_table_GPU[idx] = sinf(idx * DEG2RAD);
		cos_table_GPU[idx] = cosf(idx * DEG2RAD);
	}
}

__global__ void houghtransform_kernel(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height,
									  float *sin_table, float *cos_table)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int theta;

	if (i >= height || j >= width)
		return;

	float hough_h = ((sqrtf(2.0) * (float)(height > width ? height : width)) / 2.0);

	float center_x = width / 2.0;
	float center_y = height / 2.0;

	if (im[i * width + j] > 250)
	{
		for (theta = 0; theta < 180; theta++)
		{
			float rho = (((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
			atomicAdd(&accumulators[(int)((roundf(rho + hough_h) * 180.0)) + theta], 1);
		}
	}
}
//-------------------------------- Kernels del Hough --------------------------------//


//-------------------------------- CPU Pintar lineas --------------------------------//
void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int rho, theta, ii, jj;
	uint32_t max;

	for(rho=0;rho<accu_height;rho++)
	{
		for(theta=0;theta<accu_width;theta++)  
		{  

			if(accumulators[(rho*accu_width) + theta] >= threshold)  
			{  
				//Is this point a local maxima (9x9)  
				max = accumulators[(rho*accu_width) + theta]; 
				for(int ii=-4;ii<=4;ii++)  
				{  
					for(int jj=-4;jj<=4;jj++)  
					{  
						if( (ii+rho>=0 && ii+rho<accu_height) && (jj+theta>=0 && jj+theta<accu_width) )  
						{  
							if( accumulators[((rho+ii)*accu_width) + (theta+jj)] > max )  
							{
								max = accumulators[((rho+ii)*accu_width) + (theta+jj)];
							}  
						}  
					}  
				}  

				if(max == accumulators[(rho*accu_width) + theta]) //local maxima
				{
					int x1, y1, x2, y2;  
					x1 = y1 = x2 = y2 = 0;  

					if(theta >= 45 && theta <= 135)  
					{
						if (theta>90) {
							//y = (r - x cos(t)) / sin(t)  
							x1 = width/2;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);  
						} else {
							//y = (r - x cos(t)) / sin(t)  
							x1 = 0;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width*2/5;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2); 
						}
					} else {
						//x = (r - y sin(t)) / cos(t);  
						y1 = 0;  
						x1 = ((float)(rho-(accu_height/2)) - ((y1 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
						y2 = height;  
						x2 = ((float)(rho-(accu_height/2)) - ((y2 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
					}
					x1_lines[*lines] = x1;
					y1_lines[*lines] = y1;
					x2_lines[*lines] = x2;
					y2_lines[*lines] = y2;
					(*lines)++;
				}
			}
		}
	}
}
//-------------------------------- CPU Pintar lineas --------------------------------//