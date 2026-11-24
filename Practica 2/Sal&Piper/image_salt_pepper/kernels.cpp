
#include <CL/sycl.hpp>


using  namespace  cl::sycl;

void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width)
{

	buffer<float,1> buff_im{im, range<1>{(size_t)width*height}};
	/*Sumit kernel*/
	Q.submit([&](handler &h) {
		accessor acc_im{buff_im, h, read_write};
		range num_groups= range<2>(height, width);
		range group_size = range<2>(window_size, window_size);

		h.parallel_for_work_group(num_groups, group_size, [=] (group<2> grp) {
			// kernel function is executed once per work-group
			int ib = grp.get_group_id(0);
			int jb = grp.get_group_id(1);
			int ws2 = (window_size-1)>>1;

			grp.parallel_for_work_item([&](h_item<2> it) {
				// kernel function is executed once per work-item
				int i = ib * window_size + it.get_local_id(0);
				int j = jb * window_size + it.get_local_id(1);
				
				if (i < height - ws2)
				{
					if (j < width - ws2)
					{
						// Noise reduction
						float tem[9];

						/* buble_sort */
						int k = 0;
						for(int ii=0; ii < window_size; ii++)
							for(int jj=0; jj < window_size; jj++) {
								tem[k] = acc_im[(i + ii)*width + (j + jj)];
								k++;
							}
						
						int size = window_size*window_size;
						//buble_sort(tem[0], window_size * window_size);
						for (int a=1; a<size; a++)
							for (int b=0 ; b<size - a; b++)
								if (tem[b] > tem[b+1]){
									float tmp = tem[b];
									tem[b] = tem[b+1];
									tem[b+1] = tmp;
								}

						/* buble_sort */

						int mediana = tem[(window_size*window_size-1)>>1];
						if(fabsf((mediana - im[i*width + j])/mediana) <= thredshold) {
							image_out[i*width + j] = im[i*width + j];
						}
						else{
							image_out[i*width + j] = mediana;
						}

					}
				}
			});
		});// End of the kernel function
	}).wait();
}
