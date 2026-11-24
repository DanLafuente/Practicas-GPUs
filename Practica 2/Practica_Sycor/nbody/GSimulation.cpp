/*
    This file is part of the example codes which have been used
    for the "Code Optmization Workshop".
    
    Copyright (C) 2016  Fabio Baruffa <fbaru-dev@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "GSimulation.hpp"
#include "cpu_time.hpp"
#include <CL/sycl.hpp>
using namespace cl::sycl;

GSimulation :: GSimulation()
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(16000); 
  set_nsteps(10);
  set_tstep(0.1); 
  set_sfreq(1);
}

void GSimulation :: set_number_of_particles(int N)  
{
  set_npart(N);
}

void GSimulation :: set_number_of_steps(int N)  
{
  set_nsteps(N);
}

void GSimulation :: init_pos()  
{
  std::random_device rd;	//random number generator
  std::mt19937 gen(42);      
  std::uniform_real_distribution<real_type> unif_d(0,1.0);
  
  for(int i=0; i<get_npart(); ++i)
  {
    particles->pos_x[i] = unif_d(gen);
    particles->pos_y[i] = unif_d(gen);
    particles->pos_z[i] = unif_d(gen);
  }
}

void GSimulation :: init_vel()  
{
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(-1.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles->vel_x[i] = unif_d(gen) * 1.0e-3f;
    particles->vel_y[i] = unif_d(gen) * 1.0e-3f;
    particles->vel_z[i] = unif_d(gen) * 1.0e-3f; 
  }
}

void GSimulation :: init_acc() 
{
  for(int i=0; i<get_npart(); ++i)
  {
    particles->acc_x[i] = 0.f; 
    particles->acc_y[i] = 0.f;
    particles->acc_z[i] = 0.f;
  }
}

void GSimulation :: init_mass() 
{
  real_type n   = static_cast<real_type> (get_npart());
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles->mass[i] = n * unif_d(gen); 
  }
}

/*
*  1.
*/
void GSimulation :: get_acceleration(int n)
{
   int i,j;

   const float softeningSquared = 1e-3f;
   const float G = 6.67259e-11f;


   for (i = 0; i < n; i++)// update acceleration
   {
     real_type ax_i = particles->acc_x[i];
     real_type ay_i = particles->acc_y[i];
     real_type az_i = particles->acc_z[i];
     for (j = 0; j < n; j++)
     {
        real_type dx, dy, dz;
        real_type distanceSqr = 0.0f;
        real_type distanceInv = 0.0f;
            
        dx = particles->pos_x[j] - particles->pos_x[i];	//1flop
        dy = particles->pos_y[j] - particles->pos_y[i];	//1flop	
        dz = particles->pos_z[j] - particles->pos_z[i];	//1flop
        
        distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
        distanceInv = 1.0f / sqrtf(distanceSqr);			//1div+1sqrt

        ax_i += dx * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
        ay_i += dy * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
        az_i += dz * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
     }
     particles->acc_x[i] = ax_i;
     particles->acc_y[i] = ay_i;
     particles->acc_z[i] = az_i;
   }
}
void GSimulation :: get_acceleration_SoA(int n)
{
  const float softeningSquared = 1e-3f;
  const float G = 6.67259e-11f;

  /*Syquel declaration*/
  sycl::device dev;
  dev = sycl::device(sycl::gpu_selector_v);
	sycl::queue Q(dev);

  //Create a submit a kernel
  { 
    const int N = n;
    /*Buffers*/
     buffer<float, 1> buffer_x{particles->acc_x, range<1> {(size_t)N}};
     buffer<float, 1> buffer_y{particles->acc_y, range<1> {(size_t)N}};
     buffer<float, 1> buffer_z{particles->acc_z, range<1> {(size_t)N}};
     buffer<float, 1> buffer_mass{particles->mass, range<1> {(size_t)N}};
    
    Q.submit([&](handler &h) {
      /*Accesor*/
      accessor acc_x{buffer_x, h, read_write};
      accessor acc_y{buffer_y, h, read_write};
      accessor acc_z{buffer_z, h, read_write};
      accessor acc_mass{buffer_mass, h, read_write};

      const int B = 16;
      range num_groups= range<2>(n, n);
		  range group_size = range<2>(B, B);

      h.parallel_for_work_group(num_groups, group_size, [=] (group<2> grp) {

          int ib = grp.get_group_id(0);
			    int jb = grp.get_group_id(1);

          real_type ax_i = acc_x[ib];
          real_type ay_i = acc_y[ib];
          real_type az_i = acc_z[ib];

          grp.parallel_for_work_item([&](h_item<2> it) {

            int i = ib * B + it.get_local_id(0);
				    int j = jb * B + it.get_local_id(1);

            
            
                real_type dx, dy, dz;
                real_type distanceSqr = 0.0f;
                real_type distanceInv = 0.0f;
                    
                dx = acc_x[j] - acc_x[i];	//1flop
                dy = acc_y[j] - acc_y[i];	//1flop	
                dz = acc_z[j] - acc_z[i];	//1flop
                
                distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
                distanceInv = 1.0f / sqrtf(distanceSqr);			//1div+1sqrt

                ax_i += dx * G * acc_mass[j] * distanceInv * distanceInv * distanceInv; //6flops
                ay_i += dy * G * acc_mass[j] * distanceInv * distanceInv * distanceInv; //6flops
                az_i += dz * G * acc_mass[j] * distanceInv * distanceInv * distanceInv; //6flops
            
            
          });
          acc_x[ib] = ax_i;
          acc_y[ib] = ay_i;
          acc_z[ib] = az_i;
      });
     }).wait();
   }
}

/*
*  2.
*/
real_type GSimulation :: updateParticles(int n, real_type dt)
{
  
  int i;
  real_type energy = 0;

  for (i = 0; i < n; ++i)// update position
  {
    particles->vel_x[i] += particles->acc_x[i] * dt; //2flops
    particles->vel_y[i] += particles->acc_y[i] * dt; //2flops
    particles->vel_z[i] += particles->acc_z[i] * dt; //2flops
    
    particles->pos_x[i] += particles->vel_x[i] * dt; //2flops
    particles->pos_y[i] += particles->vel_y[i] * dt; //2flops
    particles->pos_z[i] += particles->vel_z[i] * dt; //2flops

    particles->acc_x[i] = 0.;
    particles->acc_y[i] = 0.;
    particles->acc_z[i] = 0.;

    energy += particles->mass[i] * (
        particles->vel_x[i]*particles->vel_x[i] + 
              particles->vel_y[i]*particles->vel_y[i] +
              particles->vel_z[i]*particles->vel_z[i]); //7flops
  
  return energy;
}
}
real_type GSimulation :: updateParticles_SoA(int n, real_type dt)
{
  /*Syquel declaration*/
  sycl::device dev;
  dev = sycl::device(sycl::gpu_selector_v);
	sycl::queue Q(dev);

  
  {
    real_type energy = 0;
    const int N = n, A = 1;
    /*Buffers*/
    buffer<float, 1> buffer_vel_x{particles->vel_x, range<1> {(size_t)N}};
    buffer<float, 1> buffer_vel_y{particles->vel_y, range<1> {(size_t)N}};
    buffer<float, 1> buffer_vel_z{particles->vel_z, range<1> {(size_t)N}};

    buffer<float, 1> buffer_pos_x{particles->pos_x, range<1> {(size_t)N}};
    buffer<float, 1> buffer_pos_y{particles->pos_y, range<1> {(size_t)N}};
    buffer<float, 1> buffer_pos_z{particles->pos_z, range<1> {(size_t)N}};

    buffer<float, 1> buffer_acc_x{particles->acc_x, range<1> {(size_t)N}};
    buffer<float, 1> buffer_acc_y{particles->acc_y, range<1> {(size_t)N}};
    buffer<float, 1> buffer_acc_z{particles->acc_z, range<1> {(size_t)N}};
    
    buffer<float, 1> buffer_mass{particles->mass, range<1> {(size_t)N}};

    buffer<real_type, 1> buffer_energy{&energy, range<1> {(size_t)(1)}};

    Q.submit([&](handler &h) {
      /*Accesor*/
      accessor acc_vel_x{buffer_vel_x, h, read_write};
      accessor acc_vel_y{buffer_vel_y, h, read_write};
      accessor acc_vel_z{buffer_vel_z, h, read_write};

      accessor acc_pos_x{buffer_pos_x, h, read_write};
      accessor acc_pos_y{buffer_pos_y, h, read_write};
      accessor acc_pos_z{buffer_pos_z, h, read_write};

      accessor acc_acc_x{buffer_acc_x, h, read_write};
      accessor acc_acc_y{buffer_acc_y, h, read_write};
      accessor acc_acc_z{buffer_acc_z, h, read_write};

      accessor acc_mass{buffer_mass, h, read_write};

      auto acc_energy = buffer_energy.get_access<sycl::mode::read_write(h)>;

      h.parallel_for(n, [=](id<1> i) {
      
        acc_vel_x[i] += acc_acc_x[i] * dt; //2flops
        acc_vel_y[i] += acc_acc_y[i] * dt; //2flops
        acc_vel_z[i] += acc_acc_z[i] * dt; //2flops
        
        acc_pos_x[i] += acc_vel_x[i] * dt; //2flops
        acc_pos_y[i] += acc_vel_y[i] * dt; //2flops
        acc_pos_z[i] += acc_vel_z[i] * dt; //2flops

        acc_acc_x[i] = 0.;
        acc_acc_y[i] = 0.;
        acc_acc_z[i] = 0.;

        acc_energy[0] += acc_mass[i] * (
            acc_vel_x[i]*acc_vel_x[i] + 
                  acc_vel_y[i]*acc_vel_y[i] +
                 acc_vel_z[i]*acc_vel_z[i]); //7flops
      });
    }).wait();
    return energy;
  }
  
}

void GSimulation :: start() 
{
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();
  
  //allocate particles
  particles = new ParticleSoA;
  
  particles->pos_x = new real_type[n];
  particles->pos_y = new real_type[n];
  particles->pos_z = new real_type[n];
  particles->vel_x = new real_type[n];
  particles->vel_y = new real_type[n];
  particles->vel_z = new real_type[n];
  particles->acc_x = new real_type[n];
  particles->acc_y = new real_type[n];
  particles->acc_z = new real_type[n];
  particles->mass  = new real_type[n]; 

  init_pos();	
  init_vel();
  init_acc();
  init_mass();
  
  print_header();
  
  _totTime = 0.; 
  
  
  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  double nd = double(n);
  double gflops = 1e-9 * ( (11. + 18. ) * nd*nd  +  nd * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;
  
  const double t0 = time.start();
  for (int s=1; s<=get_nsteps(); ++s)
  {   
   ts0 += time.start(); 


    get_acceleration_SoA(n);

    energy = updateParticles(n, dt);
    _kenergy = 0.5 * energy; 
    
    ts1 += time.stop();
    if(!(s%get_sfreq()) ) 
    {
      nf += 1;      
      std::cout << " " 
		<<  std::left << std::setw(8)  << s
		<<  std::left << std::setprecision(5) << std::setw(8)  << s*get_tstep()
		<<  std::left << std::setprecision(5) << std::setw(12) << _kenergy
		<<  std::left << std::setprecision(5) << std::setw(12) << (ts1 - ts0)
		<<  std::left << std::setprecision(5) << std::setw(12) << gflops*get_sfreq()/(ts1 - ts0)
		<<  std::endl;
      if(nf > 2) 
      {
	av  += gflops*get_sfreq()/(ts1 - ts0);
	dev += gflops*get_sfreq()*gflops*get_sfreq()/((ts1-ts0)*(ts1-ts0));
      }
      
      ts0 = 0;
      ts1 = 0;
    }
  
  } //end of the time step loop
  
  const double t1 = time.stop();
  _totTime  = (t1-t0);
  _totFlops = gflops*get_nsteps();
  
  av/=(double)(nf-2);
  dev=sqrt(dev/(double)(nf-2)-av*av);
  

  std::cout << std::endl;
  std::cout << "# Total Time (s)      : " << _totTime << std::endl;
  std::cout << "# Average Performance : " << av << " +- " <<  dev << std::endl;
  std::cout << "===============================" << std::endl;

}


void GSimulation :: print_header()
{
	    
  std::cout << " nPart = " << get_npart()  << "; " 
	    << "nSteps = " << get_nsteps() << "; " 
	    << "dt = "     << get_tstep()  << std::endl;
	    
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " " 
	    <<  std::left << std::setw(8)  << "s"
	    <<  std::left << std::setw(8)  << "dt"
	    <<  std::left << std::setw(12) << "kenergy"
	    <<  std::left << std::setw(12) << "time (s)"
	    <<  std::left << std::setw(12) << "GFlops"
	    <<  std::endl;
  std::cout << "------------------------------------------------" << std::endl;


}

GSimulation :: ~GSimulation()
{
  delete [] particles->pos_x;
  delete [] particles->pos_y;
  delete [] particles->pos_z;
  delete [] particles->vel_x;
  delete [] particles->vel_y;
  delete [] particles->vel_z;
  delete [] particles->acc_x;
  delete [] particles->acc_y;
  delete [] particles->acc_z;
  delete [] particles->mass;
  delete particles;
}
