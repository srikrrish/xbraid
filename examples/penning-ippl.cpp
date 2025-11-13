/*BHEADER**********************************************************************
 * Copyright (c) 2013, Lawrence Livermore National Security, LLC. 
 * Produced at the Lawrence Livermore National Laboratory. Written by 
 * Jacob Schroder, Rob Falgout, Tzanio Kolev, Ulrike Yang, Veselin 
 * Dobrev, et al. LLNL-CODE-660355. All rights reserved.
 * 
 * This file is part of XBraid. For support, post issues to the XBraid Github page.
 * 
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the terms and conditions of the GNU General Public
 * License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc., 59
 * Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 ***********************************************************************EHEADER*/

#include <stdlib.h>
//#include <stdio.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <set>

#include<Kokkos_Random.hpp>

#include <random>
#include "braid.hpp"
#include "ipplDefs.hpp"
#include "Utility/IpplTimings.h"

// --------------------------------------------------------------------------
// User-defined routines and objects 
// --------------------------------------------------------------------------

template <typename T>
struct Newton1D {

  double tol = 1e-12;
  int max_iter = 20;
  double pi = std::acos(-1.0);
  
  T mu, sigma, u;

  KOKKOS_INLINE_FUNCTION
  Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  Newton1D(const T& mu_, const T& sigma_, 
           const T& u_) 
  : mu(mu_), sigma(sigma_), u(u_) {}

  KOKKOS_INLINE_FUNCTION
  ~Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  T f(T& x) {
      T F;
      F = std::erf((x - mu)/(sigma * std::sqrt(2.0))) 
          - 2 * u + 1;
      return F;
  }

  KOKKOS_INLINE_FUNCTION
  T fprime(T& x) {
      T Fprime;
      Fprime = (1 / sigma) * std::sqrt(2 / pi) * 
               std::exp(-0.5 * (std::pow(((x - mu) / sigma),2)));
      return Fprime;
  }

  KOKKOS_FUNCTION
  void solve(T& x) {
      int iterations = 0;
      while ((iterations < max_iter) && (std::fabs(f(x)) > tol)) {
          x = x - (f(x)/fprime(x));
          iterations += 1;
      }
  }
};


template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {

  using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
  using value_type  = typename T::value_type;
  // Output View for the random numbers
  view_type x, v;

  // The GeneratorPool
  GeneratorPool rand_pool;

  T mu, sigma, minU, maxU;

  double pi = std::acos(-1.0);

  // Initialize all members
  generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_,
                  T& mu_, T& sigma_, T& minU_, T& maxU_)
      : x(x_), v(v_), rand_pool(rand_pool_), 
        mu(mu_), sigma(sigma_), minU(minU_), maxU(maxU_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    value_type u;
    for (unsigned d = 0; d < Dim; ++d) {
        u = rand_gen.drand(minU[d], maxU[d]);
        x(i)[d] = (std::sqrt(pi / 2) * (2 * u - 1)) * 
                  sigma[d] + mu[d];
        Newton1D<value_type> solver(mu[d], sigma[d], u);
        solver.solve(x(i)[d]);
        v(i)[d] = rand_gen.normal(0.0, 1.0);
    }

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

double CDF(const double& x, const double& mu, const double& sigma) {
   double cdf = 0.5 * (1.0 + std::erf((x - mu)/(sigma * std::sqrt(2))));
   return cdf;
}



// Define BraidVector, can contain anything, and be named anything
// --> Put all time-dependent information here
template<class PLayout>
class BraidVector : public ippl::ParticleBase<PLayout>
{
public:
   ParticleAttrib<double>     q; // charge
   typename ippl::ParticleBase<PLayout>::particle_position_type P;  // G(P^(k)_n)
   typename ippl::ParticleBase<PLayout>::particle_position_type E;  // electric field at particle position

   BraidVector(PLayout& pl)
   : ippl::ParticleBase<PLayout>(pl)
   {
       // register the particle attributes
       this->addAttribute(q);
       this->addAttribute(P);
       this->addAttribute(E);
   }

   // Deconstructor
   virtual ~BraidVector() {};

};


// Wrapper for BRAID's App object 
// --> Put all time INDEPENDENT information here
class MyBraidApp : public BraidApp
{
protected:
   // BraidApp defines tstart, tstop, ntime and comm_t

private:
   int next_id_ = 1;
   std::map<void*, int> buffer_ptr_to_id_;  // raw ptr to id mapping
   std::map<int, ippl::Communicate::buffer_type> buffer_pool_;  // Pool: unique id → list of buffers

public:

   CxField_t rhoPIF_m;
   Field_t Sk_m;
   Field_t rhoPIC_m;
   VField_t EfieldPIC_m;

   ippl::e_dim_tag decomp_m[3];

   Vector_t hrPIC_m;
   Vector_t hrPIF_m;
   Vector_t rmin_m;
   Vector_t rmax_m;
   Vector_t length_m;

   Vector_t mu_m;
   Vector_t sd_m;

   Vector_i nrPIC_m;
   Vector_i nmPIF_m;

   double Q_m;
   double Bext_m;

   size_type Np_m;
   size_type nloc_m;
   
   std::shared_ptr<Solver_t> solver_mp;
   
   double time_m;

   std::string shapetype_m;

   std::string coarsetype_m;

   PLayout_t PL_m;

   int shapedegree_m;

   double dtFine_m;
   double dtCoarse_m;
   double dtSlice_m;

   double coarseTol_m, fineTol_m;
   double cfactortime_m, cfactorspace_m;

   MPI_Comm spaceComm, timeComm;

   size_type bufSize_m;


   std::vector<std::shared_ptr<ippl::FFT<ippl::NUFFTransform, 3, double>>> nufftType1_m;
   std::vector<std::shared_ptr<ippl::FFT<ippl::NUFFTransform, 3, double>>> nufftType2_m;
   //BraidVector<PLayout_t> utemp(PLayout_t);
   
   BraidVector<PLayout_t> utemp{PL_m};
   // We will need the MPI Rank
   int rank, rankSpace, rankTime;
   int num_procs, sizeSpace, sizeTime;

   // Constructor 
   MyBraidApp(MPI_Comm comm_t_, MPI_Comm &comm_s_, int rank_, int rankSpace_, 
                       int rankTime_, int sizeSpace_, int sizeTime_, int num_procs_, 
                       double tstart_, double tstop_, int ntime_, 
                       Vector_i nmPIF, Vector_i nrPIC, Vector_t rmin, Vector_t rmax, 
                       size_type Np, double dtFine, 
                       double dtCoarse, double dtSlice, std::string& coarsetype, std::string& shapetype,
                       int shapedegree, double coarseTol, double fineTol, double cfactortime, double cfactorspace);


   // Deconstructor
   virtual ~MyBraidApp() {};

   void initFFTSolver() {
       ippl::ParameterList sp;
       sp.add("output_type", Solver_t::GRAD);
       sp.add("use_heffte_defaults", false);  
       sp.add("use_pencils", true);  
       sp.add("use_reorder", false);  
       sp.add("use_gpu_aware", true);  
       sp.add("comm", ippl::p2p_pl);  
       sp.add("r2c_direction", 0);  

       solver_mp = std::make_shared<Solver_t>();

       solver_mp->mergeParameters(sp);

       solver_mp->setRhs(rhoPIC_m);

       solver_mp->setLhs(EfieldPIC_m);
   }
   
   void initNUFFTs(FieldLayout_t& FLPIF, int numLevels) {

        std::vector<ippl::ParameterList> fftParamsPerLevel;
       
        fftParamsPerLevel.resize(numLevels);
        nufftType1_m.resize(numLevels);
        nufftType2_m.resize(numLevels);

        for (int level = 0; level < numLevels; ++level) {
            auto& plist = fftParamsPerLevel[level];
        
            //Example: vary tolerance by level
            double coarseTol = coarseTol_m * std::pow(cfactorspace_m, level);
            double tol = (level == 0) ? fineTol_m : coarseTol;
            plist.add("tolerance", tol);
#ifdef GPU_BUILD
            plist.add("gpu_method", 2);
            plist.add("gpu_sort", 0);
            plist.add("gpu_kerevalmeth", 1);
            plist.add("gpu_binsizex", 8);
            plist.add("gpu_binsizey", 8);
            plist.add("gpu_binsizez", 2);
            plist.add("gpu_maxsubprobsize", 1024);
#else
            
            plist.add("spread_kerevalmeth", 1);
            plist.add("spread_sort", 2);
            plist.add("nthreads", 0);
#endif
            plist.add("use_finufft_defaults", false);

            nufftType1_m[level] = std::make_shared<ippl::FFT<ippl::NUFFTransform, 3, double>>(FLPIF, nloc_m, 1, plist);
            nufftType2_m[level] = std::make_shared<ippl::FFT<ippl::NUFFTransform, 3, double>>(FLPIF, nloc_m, 2, plist);
        }
   }

   void initializeShapeFunctionPIF() {

       using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
       auto Skview = Sk_m.getView();
       auto N = nmPIF_m;
       const int nghost = Sk_m.getNghost();
       const Mesh_t& mesh = rhoPIF_m.get_mesh();
       const Vector_t& dx = mesh.getMeshSpacing();
       const Vector_t& Len = rmax_m - rmin_m;
       const double pi = std::acos(-1.0);
       int order = shapedegree_m + 1;
       
       if(shapetype_m == "Gaussian") {

           throw IpplException("initializeShapeFunctionPIF",
                               "Gaussian shape function not implemented yet");

       }
       else if(shapetype_m == "B-spline") {

           Kokkos::parallel_for("B-spline shape functions",
                               mdrange_type({0, 0, 0},
                                            {N[0], N[1], N[2]}),
                               KOKKOS_LAMBDA(const int i,
                                             const int j,
                                             const int k)
           {
               
               Vector<int, 3> iVec = {i, j, k};
               Vector<double, 3> kVec;
               double Sk = 1.0;
               for(size_t d = 0; d < Dim; ++d) {
                   kVec[d] = 2 * pi / Len[d] * (iVec[d] - (N[d] / 2));
                   double khbytwo = kVec[d] * dx[d] / 2;
                   bool isNotZero = (khbytwo != 0.0);
                   double factor = (1.0 / (khbytwo + ((!isNotZero) * 1.0)));
                   double arg = isNotZero * (Kokkos::sin(khbytwo) * factor) + 
                                (!isNotZero) * 1.0;
                   //Fourier transform of CIC
                   Sk *= std::pow(arg, order);
               }
                   Skview(i+nghost, j+nghost, k+nghost) = Sk;
           });
       }
       else {
           throw IpplException("initializeShapeFunctionPIF",
                               "Unrecognized shape function type");
       }

   }

   void BorisPIF(BraidVector<PLayout_t>& u, const double& dt, const unsigned int& nt, const int& level) {
    
        //BraidVector *u = (BraidVector*) u_;
        //PLayout_t& PL = u.getLayout();
        u.setParticleBC(ippl::BC::PERIODIC);
        auto &Rtemp = u.R;
        auto &Ptemp = u.P;
        auto &q = u.q;
        auto &E = u.E;
        //if(level > 0) {
        rhoPIF_m = {0.0, 0.0};
        q = Q_m / Np_m;
        PL_m.applyBC(Rtemp, PL_m.getRegionLayout().getDomain());
        scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1_m[level].get(), spaceComm);

        rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
        // Solve for and gather E field
        gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2_m[level].get(), q);
        //}

        //Reset the value of q here as we used it as a temporary object in gather to 
        //save memory
        q = Q_m / Np_m;
    
        double alpha = -0.5 * dt;
        double DrInv = 1.0 / (1 + (std::pow((alpha * Bext_m), 2)));
        Vector_t rmax = rmax_m;
        double Bext = Bext_m;
        size_type nloc = nloc_m;
        for (unsigned int it=0; it<nt; it++) {
            // kick
            // Staggered Leap frog or Boris algorithm as per 
            // https://www.sciencedirect.com/science/article/pii/S2590055219300526
            // eqns 4(a)-4(c). Note we don't use the Boris trick here and do
            // the analytical matrix inversion which is not complex in this case.
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            // kick
            auto Rview = Rtemp.getView();
            auto Pview = Ptemp.getView();
            auto Eview = E.getView();
            double V0 = 30*rmax[2];
            Kokkos::parallel_for("Kick1", nloc,
                                  KOKKOS_LAMBDA(const size_t j){
                double Eext_x = -(Rview(j)[0] - 0.5*rmax[0]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_y = -(Rview(j)[1] - 0.5*rmax[1]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_z =  (Rview(j)[2] - 0.5*rmax[2]) * (V0/(std::pow(rmax[2],2)));

                Eext_x += Eview(j)[0];
                Eext_y += Eview(j)[1];
                Eext_z += Eview(j)[2];
                
                Pview(j)[0] += alpha * (Eext_x  + Pview(j)[1] * Bext);
                Pview(j)[1] += alpha * (Eext_y  - Pview(j)[0] * Bext);
                Pview(j)[2] += alpha * Eext_z;
            });
    
            //drift
            Rtemp = Rtemp + dt * Ptemp;
    
            //Apply particle BC
            PL_m.applyBC(Rtemp, PL_m.getRegionLayout().getDomain());
            //scatter the charge onto the underlying grid
            rhoPIF_m = {0.0, 0.0};
            
            scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1_m[level].get(), spaceComm);
    
    
            rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
            // Solve for and gather E field
            gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2_m[level].get(), q);

            q = Q_m / Np_m;

            //kick
            auto R2view = Rtemp.getView();
            auto P2view = Ptemp.getView();
            auto E2view = E.getView();
            Kokkos::parallel_for("Kick2", nloc,
                                  KOKKOS_LAMBDA(const size_t j){
                double Eext_x = -(R2view(j)[0] - 0.5*rmax[0]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_y = -(R2view(j)[1] - 0.5*rmax[1]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_z =  (R2view(j)[2] - 0.5*rmax[2]) * (V0/(std::pow(rmax[2],2)));

                Eext_x += E2view(j)[0];
                Eext_y += E2view(j)[1];
                Eext_z += E2view(j)[2];
                
                P2view(j)[0]  = DrInv * ( P2view(j)[0] + alpha * (Eext_x 
                                + P2view(j)[1] * Bext + alpha * Bext * Eext_y) );
                P2view(j)[1]  = DrInv * ( P2view(j)[1] + alpha * (Eext_y 
                                - P2view(j)[0] * Bext - alpha * Bext * Eext_x) );
                P2view(j)[2] += alpha * Eext_z;
            });
        }
    }


    void BorisPIC(BraidVector<PLayout_t>& u, const double& dt, const unsigned int& nt) {
    
        //BraidVector *u = (BraidVector*) u_;
        //PLayout_t& PL = u.getLayout();
        u.setParticleBC(ippl::BC::PERIODIC);
        auto &Rtemp = u.R;
        auto &Ptemp = u.P;
        auto &q = u.q;
        auto &E = u.E;
        rhoPIC_m = 0.0;
        q = Q_m / Np_m;
        PL_m.applyBC(Rtemp, PL_m.getRegionLayout().getDomain());
        scatter(q, rhoPIC_m, Rtemp, spaceComm);
    
        rhoPIC_m = rhoPIC_m / (hrPIC_m[0] * hrPIC_m[1] * hrPIC_m[2]);
        rhoPIC_m = rhoPIC_m - (Q_m/((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2])));
    
        //Field solve
        solver_mp->solve();
    
        // gather E field
        gather(E, EfieldPIC_m, Rtemp);
    
        double alpha = -0.5 * dt;
        double DrInv = 1.0 / (1 + (std::pow((alpha * Bext_m), 2)));
        Vector_t rmax = rmax_m;
        double Bext = Bext_m;
        size_type  nloc = nloc_m;
        for (unsigned int it=0; it<nt; it++) {
            // kick
            // Staggered Leap frog or Boris algorithm as per 
            // https://www.sciencedirect.com/science/article/pii/S2590055219300526
            // eqns 4(a)-4(c). Note we don't use the Boris trick here and do
            // the analytical matrix inversion which is not complex in this case.
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            // kick
            auto Rview = Rtemp.getView();
            auto Pview = Ptemp.getView();
            auto Eview = E.getView();
            double V0 = 30*rmax[2];
            Kokkos::parallel_for("Kick1", nloc,
                                  KOKKOS_LAMBDA(const size_t j){
                double Eext_x = -(Rview(j)[0] - 0.5*rmax[0]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_y = -(Rview(j)[1] - 0.5*rmax[1]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_z =  (Rview(j)[2] - 0.5*rmax[2]) * (V0/(std::pow(rmax[2],2)));

                Eext_x += Eview(j)[0];
                Eext_y += Eview(j)[1];
                Eext_z += Eview(j)[2];
                
                Pview(j)[0] += alpha * (Eext_x  + Pview(j)[1] * Bext);
                Pview(j)[1] += alpha * (Eext_y  - Pview(j)[0] * Bext);
                Pview(j)[2] += alpha * Eext_z;
            });
    
            //drift
            Rtemp = Rtemp + dt * Ptemp;
    
            //Apply particle BC
            PL_m.applyBC(Rtemp, PL_m.getRegionLayout().getDomain());
    
            //scatter the charge onto the underlying grid
            rhoPIC_m = 0.0;
            scatter(q, rhoPIC_m, Rtemp, spaceComm);
    
            rhoPIC_m = rhoPIC_m / (hrPIC_m[0] * hrPIC_m[1] * hrPIC_m[2]);
            rhoPIC_m = rhoPIC_m - (Q_m/((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2])));
    
            //Field solve
            solver_mp->solve();
    
            // gather E field
            gather(E, EfieldPIC_m, Rtemp);
    
            //kick
            auto R2view = Rtemp.getView();
            auto P2view = Ptemp.getView();
            auto E2view = E.getView();
            Kokkos::parallel_for("Kick2", nloc,
                                  KOKKOS_LAMBDA(const size_t j){
                double Eext_x = -(R2view(j)[0] - 0.5*rmax[0]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_y = -(R2view(j)[1] - 0.5*rmax[1]) * (V0/(2*std::pow(rmax[2],2)));
                double Eext_z =  (R2view(j)[2] - 0.5*rmax[2]) * (V0/(std::pow(rmax[2],2)));
         
                Eext_x += E2view(j)[0];
                Eext_y += E2view(j)[1];
                Eext_z += E2view(j)[2];
                
                P2view(j)[0]  = DrInv * ( P2view(j)[0] + alpha * (Eext_x 
                                + P2view(j)[1] * Bext + alpha * Bext * Eext_y) );
                P2view(j)[1]  = DrInv * ( P2view(j)[1] + alpha * (Eext_y 
                                - P2view(j)[0] * Bext - alpha * Bext * Eext_x) );
                P2view(j)[2] += alpha * Eext_z;
            });
        }
    }

    int PrintParticle(BraidVector<PLayout_t>& u)
    {
       //BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;
    
       typename ParticleAttrib<Vector_t>::HostMirror R_host = u.R.getHostMirror();
       typename ParticleAttrib<Vector_t>::HostMirror P_host = u.P.getHostMirror();
       Kokkos::deep_copy(R_host, u.R.getView());
       Kokkos::deep_copy(P_host, u.P.getView());
       std::stringstream pname;
       pname << "data/Particle_";
       pname << Ippl::Comm->rank();
       pname << ".csv";
       Inform pcsvout(NULL, pname.str().c_str(), Inform::APPEND, Ippl::Comm->rank());
       pcsvout.precision(10);
       pcsvout.setf(std::ios::scientific, std::ios::floatfield);
       pcsvout << "R_x, R_y, R_z, V_x, V_y, V_z" << endl;
       for (size_type i = 0; i< nloc_m; i++) {
           pcsvout << R_host(i)[0] << " "
                   << R_host(i)[1] << " "
                   << R_host(i)[2] << " "
                   << P_host(i)[0] << " "
                   << P_host(i)[1] << " "
                   << P_host(i)[2] << endl;
       }
       //Ippl::Comm->barrier();
       return 0;
    }


   // Define all the Braid Wrapper routines
   // Note: braid_Vector == BraidVector*
   virtual int Step(braid_Vector    u_,
                    braid_Vector    ustop_,
                    braid_Vector    fstop_,
                    BraidStepStatus &pstatus);

   virtual int Clone(braid_Vector  u_,
                     braid_Vector *v_ptr);

   virtual int Init(double        t,
                    braid_Vector *u_ptr);

   virtual int Free(braid_Vector u_);

   virtual int Sum(double       alpha,
                   braid_Vector x_,
                   double       beta,
                   braid_Vector y_);

   virtual int SpatialNorm(braid_Vector  u_,
                           double       *norm_ptr);

   virtual int BufSize(int *size_ptr,
                       BraidBufferStatus  &status);

   virtual int BufPack(braid_Vector  u_,
                       void         *buffer,
                       BraidBufferStatus  &status);

   virtual int BufUnpack(void         *buffer,
                         braid_Vector *u_ptr,
                         BraidBufferStatus  &status);

   virtual int Access(braid_Vector       u_,
                      BraidAccessStatus &astatus);

   virtual int BufAlloc(void              **buffer,
                        int               nbytes,
                        BraidBufferStatus &bstatus);

   virtual braid_Int BufFree(void          **buffer);


   // Not needed in this example
   virtual int Residual(braid_Vector     u_,
                        braid_Vector     r_,
                        BraidStepStatus &pstatus) { return 0; }

   // Not needed in this example
   virtual int Coarsen(braid_Vector   fu_,
                       braid_Vector  *cu_ptr,
                       BraidCoarsenRefStatus &status) { return 0; }

   // Not needed in this example
   virtual int Refine(braid_Vector   cu_,
                      braid_Vector  *fu_ptr,
                      BraidCoarsenRefStatus &status)  { return 0; }

};

// Braid App Constructor
MyBraidApp::MyBraidApp(MPI_Comm comm_t_, MPI_Comm &comm_s_, int rank_, int rankSpace_, 
                       int rankTime_, int sizeSpace_, int sizeTime_, int num_procs_, 
                       double tstart_, double tstop_, int ntime_, 
                       Vector_i nmPIF, Vector_i nrPIC, Vector_t rmin, Vector_t rmax, 
                       size_type Np, double dtFine, 
                       double dtCoarse, double dtSlice, std::string& coarsetype, std::string& shapetype,
                       int shapedegree, double coarseTol, double fineTol, double cfactortime, double cfactorspace) : BraidApp(comm_t_, tstart_, tstop_, ntime_)
{
   timeComm = comm_t_;
   spaceComm = comm_s_;
   rank = rank_;
   rankSpace = rankSpace_;
   rankTime = rankTime_;
   sizeSpace = sizeSpace_;
   sizeTime = sizeTime_;
   rmin_m = rmin;
   rmax_m = rmax;
   length_m = rmax_m - rmin_m;
   Q_m = -1562.5;
   Bext_m = 5.0;
   nmPIF_m = nmPIF;
   nrPIC_m = nrPIC;
   sd_m[0] = 0.10*20.0;//length[0];
   sd_m[1] = 0.05*20.0;//length[1];
   sd_m[2] = 0.15*20.0;//length[2];
   for (unsigned d = 0; d < 3; d++) {
        hrPIC_m[d] = length_m[d] / nrPIC_m[d];
        hrPIF_m[d] = length_m[d] / nmPIF_m[d];
        mu_m[d] = 0.5 * length_m[d];
   }
   Np_m = Np;
   double factor = 1.0 / sizeSpace;
   nloc_m = (size_type)(factor * Np_m);
   dtFine_m = dtFine;
   dtCoarse_m = dtCoarse;
   dtSlice_m = dtSlice;
   coarsetype_m = coarsetype;
   shapetype_m = shapetype;
   shapedegree_m = shapedegree;
   coarseTol_m = coarseTol;
   fineTol_m = fineTol;
   cfactortime_m = cfactortime;
   cfactorspace_m = cfactorspace;
}

// 
int MyBraidApp::Step(braid_Vector    u_,
                     braid_Vector    ustop_,
                     braid_Vector    fstop_,
                     BraidStepStatus &pstatus)
{
   
   static IpplTimings::TimerRef finePropagator = IpplTimings::getTimer("finePropagator");
   static IpplTimings::TimerRef coarsePropagator = IpplTimings::getTimer("coarsePropagator");
   BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;
   double tstart;             // current time
   double tstop;              // evolve to this time

   // Get time step information
   pstatus.GetTstartTstop(&tstart, &tstop);

   //unsigned int ntFine = 1;//std::ceil((tstop - tstart) / dtFine_m);
   //unsigned int ntCoarse = std::ceil((tstop - tstart) / dtCoarse_m);
   //unsigned int ntCoarse = std::ceil(dtSlice_m / dtCoarse_m);
   unsigned int ntCoarse;// = std::ceil(dtSlice_m / dtCoarse_m);
   unsigned int ntFine = std::ceil(dtSlice_m / dtFine_m);

   //unsigned int ntFine = 1;
   //unsigned int ntCoarse = 1;

   //double dt = tstop - tstart;
   //dtFine_m = dt;
   //double dtCoarselevel = dt;

   int level;
   int max_levels;
   pstatus.GetLevel(&level);
   pstatus.GetNLevels(&max_levels);

   //std::cout << "Rank: " << Ippl::Comm->rank() << " Max levels: " << max_levels << std::endl;
   if(max_levels == 1) {
        IpplTimings::startTimer(finePropagator);
        BorisPIF(*u, dtFine_m, ntFine, level);
        IpplTimings::stopTimer(finePropagator);
   }
   else {
        double dtCoarselevel = dtCoarse_m * std::pow(cfactortime_m, level);
        ntCoarse = std::ceil(dtSlice_m / dtCoarselevel);
        if(coarsetype_m == "PIF") {
            if(level == 0) {
                IpplTimings::startTimer(finePropagator);
                 BorisPIF(*u, dtFine_m, ntFine, level);
                 IpplTimings::stopTimer(finePropagator);
            }
            else {
                IpplTimings::startTimer(coarsePropagator);
                 BorisPIF(*u, dtCoarselevel, ntCoarse, level);
                IpplTimings::stopTimer(coarsePropagator);
            }
        }
        else if(coarsetype_m == "PIC") {
            if(level == 0) {
                IpplTimings::startTimer(finePropagator);
                 BorisPIF(*u, dtFine_m, ntFine, level);
                 IpplTimings::stopTimer(finePropagator);
            }
            else if((level > 0) && (level < (max_levels-1))) {
                IpplTimings::startTimer(coarsePropagator);
                 BorisPIF(*u, dtCoarselevel, ntCoarse, level);
                IpplTimings::stopTimer(coarsePropagator);
            }
            else {
                IpplTimings::startTimer(coarsePropagator);
                 BorisPIC(*u, dtCoarselevel, ntCoarse);  
                IpplTimings::stopTimer(coarsePropagator);
            }
        }
   }

   // no refinement
   pstatus.SetRFactor(1);
   
   return 0;

}

int MyBraidApp::Init(double        t,
                       braid_Vector *u_ptr)
{
   BraidVector<PLayout_t> *u = new BraidVector<PLayout_t>(PL_m);
   static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
   IpplTimings::startTimer(particleCreation);
   u->create(nloc_m);


   if (t != tstart)
   {
      u->R = 0.0;
      u->P = 0.0;
      u->q = 0.0;
      u->E = 0.0;
   }
   else
   {
     Vector_t minU, maxU;
     for (unsigned d = 0; d <Dim; ++d) {
        minU[d] = CDF(rmin_m[d], mu_m[d], sd_m[d]);
        maxU[d]   = CDF(rmax_m[d], mu_m[d], sd_m[d]);
      }
      Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100*rankSpace));
      Kokkos::parallel_for(nloc_m, generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                           u->R.getView(), u->P.getView(), rand_pool64, mu_m, sd_m, minU, maxU));
    
      Kokkos::fence();
      u->q = Q_m / Np_m;
      //rhoPIF_m = {0.0, 0.0};
      //scatterPIFNUFFT(u->q, rhoPIF_m, Sk_m, u->R, nufftType1_m[0].get(), spaceComm);

      //rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
      //// Solve for and gather E field
      //gatherPIFNUFFT(u->E, rhoPIF_m, Sk_m, u->R, nufftType2_m[0].get(), u->q);
      //u->q = Q_m / Np_m;

   }

   u->setParticleBC(ippl::BC::PERIODIC);
   *u_ptr = (braid_Vector) u;
   IpplTimings::stopTimer(particleCreation);
   return 0;

}

int MyBraidApp::Clone(braid_Vector  u_,
                        braid_Vector *v_ptr)
{
   static IpplTimings::TimerRef Clone = IpplTimings::getTimer("Clone");
   IpplTimings::startTimer(Clone);
   BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;
   BraidVector<PLayout_t> *v = new BraidVector<PLayout_t>(PL_m); 
   v->create(nloc_m);
   Kokkos::deep_copy(v->R.getView(), u->R.getView());
   Kokkos::deep_copy(v->P.getView(), u->P.getView());
   Kokkos::deep_copy(v->q.getView(), u->q.getView());
   Kokkos::deep_copy(v->E.getView(), u->E.getView());

   v->setParticleBC(ippl::BC::PERIODIC);
   *v_ptr = (braid_Vector) v;
   IpplTimings::stopTimer(Clone);
   return 0;
}


int MyBraidApp::Free(braid_Vector u_)
{
   static IpplTimings::TimerRef Free = IpplTimings::getTimer("Free");
   IpplTimings::startTimer(Free);
   BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;
   delete u;
   IpplTimings::stopTimer(Free);
   return 0;
}

int MyBraidApp::Sum(double       alpha,
                      braid_Vector x_,
                      double       beta,
                      braid_Vector y_)
{
   static IpplTimings::TimerRef Sum = IpplTimings::getTimer("Sum");
   IpplTimings::startTimer(Sum);
   BraidVector<PLayout_t> *x = (BraidVector<PLayout_t>*) x_;
   BraidVector<PLayout_t> *y = (BraidVector<PLayout_t>*) y_;
   (y->R) = alpha*(x->R) + beta*(y->R);
   (y->P) = alpha*(x->P) + beta*(y->P);
   IpplTimings::stopTimer(Sum);
   return 0;
}

int MyBraidApp::SpatialNorm(braid_Vector  u_,
                              double       *norm_ptr)
{
   static IpplTimings::TimerRef SpatialNorm = IpplTimings::getTimer("SpatialNorm");
   IpplTimings::startTimer(SpatialNorm);
   BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;

   auto Pview = u->P.getView();
   double localNorm = 0.0;

   Kokkos::parallel_reduce("Local spatial norm", u->P.size(),
                           KOKKOS_LAMBDA(const int i, double& valLnorm){
                               double myValnorm = dot(Pview(i), Pview(i)).apply();
                               valLnorm += myValnorm;
                           }, Kokkos::Sum<double>(localNorm));

   Kokkos::fence();
   double globalNorm = 0.0;
   MPI_Allreduce(&localNorm, &globalNorm, 1, MPI_DOUBLE, MPI_SUM, spaceComm);
   *norm_ptr = std::sqrt(globalNorm / Np_m);

   IpplTimings::stopTimer(SpatialNorm);
   return 0;
}

int MyBraidApp::BufSize(int                *size_ptr,
                          BraidBufferStatus  &status)                           
{
   
   static IpplTimings::TimerRef BufSize = IpplTimings::getTimer("BufSize");
   IpplTimings::startTimer(BufSize);
   //BraidVector<PLayout_t> utemp(PL_m);
   bufSize_m = utemp.packedSize(nloc_m); 
   *size_ptr = (int)bufSize_m;
   IpplTimings::stopTimer(BufSize);
   return 0;
}

int MyBraidApp::BufPack(braid_Vector       u_,
                          void               *buffer,
                          BraidBufferStatus  &status)
{
   static IpplTimings::TimerRef BufPack = IpplTimings::getTimer("BufPack");
   IpplTimings::startTimer(BufPack);
   BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;
   using buffer_type = ippl::Communicate::buffer_type;
   void* raw_ptr = buffer;
   int id = buffer_ptr_to_id_.at(raw_ptr);
   buffer_type buf = Ippl::Comm->getBuffer(id, bufSize_m);
   u->serialize(*buf, nloc_m);
   status.SetSize(buf->getSize());
   buf->resetWritePos();

   IpplTimings::stopTimer(BufPack);
   return 0;
}

int MyBraidApp::BufUnpack(void              *buffer,
                            braid_Vector      *u_ptr,
                            BraidBufferStatus &status)
{
   static IpplTimings::TimerRef BufUnpack = IpplTimings::getTimer("BufUnpack");
   IpplTimings::startTimer(BufUnpack);
   using buffer_type = ippl::Communicate::buffer_type;
   BraidVector<PLayout_t> *u = new BraidVector<PLayout_t>(PL_m);
   u->create(nloc_m);
   u->setParticleBC(ippl::BC::PERIODIC);

   void* raw_ptr = buffer;
   int id = buffer_ptr_to_id_.at(raw_ptr);
   buffer_type buf = Ippl::Comm->getBuffer(id, bufSize_m);
   u->deserialize(*buf, nloc_m);
   buf->resetReadPos();
   *u_ptr = (braid_Vector) u;
   IpplTimings::stopTimer(BufUnpack);
   return 0;
}

int MyBraidApp::BufAlloc(void **buffer, int nbytes, BraidBufferStatus &bstatus)
{
    using buffer_type = ippl::Communicate::buffer_type;

    int id;

    // Try to reuse buffer from the pool
    if (!buffer_pool_.empty()) {
        // Reuse a buffer
        auto it = buffer_pool_.begin();
        id = it->first;
        buffer_type buf = it->second;
        buffer_pool_.erase(it);

        *buffer = static_cast<void*>(buf->getBuffer());
        return 0;
    }

    // No buffer to reuse; create new one
    id = next_id_++;
    buffer_type buf = Ippl::Comm->getBuffer(id, bufSize_m);
    void* raw_ptr = static_cast<void*>(buf->getBuffer());

    buffer_ptr_to_id_[raw_ptr] = id;
    *buffer = raw_ptr;

    return 0;
}

braid_Int MyBraidApp::BufFree(void **buffer)
{
    void* raw_ptr = *buffer;

    auto it = buffer_ptr_to_id_.find(raw_ptr);
    if (it != buffer_ptr_to_id_.end()) {
        int id = it->second;

        ippl::Communicate::buffer_type buf = Ippl::Comm->getBuffer(id, bufSize_m);
        // Return to pool
        buffer_pool_[id] = buf;
    }

    *buffer = nullptr;
    return 0;
}


//int MyBraidApp::Access(braid_Vector       u_,
//                         BraidAccessStatus &astatus)
//{
//   //char       filename[255];
//   //FILE      *file;
//   BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;
//
//   // Extract information from astatus
//   int done, level, iter;
//   double t;
//   astatus.GetTILD(&t, &iter, &level, &done);
//   //astatus.GetTIndex(&index);
//
//   typename ParticleAttrib<Vector_t>::HostMirror R_host = u->R.getHostMirror();
//   typename ParticleAttrib<Vector_t>::HostMirror P_host = u->P.getHostMirror();
//   Kokkos::deep_copy(R_host, u->R.getView());
//   Kokkos::deep_copy(P_host, u->P.getView());
//   std::stringstream pname;
//   pname << "data/Particle_";
//   pname << Ippl::Comm->rank();
//   pname << ".csv";
//   Inform pcsvout(NULL, pname.str().c_str(), Inform::OVERWRITE, Ippl::Comm->rank());
//   pcsvout.precision(10);
//   pcsvout.setf(std::ios::scientific, std::ios::floatfield);
//   pcsvout << "R_x, R_y, R_z, V_x, V_y, V_z" << endl;
//   for (size_type i = 0; i< nloc_m; i++) {
//       pcsvout << R_host(i)[0] << " "
//               << R_host(i)[1] << " "
//               << R_host(i)[2] << " "
//               << P_host(i)[0] << " "
//               << P_host(i)[1] << " "
//               << P_host(i)[2] << endl;
//   }
//   Ippl::Comm->barrier();
//   return 0;
//}

int MyBraidApp::Access(braid_Vector       u_,
                         BraidAccessStatus &astatus)
{
   static IpplTimings::TimerRef Access = IpplTimings::getTimer("Access");
   IpplTimings::startTimer(Access);
   BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;

   auto Eview = u->E.getView();
   double localEnergy = 0.0;

   Kokkos::parallel_reduce("Local Esquared", u->E.size(),
                           KOKKOS_LAMBDA(const int i, double& valL){
                               double myVal = Eview(i)[2] * Eview(i)[2];
                               valL += myVal;
                           }, Kokkos::Sum<double>(localEnergy));

   Kokkos::fence();
   double globalEnergy = 0.0;
   MPI_Allreduce(&localEnergy, &globalEnergy, 1, MPI_DOUBLE, MPI_SUM, spaceComm);
   double volume = (rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]);
   globalEnergy = globalEnergy * (volume / Np_m);

   // Extract information from astatus
   int done, level, iter, index;
   double t;
   astatus.GetTILD(&t, &iter, &level, &done);
   astatus.GetTIndex(&index);

   if(rankSpace == 0) {
       std::stringstream fname;
       fname << "data/ParticleLandau_rank_";
       fname << rankTime;
       fname << ".csv";


       Inform csvout(NULL, fname.str().c_str(), Inform::APPEND, Ippl::Comm->rank());
       csvout.precision(10);
       csvout.setf(std::ios::scientific, std::ios::floatfield);


       csvout << t << " "
              << globalEnergy << endl;
   }

   IpplTimings::stopTimer(Access);
   return 0;
}



//int MyBraidApp::Access(braid_Vector       /*u_*/,
//                         BraidAccessStatus &astatus)
//{
//   //char       filename[255];
//   //FILE      *file;
//   //BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;
//
//   // Extract information from astatus
//   int done, level, iter, index;
//   double t;
//   astatus.GetTILD(&t, &iter, &level, &done);
//   astatus.GetTIndex(&index);
//
//   double fieldEnergy = 0.0; 
//   double EzAmp = 0.0;
//
//   auto rhoview = rhoPIF_m.getView();
//   const int nghost = rhoPIF_m.getNghost();
//   using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;
//   
//   const FieldLayout_t& layout = rhoPIF_m.getLayout(); 
//   const Mesh_t& mesh = rhoPIF_m.get_mesh();
//   const Vector<double, Dim>& dx = mesh.getMeshSpacing();
//   const auto& domain = layout.getDomain();
//   Vector<double, Dim> Len;
//   Vector<int, Dim> N;
//
//   for (unsigned d=0; d < Dim; ++d) {
//       N[d] = domain[d].length();
//       Len[d] = dx[d] * N[d];
//   }
//
//
//   Kokkos::complex<double> imag = {0.0, 1.0};
//   double pi = std::acos(-1.0);
//   Kokkos::parallel_reduce("Ez energy and Max",
//                         mdrange_type({0, 0, 0},
//                                      {N[0],
//                                       N[1],
//                                       N[2]}),
//                         KOKKOS_LAMBDA(const int i,
//                                       const int j,
//                                       const int k,
//                                       double& tlSum,
//                                       double& tlMax)
//   {
//   
//       Vector<int, 3> iVec = {i, j, k};
//       Vector<double, 3> kVec;
//       double Dr = 0.0;
//       for(size_t d = 0; d < Dim; ++d) {
//           kVec[d] = 2 * pi / Len[d] * (iVec[d] - (N[d] / 2));
//           Dr += kVec[d] * kVec[d];
//       }
//
//       Kokkos::complex<double> Ek = {0.0, 0.0}; 
//       bool isNotZero = (Dr != 0.0);
//       double factor = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0))); 
//       Ek = -(imag * kVec[2] * rhoview(i+nghost,j+nghost,k+nghost) * factor);
//       double myVal = Ek.real() * Ek.real() + Ek.imag() * Ek.imag();
//
//       tlSum += myVal;
//
//       double myValMax = std::sqrt(myVal);
//
//       if(myValMax > tlMax) tlMax = myValMax;
//
//   }, Kokkos::Sum<double>(fieldEnergy), Kokkos::Max<double>(EzAmp));
//   
//
//   Kokkos::fence();
//   double volume = (rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]);
//   fieldEnergy *= volume;
//
//
//   if(rankSpace == 0) {
//       std::stringstream fname;
//       fname << "data/FieldLandau_rank_";
//       fname << rankTime;
//       fname << ".csv";
//
//
//       Inform csvout(NULL, fname.str().c_str(), Inform::APPEND, Ippl::Comm->rank());
//       csvout.precision(10);
//       csvout.setf(std::ios::scientific, std::ios::floatfield);
//
//
//       csvout << t << " "
//              << fieldEnergy << " "
//              << EzAmp << endl;
//   }
//
//   return 0;
//}


// --------------------------------------------------------------------------
// Main driver
// --------------------------------------------------------------------------

int main (int argc, char *argv[])
{

   MPI_Comm comm, spaceComm, timeComm;
   int rank, rankSpace, rankTime;
   int num_procs, sizeSpace, sizeTime;
   comm = MPI_COMM_WORLD;
   // Initialize MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &num_procs);
   BraidUtil util;
   
   Ippl ippl(argc, argv);
   {
   double        tstart, tstop;
   int           ntime;

   // Define time domain: ntime intervals
   tstart = 0.0;

   Vector_t rmin(0.0);
   Vector_t rmax(25.0);

   int num_procs_x = std::atoi(argv[15]);
   //int timeProcs = std::atoi(argv[16]);

    ippl::Vector<int,Dim> nmPIF = {
        std::atoi(argv[1]),
        std::atoi(argv[2]),
        std::atoi(argv[3])
    };

    ippl::Vector<int,Dim> nrPIC = {
        std::atoi(argv[4]),
        std::atoi(argv[5]),
        std::atoi(argv[6])
    };


    static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("mainTimer");
    //static IpplTimings::TimerRef computeErrors = IpplTimings::getTimer("computeErrors");

   IpplTimings::startTimer(mainTimer);
   
   size_type totalP = std::atoll(argv[7]);
   double tEnd = std::atof(argv[8]);
   tstop = tEnd;
   unsigned int nCycles = std::atoi(argv[12]);
   double tEndCycle = tEnd / nCycles;
   double dtFine = std::atof(argv[9]);
   double dtCoarse = std::atof(argv[10]);
   //unsigned int ntFine = std::ceil(dtSlice / dtFine);
   //unsigned int ntCoarse = std::ceil(dtSlice / dtCoarse);
   double tol = std::atof(argv[11]);
   //ntime = std::ceil(tEnd / dtFine);
   //if ((ntime & (timeProcs - 1)) != 0) { // not divisible
   //     ntime = (ntime + timeProcs - 1) & ~(timeProcs - 1);
   //     //std::cout << n << " is not divisible by " << p
   //     //     << ", rounding up to " << next << endl;
   //}

   std::string coarsetype = argv[18];
   int nLevels = std::atoi(argv[19]);
   int nrelax = std::atoi(argv[20]);
   int nrelax0 = std::atoi(argv[21]);
   double cfactortime = std::atof(argv[22]);
   double cfactorspace = std::atof(argv[23]);
   std::string shapetype = argv[13];
   int shapedegree = std::atoi(argv[14]);
   double coarseTol = std::atof(argv[16]);  
   double fineTol   = std::atof(argv[17]);
   

   util.SplitCommworld(&comm, num_procs_x, &spaceComm, &timeComm);
   MPI_Comm_rank(spaceComm, &rankSpace);
   MPI_Comm_rank(timeComm, &rankTime);

   MPI_Comm_size(spaceComm, &sizeSpace);
   MPI_Comm_size(timeComm, &sizeTime);

   ntime = sizeTime;
   double dtSlice = tEndCycle / sizeTime;
   //int CFactor = (int)(dtSlice/dtFine) + 1;
   //int CFactor = std::ceil(dtSlice/dtFine);
   // set up app structure
   MyBraidApp app(timeComm, spaceComm, rank, rankSpace, rankTime, 
                  sizeSpace, sizeTime, num_procs, tstart, tstop, 
                  ntime, nmPIF, nrPIC, rmin, rmax, totalP, 
                  dtFine, dtCoarse, dtSlice, coarsetype, shapetype, shapedegree,
                  coarseTol, fineTol, cfactortime, cfactorspace);

   //std::cout << "Rank: " << Ippl::Comm->rank() << "after braid app" << std::endl;

   ippl::NDIndex<Dim> domainPIC;
   ippl::NDIndex<Dim> domainPIF;
   for (unsigned i = 0; i< Dim; i++) {
       domainPIC[i] = ippl::Index(nrPIC[i]);
       domainPIF[i] = ippl::Index(nmPIF[i]);
   }

   ippl::e_dim_tag decomp[Dim];
   for (unsigned d = 0; d < Dim; ++d) {
       decomp[d] = ippl::SERIAL;
   }

   Vector_t origin = {rmin[0], rmin[1], rmin[2]};

   const bool isAllPeriodic=true;
   Mesh_t meshPIC(domainPIC, app.hrPIC_m, origin);
   Mesh_t meshPIF(domainPIF, app.hrPIF_m, origin);
   FieldLayout_t FLPIC(domainPIC, decomp, isAllPeriodic);
   FieldLayout_t FLPIF(domainPIF, decomp, isAllPeriodic);
   PLayout_t PL(FLPIC, meshPIC);

   app.PL_m = PL;
   //PL_m.updateLayout(FLPIC, meshPIC);
   app.rhoPIF_m.initialize(meshPIF, FLPIF);
   app.Sk_m.initialize(meshPIF, FLPIF);

   if(coarsetype == "PIC") {
        app.rhoPIC_m.initialize(meshPIC, FLPIC);
        app.EfieldPIC_m.initialize(meshPIC, FLPIC);
        app.initFFTSolver();
        //Dummy solve done to do the initializations for heFFTe
        app.rhoPIC_m = 0.0;
        app.solver_mp->solve();
   }

   app.initNUFFTs(FLPIF, nLevels);
   app.initializeShapeFunctionPIF();


   //app.initRequiredFields(coarseTol, fineTol);
   //app.initializeShapeFunctionPIF();

   //util.TestInitAccess(&app, spaceComm, stdout, tstart);
   //util.TestClone(&app, spaceComm, stdout, tstart);
   //util.TestSum( &app, spaceComm, stdout, tstart);
   //util.TestSpatialNorm( &app, spaceComm, stdout, tstart);
   //int correct=1;
   //correct = util.TestBuf(&app, spaceComm, stdout, tstart);

   //std::cout << "Correct: " << correct << std::endl;

   // Initialize Braid Core Object and set some solver options
   BraidCore core(comm, &app);
   core.SetPrintLevel(2);
   core.SetAccessLevel(0);
   core.SetMaxLevels(nLevels);
   core.SetMaxIter(10);
   //core.SetRelTol(tol);
   core.SetAbsTol(tol);
   int tnorm = 3; //Infinity norm
   core.SetTemporalNorm(tnorm);
   //core.SetCFactor(-1, (int)cfactortime);
   //core.SetCFactor(0, 8);
   core.SetCFactor(-1, 1);
   
   //core.SetCFactor(0, 4);
   core.SetNRelax(-1, nrelax);
   core.SetNRelax(0, nrelax0);
   
   //std::cout << "Rank: " << Ippl::Comm->rank() << "Levels: "  <<  nLevels << std::endl;
   // Run Simulation
   core.SetBufAllocFree();
   //core.SetFMG();
   //core.SetCRelaxWt(-1, 1.3);
   //core.SetSeqSoln(1);
   core.Drive();
   //std::cout << "Rank: " << Ippl::Comm->rank() << "after core drive" << std::endl;

   IpplTimings::stopTimer(mainTimer);
   //IpplTimings::print();
   //IpplTimings::print(std::string("timing.dat"));
   // Clean up
   MPI_Comm_free(&spaceComm);
   MPI_Comm_free(&timeComm);
   }
   Ippl::finalize();
   MPI_Finalize();

   return (0);
}



