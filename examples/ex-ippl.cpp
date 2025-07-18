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

//
// Example:       ex-01-pp.cpp
//
// Interface:     C++
// 
// Requires:      C-language and C++ support     
//
// Compile with:  make ex-01-pp
//
// Help with:     ex-01-pp -help
//
// Sample run:    mpirun -np 2 ex-01-pp
//
// Description:   solve the scalar ODE 
//                   u' = lambda u, 
//                   with lambda=-1 and y(0) = 1
//
//                Same as ex-01, only implements more advanced XBraid features.
//                
//                When run with the default 10 time steps, the solution is:
//                $ ./ex-01-pp
//                $ cat ex-01.out.00*
//                  1.00000000000000e+00
//                  6.66666666666667e-01
//                  4.44444444444444e-01
//                  2.96296296296296e-01
//                  1.97530864197531e-01
//                  1.31687242798354e-01
//                  8.77914951989026e-02
//                  5.85276634659351e-02
//                  3.90184423106234e-02
//                  2.60122948737489e-02
//                  1.73415299158326e-02
//

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

// --------------------------------------------------------------------------
// User-defined routines and objects 
// --------------------------------------------------------------------------

template <typename T>
struct Newton1D {

  double tol = 1e-12;
  int max_iter = 20;
  double pi = std::acos(-1.0);
  
  T k, alpha, u;

  KOKKOS_INLINE_FUNCTION
  Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  Newton1D(const T& k_, const T& alpha_, 
           const T& u_) 
  : k(k_), alpha(alpha_), u(u_) {}

  KOKKOS_INLINE_FUNCTION
  ~Newton1D() {}

  KOKKOS_INLINE_FUNCTION
  T f(T& x) {
      T F;
      F = x  + (alpha  * (std::sin(k * x) / k)) - u;
      return F;
  }

  KOKKOS_INLINE_FUNCTION
  T fprime(T& x) {
      T Fprime;
      Fprime = 1  + (alpha  * std::cos(k * x));
      return Fprime;
  }

  KOKKOS_FUNCTION
  void solve(T& x) {
      int iterations = 0;
      while (iterations < max_iter && std::fabs(f(x)) > tol) {
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

  //value_type alpha;

  T alpha, k, minU, maxU;

  // Initialize all members
  generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, 
                  T& alpha_, T& k_, T& minU_, T& maxU_)
      : x(x_), v(v_), rand_pool(rand_pool_), 
        alpha(alpha_), k(k_), minU(minU_), maxU(maxU_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    value_type u;
    for (unsigned d = 0; d < Dim; ++d) {

        u = rand_gen.drand(minU[d], maxU[d]);
        x(i)[d] = u / (1 + alpha[d]);
        Newton1D<value_type> solver(k[d], alpha[d], u);
        solver.solve(x(i)[d]);
        v(i)[d] = rand_gen.normal(0.0, 1.0);
    }

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};

double CDF(const double& x, const double& alpha, const double& k) {
   double cdf = x + (alpha / k) * std::sin(k * x);
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

   Vector_t alpha_m;
   Vector_t kw_m;

   Vector_i nrPIC_m;
   Vector_i nmPIF_m;

   double Q_m;

   size_type Np_m;
   size_type nloc_m;
   
   std::shared_ptr<Solver_t> solver_mp;
   //std::shared_ptr<FFT_t> fft_mp;
   
   double time_m;

   std::string shapetype_m;

   std::string coarsetype_m;

   PLayout_t PL_m;

   int shapedegree_m;
   std::string coarse = "Coarse";
   std::string fine = "Fine";

   double dtFine_m;
   double dtCoarse_m;

   MPI_Comm spaceComm, timeComm;

   size_type bufSize_m;

   std::shared_ptr<ippl::FFT<ippl::NUFFTransform, 3, double>> nufftType1Fine_mp,nufftType2Fine_mp,
                                                               nufftType1Coarse_mp,nufftType2Coarse_mp;

   // We will need the MPI Rank
   int rank, rankSpace, rankTime;
   int num_procs, sizeSpace, sizeTime;

   // Constructor 
   MyBraidApp(MPI_Comm comm_t_, MPI_Comm &comm_s_, int rank_, int rankSpace_, 
                       int rankTime_, int sizeSpace_, int sizeTime_, int num_procs_, 
                       double tstart_, double tstop_, int ntime_, 
                       Vector_i nmPIF, Vector_i nrPIC, Vector_t rmin, Vector_t rmax, 
                       size_type Np, Vector_t alpha, Vector_t kw, double& dtFine, 
                       double& dtCoarse, std::string& coarsetype, std::string& shapetype,
                       int& shapedegree);


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
   
   void initNUFFTs(FieldLayout_t& FLPIF, double& coarseTol,
                    double& fineTol) {
        
        ippl::ParameterList fftCoarseParams,fftFineParams;

        fftFineParams.add("gpu_method", 2);
        fftFineParams.add("gpu_sort", 0);
        fftFineParams.add("gpu_kerevalmeth", 1);
        fftFineParams.add("tolerance", fineTol);
        fftFineParams.add("gpu_binsizex", 8);
        fftFineParams.add("gpu_binsizey", 8);
        fftFineParams.add("gpu_binsizez", 2);
        fftFineParams.add("gpu_maxsubprobsize", 1024);

        fftCoarseParams.add("gpu_method", 2);
        fftCoarseParams.add("gpu_sort", 0);
        fftCoarseParams.add("gpu_kerevalmeth", 1);
        fftCoarseParams.add("tolerance", coarseTol);
        fftCoarseParams.add("gpu_binsizex", 8);
        fftCoarseParams.add("gpu_binsizey", 8);
        fftCoarseParams.add("gpu_binsizez", 2);
        fftCoarseParams.add("gpu_maxsubprobsize", 1024);

        fftFineParams.add("use_cufinufft_defaults", false);
        fftCoarseParams.add("use_cufinufft_defaults", false);
        
        nufftType1Fine_mp = std::make_shared<ippl::FFT<ippl::NUFFTransform, 3, double>>(FLPIF, nloc_m, 1, fftFineParams);
        nufftType2Fine_mp = std::make_shared<ippl::FFT<ippl::NUFFTransform, 3, double>>(FLPIF, nloc_m, 2, fftFineParams);

        nufftType1Coarse_mp = std::make_shared<ippl::FFT<ippl::NUFFTransform, 3, double>>(FLPIF, nloc_m, 1, fftCoarseParams);
        nufftType2Coarse_mp = std::make_shared<ippl::FFT<ippl::NUFFTransform, 3, double>>(FLPIF, nloc_m, 2, fftCoarseParams);
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

   void initRequiredFields(double& coarseTol, double& fineTol) {

       ippl::NDIndex<Dim> domainPIC;
       ippl::NDIndex<Dim> domainPIF;
       for (unsigned i = 0; i< Dim; i++) {
           domainPIC[i] = ippl::Index(nrPIC_m[i]);
           domainPIF[i] = ippl::Index(nmPIF_m[i]);
       }

       ippl::e_dim_tag decomp[Dim];
       for (unsigned d = 0; d < Dim; ++d) {
           decomp[d] = ippl::SERIAL;
       }

       Vector_t origin = {rmin_m[0], rmin_m[1], rmin_m[2]};

       const bool isAllPeriodic=true;
       Mesh_t meshPIC(domainPIC, hrPIC_m, origin);
       Mesh_t meshPIF(domainPIF, hrPIF_m, origin);
       FieldLayout_t FLPIC(domainPIC, decomp, isAllPeriodic);
       FieldLayout_t FLPIF(domainPIF, decomp, isAllPeriodic);
       PLayout_t PL(FLPIC, meshPIC);

       PL_m.updateLayout(FLPIC, meshPIC);
       rhoPIF_m.initialize(meshPIF, FLPIF);
       Sk_m.initialize(meshPIF, FLPIF);

       if(coarsetype_m == "PIC") {
            rhoPIC_m.initialize(meshPIC, FLPIC);
            EfieldPIC_m.initialize(meshPIC, FLPIC);
            initFFTSolver();
	        //Dummy solve done to do the initializations for heFFTe
            rhoPIC_m = 0.0;
            solver_mp->solve();
       }

       initNUFFTs(FLPIF, coarseTol, fineTol);
   }

   void LeapFrogPIF(BraidVector<PLayout_t>& u, const double& dt, const unsigned int& nt, const std::string& propagator) {
    
        //BraidVector *u = (BraidVector*) u_;
        PLayout_t& PL = u.getLayout();
        auto &Rtemp = u.R;
        auto &Ptemp = u.P;
        auto &q = u.q;
        auto &E = u.E;
        rhoPIF_m = {0.0, 0.0};
        if(propagator == "Coarse") {
            scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1Coarse_mp.get(), spaceComm);
        }
        else if(propagator == "Fine") {
            scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1Fine_mp.get(), spaceComm);
        }
    
        rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
        // Solve for and gather E field
        if(propagator == "Coarse") {
            gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2Coarse_mp.get(), q);
        }
        else if(propagator == "Fine") {
            gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2Fine_mp.get(), q);
        }

        //Reset the value of q here as we used it as a temporary object in gather to 
        //save memory
        q = Q_m / Np_m;
    
        for (unsigned int it=0; it<nt; it++) {
            // kick
            Ptemp = Ptemp - 0.5 * dt * E;
    
            //drift
            Rtemp = Rtemp + dt * Ptemp;
    
            //Apply particle BC
            PL_m.applyBC(Rtemp, PL_m.getRegionLayout().getDomain());
    
            //scatter the charge onto the underlying grid
            rhoPIF_m = {0.0, 0.0};
            if(propagator == "Coarse") {
                scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1Coarse_mp.get(), spaceComm);
            }
            else if(propagator == "Fine") {
                scatterPIFNUFFT(q, rhoPIF_m, Sk_m, Rtemp, nufftType1Fine_mp.get(), spaceComm);
            }
    
            rhoPIF_m = rhoPIF_m / ((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]));
    
            // Solve for and gather E field
            if(propagator == "Coarse") {
                gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2Coarse_mp.get(), q);
            }
            else if(propagator == "Fine") {
                gatherPIFNUFFT(E, rhoPIF_m, Sk_m, Rtemp, nufftType2Fine_mp.get(), q);
            }

            q = Q_m / Np_m;

            //kick
            Ptemp = Ptemp - 0.5 * dt * E;
        }
    }


    void LeapFrogPIC(BraidVector<PLayout_t>& u, const double& dt, const unsigned int& nt) {
    
        //BraidVector *u = (BraidVector*) u_;
        PLayout_t& PL = u.getLayout();
        auto &Rtemp = u.R;
        auto &Ptemp = u.P;
        auto &q = u.q;
        auto &E = u.E;
        rhoPIC_m = 0.0;
        scatter(q, rhoPIC_m, Rtemp, spaceComm);
    
        rhoPIC_m = rhoPIC_m / (hrPIC_m[0] * hrPIC_m[1] * hrPIC_m[2]);
        rhoPIC_m = rhoPIC_m - (Q_m/((rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2])));
    
        //Field solve
        solver_mp->solve();
    
        // gather E field
        gather(E, EfieldPIC_m, Rtemp);
    
        for (unsigned int it=0; it<nt; it++) {
            // kick
            Ptemp = Ptemp - 0.5 * dt * E;
    
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
            Ptemp = Ptemp - 0.5 * dt * E;
        }
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
                       size_type Np, Vector_t alpha, Vector_t kw, double& dtFine, 
                       double& dtCoarse, std::string& coarsetype, std::string& shapetype,
                       int& shapedegree) : BraidApp(comm_t_, tstart_, tstop_, ntime_)
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
   Q_m = -length_m[0] * length_m[1] * length_m[2];
   nmPIF_m = nmPIF;
   nrPIC_m = nrPIC;
   for (unsigned d = 0; d < 3; d++) {
        hrPIC_m[d] = length_m[d] / nrPIC_m[d];
        hrPIF_m[d] = length_m[d] / nmPIF_m[d];
   }
   Np_m = Np;
   alpha_m = alpha;
   kw_m = kw;
   double factor = 1.0 / sizeSpace;
   nloc_m = (size_type)(factor * Np_m);
   dtFine_m = dtFine;
   dtCoarse_m = dtCoarse;
   coarsetype_m = coarsetype;
   shapetype_m = shapetype;
   shapedegree_m = shapedegree;
}

// 
int MyBraidApp::Step(braid_Vector    u_,
                     braid_Vector    ustop_,
                     braid_Vector    fstop_,
                     BraidStepStatus &pstatus)
{
   
   BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;
   double tstart;             // current time
   double tstop;              // evolve to this time

   // Get time step information
   pstatus.GetTstartTstop(&tstart, &tstop);

   unsigned int ntFine = std::ceil((tstop - tstart) / dtFine_m);
   unsigned int ntCoarse = std::ceil((tstop - tstart) / dtCoarse_m);

   int level;
   int max_levels;
   pstatus.GetLevel(&level);
   pstatus.GetNLevels(&max_levels);

   if (level == 0) {
       LeapFrogPIF(*u, dtFine_m, ntFine, fine);  
       //LeapFrogPIC(*u, dtCoarse_m, ntCoarse);  
   }
   else if ((level == 1) && (coarsetype_m == "PIF")) {
       //TODO: To create a hierarchy of nuffts with increasingly coarse tolerance at each level
       LeapFrogPIF(*u, dtCoarse_m, ntCoarse, coarse);  
   }
   else if ((level == 1) && (coarsetype_m == "PIC")) {
       LeapFrogPIC(*u, dtCoarse_m, ntCoarse);  
   }

   // no refinement
   pstatus.SetRFactor(1);
   
   return 0;

}

int MyBraidApp::Init(double        t,
                       braid_Vector *u_ptr)
{
   BraidVector<PLayout_t> *u = new BraidVector<PLayout_t>(PL_m);
   u->create(nloc_m);


   if (t != tstart)
   {
      u->R = 0.0;
      u->P = 0.0;
      u->q = 0.0;
   }
   else
   {
     Vector_t minU, maxU;
     for (unsigned d = 0; d <Dim; ++d) {
        minU[d] = CDF(rmin_m[d], alpha_m[d], kw_m[d]);
        maxU[d]   = CDF(rmax_m[d], alpha_m[d], kw_m[d]);
      }
      Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100*rankSpace));
      Kokkos::parallel_for(nloc_m, generate_random<Vector_t, Kokkos::Random_XorShift64_Pool<>, Dim>(
                           u->R.getView(), u->P.getView(), rand_pool64, alpha_m, kw_m, minU, maxU));
    
      u->q = Q_m / Np_m;

      Kokkos::fence();
   }

   bufSize_m = u->packedSize(nloc_m);
   u->setParticleBC(ippl::BC::PERIODIC);
   *u_ptr = (braid_Vector) u;
   return 0;

}

int MyBraidApp::Clone(braid_Vector  u_,
                        braid_Vector *v_ptr)
{
   BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;
   BraidVector<PLayout_t> *v = new BraidVector<PLayout_t>(PL_m); 
   //BraidVector *v = new BraidVector(u->value); 
   //*v_ptr = (braid_Vector) v;
   v->create(nloc_m);
   Kokkos::deep_copy(v->R.getView(), u->R.getView());
   Kokkos::deep_copy(v->P.getView(), u->P.getView());
   Kokkos::deep_copy(v->q.getView(), u->q.getView());
   Kokkos::deep_copy(v->E.getView(), u->E.getView());

   v->setParticleBC(ippl::BC::PERIODIC);
   *v_ptr = (braid_Vector) v;
   return 0;
}


int MyBraidApp::Free(braid_Vector u_)
{
   BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;
   delete u;
   return 0;
}

int MyBraidApp::Sum(double       alpha,
                      braid_Vector x_,
                      double       beta,
                      braid_Vector y_)
{
   BraidVector<PLayout_t> *x = (BraidVector<PLayout_t>*) x_;
   BraidVector<PLayout_t> *y = (BraidVector<PLayout_t>*) y_;
   (y->R) = alpha*(x->R) + beta*(y->R);
   (y->P) = alpha*(x->P) + beta*(y->P);
   return 0;
}

int MyBraidApp::SpatialNorm(braid_Vector  u_,
                              double       *norm_ptr)
{
   BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;
   //dot = (u->value)*(u->value);
   //*norm_ptr = sqrt(dot);

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
   *norm_ptr = globalNorm / std::sqrt(Np_m);

   return 0;
}

int MyBraidApp::BufSize(int                *size_ptr,
                          BraidBufferStatus  &status)                           
{
   *size_ptr = (int)bufSize_m;
   return 0;
}

int MyBraidApp::BufPack(braid_Vector       u_,
                          void               *buffer,
                          BraidBufferStatus  &status)
{
   BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;
   using buffer_type = ippl::Communicate::buffer_type;
   buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_SEND, bufSize_m);
   u->serialize(*buf, nloc_m);
   status.SetSize(buf->getSize());
   buf->resetWritePos();

   return 0;
}

int MyBraidApp::BufUnpack(void              *buffer,
                            braid_Vector      *u_ptr,
                            BraidBufferStatus &status)
{
   using buffer_type = ippl::Communicate::buffer_type;
   buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_SEND, bufSize_m);
   
   BraidVector<PLayout_t> *u = new BraidVector<PLayout_t>(PL_m); 
   u->deserialize(*buf, nloc_m);
   buf->resetReadPos();
   *u_ptr = (braid_Vector) u;
   return 0;
}

int MyBraidApp::BufAlloc(void              **buffer,
                        int               nbytes,
                        BraidBufferStatus &bstatus)
{
   using buffer_type = ippl::Communicate::buffer_type;
   //TODO: If we can differentiate between send and receives by exposing request_type 
   //from braid then we can get two buffers for send and receive and reuse them without
   //freeing them and only deleting at the end. This is usually important for performance
   //in GPUs.
   buffer_type buf = Ippl::Comm->getBuffer(IPPL_PARAREAL_SEND, (size_type)nbytes);

   *buffer = (void *)buf->getBuffer();

   return 0;
}
braid_Int MyBraidApp::BufFree(void          **buffer)
{
   Ippl::Comm->deleteBuffer(IPPL_PARAREAL_SEND);
   return 0;
}

int MyBraidApp::Access(braid_Vector       /*u_*/,
                         BraidAccessStatus &astatus)
{
   //char       filename[255];
   //FILE      *file;
   //BraidVector<PLayout_t> *u = (BraidVector<PLayout_t>*) u_;

   // Extract information from astatus
   int done, level, iter;
   double t;
   astatus.GetTILD(&t, &iter, &level, &done);
   //astatus.GetTIndex(&index);

   double fieldEnergy = 0.0; 
   double EzAmp = 0.0;

   auto rhoview = rhoPIF_m.getView();
   const int nghost = rhoPIF_m.getNghost();
   using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<Dim>>;
   
   const FieldLayout_t& layout = rhoPIF_m.getLayout(); 
   const Mesh_t& mesh = rhoPIF_m.get_mesh();
   const Vector<double, Dim>& dx = mesh.getMeshSpacing();
   const auto& domain = layout.getDomain();
   Vector<double, Dim> Len;
   Vector<int, Dim> N;

   for (unsigned d=0; d < Dim; ++d) {
       N[d] = domain[d].length();
       Len[d] = dx[d] * N[d];
   }


   Kokkos::complex<double> imag = {0.0, 1.0};
   double pi = std::acos(-1.0);
   Kokkos::parallel_reduce("Ez energy and Max",
                         mdrange_type({0, 0, 0},
                                      {N[0],
                                       N[1],
                                       N[2]}),
                         KOKKOS_LAMBDA(const int i,
                                       const int j,
                                       const int k,
                                       double& tlSum,
                                       double& tlMax)
   {
   
       Vector<int, 3> iVec = {i, j, k};
       Vector<double, 3> kVec;
       double Dr = 0.0;
       for(size_t d = 0; d < Dim; ++d) {
           kVec[d] = 2 * pi / Len[d] * (iVec[d] - (N[d] / 2));
           Dr += kVec[d] * kVec[d];
       }

       Kokkos::complex<double> Ek = {0.0, 0.0}; 
       bool isNotZero = (Dr != 0.0);
       double factor = isNotZero * (1.0 / (Dr + ((!isNotZero) * 1.0))); 
       Ek = -(imag * kVec[2] * rhoview(i+nghost,j+nghost,k+nghost) * factor);
       double myVal = Ek.real() * Ek.real() + Ek.imag() * Ek.imag();

       tlSum += myVal;

       double myValMax = std::sqrt(myVal);

       if(myValMax > tlMax) tlMax = myValMax;

   }, Kokkos::Sum<double>(fieldEnergy), Kokkos::Max<double>(EzAmp));
   

   Kokkos::fence();
   double volume = (rmax_m[0] - rmin_m[0]) * (rmax_m[1] - rmin_m[1]) * (rmax_m[2] - rmin_m[2]);
   fieldEnergy *= volume;


   if(rankSpace == 0) {
       std::stringstream fname;
       fname << "data/FieldLandau_rank_";
       fname << rankTime;
       fname << ".csv";


       Inform csvout(NULL, fname.str().c_str(), Inform::APPEND, Ippl::Comm->rank());
       csvout.precision(10);
       csvout.setf(std::ios::scientific, std::ios::floatfield);


       csvout << t << " "
              << fieldEnergy << " "
              << EzAmp << endl;
   }

   return 0;
}


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

   double        tstart, tstop;
   int           ntime;

   // Define time domain: ntime intervals
   tstart = 0.0;

   Vector_t kw = {0.5, 0.5, 0.5};
   Vector_t alpha = {0.05, 0.05, 0.05};
   Vector_t rmin(0.0);
   Vector_t rmax = 2 * pi / kw ;

   int num_procs_x = std::atoi(argv[15]);
   int timeProcs = std::atoi(argv[16]);

   ntime  = timeProcs;
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

   size_type totalP = std::atoll(argv[7]);
   double tEnd = std::atof(argv[8]);
   tstop = tEnd;
   unsigned int nCycles = std::atoi(argv[12]);
   //double tEndCycle = tEnd / nCycles;
   //double dtSlice = tEndCycle / sizeTime;
   double dtFine = std::atof(argv[9]);
   double dtCoarse = std::atof(argv[10]);
   //unsigned int ntFine = std::ceil(dtSlice / dtFine);
   //unsigned int ntCoarse = std::ceil(dtSlice / dtCoarse);
   double tol = std::atof(argv[11]);

   std::string coarsetype = argv[19];
   std::string shapetype = argv[13];
   int shapedegree = std::atoi(argv[14]);
   double coarseTol = std::atof(argv[17]);  
   double fineTol   = std::atof(argv[18]);
   

   util.SplitCommworld(&comm, num_procs_x, &spaceComm, &timeComm);
   MPI_Comm_rank(spaceComm, &rankSpace);
   MPI_Comm_rank(timeComm, &rankTime);

   MPI_Comm_size(spaceComm, &sizeSpace);
   MPI_Comm_size(timeComm, &sizeTime);

   // set up app structure
   MyBraidApp app(timeComm, spaceComm, rank, rankSpace, rankTime, 
                  sizeSpace, sizeTime, num_procs, tstart, tstop, 
                  ntime, nmPIF, nrPIC, rmin, rmax, totalP, alpha, kw,
                  dtFine, dtCoarse, coarsetype, shapetype, shapedegree);


   app.initRequiredFields(coarseTol, fineTol);
   app.initializeShapeFunctionPIF();

   // Initialize Braid Core Object and set some solver options
   BraidCore core(comm, &app);
   core.SetPrintLevel(2);
   core.SetMaxLevels(1);
   core.SetRelTol(tol);
   int tnorm = 3; //Infinity norm
   core.SetTemporalNorm(tnorm);
   core.SetCFactor(-1, 2);
   
   // Run Simulation
   core.Drive();

   // Clean up
   MPI_Comm_free(&spaceComm);
   MPI_Comm_free(&timeComm);


   Ippl::finalize();
   MPI_Finalize();

   return (0);
}



