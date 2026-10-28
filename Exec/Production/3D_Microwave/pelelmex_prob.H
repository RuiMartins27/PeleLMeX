#define M_PI 3.14159265358979323846
#define TT2 false
#define TT3 false
#define CASE1 true
#define CASE2 false
#define CASE3 false
#define CASE4 false
#define CASE5 false
#ifndef PELELM_PROB_H
#define PELELM_PROB_H


#ifdef QUENCH
#undef QUENCH
#endif

// #define QUENCH

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#ifdef AMREX_USE_EB
#include <AMReX_ParmParse.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_Math.H>
#endif

#include <PeleLMeX_ProblemSpecificFunctions.H>
#include <PeleLMeX_Index.H>
#include <PMFData.H>
#include <PelePhysics.H>

#include <math.h>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>

#ifdef PELE_USE_MWSOLVER

#include <petscksp.h>
#include <harps.h>

#endif


#ifdef AMREX_USE_EB
namespace eb_fillet {

// True signed-distance infinite cylinder (axis dir=0/1/2). Convention: <0 fluid, >0 body.
struct InfiniteCylinderSDF
{
  amrex::Real R;
  int dir;
  amrex::RealArray c;

  AMREX_GPU_HOST_DEVICE
  amrex::Real operator() (AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) const noexcept
  {
    const amrex::Real px = x - c[0];
    const amrex::Real py = y - c[1];
#if (AMREX_SPACEDIM == 3)
    const amrex::Real pz = z - c[2];
#else
    const amrex::Real pz = 0.0;
#endif

#if (AMREX_SPACEDIM == 3)
    amrex::Real rperp = 0.0;
    if      (dir == 0) { rperp = sqrt(py*py + pz*pz); }
    else if (dir == 1) { rperp = sqrt(px*px + pz*pz); }
    else               { rperp = sqrt(px*px + py*py); }
#else
    amrex::Real rperp = (dir == 0) ? abs(py) : abs(px);
#endif

    return rperp - R; // SDF
  }

  AMREX_GPU_HOST_DEVICE
  amrex::Real operator() (const amrex::RealArray& p) const noexcept
  {
    return (*this)(AMREX_D_DECL(p[0], p[1], p[2]));
  }
};

// Signed-distance capped cylinder aligned with x/y/z (dir=0/1/2).
struct CappedCylinderSDF
{
  amrex::Real R;
  amrex::Real halfL;
  int dir;
  amrex::RealArray c;

  AMREX_GPU_HOST_DEVICE
  amrex::Real operator() (AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) const noexcept
  {
    const amrex::Real px = x - c[0];
    const amrex::Real py = y - c[1];
#if (AMREX_SPACEDIM == 3)
    const amrex::Real pz = z - c[2];
#else
    const amrex::Real pz = 0.0;
#endif

    amrex::Real axial = 0.0;
    amrex::Real rperp = 0.0;

#if (AMREX_SPACEDIM == 3)
    if      (dir == 0) { axial = px; rperp = sqrt(py*py + pz*pz); }
    else if (dir == 1) { axial = py; rperp = sqrt(px*px + pz*pz); }
    else               { axial = pz; rperp = sqrt(px*px + py*py); }
#else
    if (dir == 0) { axial = px; rperp = abs(py); }
    else          { axial = py; rperp = abs(px); }
#endif

    // SDF for capped cylinder
    const amrex::Real qx = rperp - R;
    const amrex::Real qy = abs(axial) - halfL;

    const amrex::Real ax = amrex::max(qx, 0.0);
    const amrex::Real ay = amrex::max(qy, 0.0);

    const amrex::Real outside = sqrt(ax*ax + ay*ay);
    const amrex::Real inside  = amrex::min(amrex::max(qx, qy), 0.0);

    return outside + inside;
  }

  AMREX_GPU_HOST_DEVICE
  amrex::Real operator() (const amrex::RealArray& p) const noexcept
  {
    return (*this)(AMREX_D_DECL(p[0], p[1], p[2]));
  }
};

// Polynomial smooth-min: rounded UNION for SDFs; k is fillet radius (meters).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real smooth_min_poly(amrex::Real a, amrex::Real b, amrex::Real k) noexcept
{
  const amrex::Real d = abs(a - b);
  const amrex::Real h = amrex::max(k - d, 0.0) / k;
  return amrex::min(a, b) - (h*h*h) * k * (1.0/6.0);
}

// Reactor IF: union of 6 cylinders, with optional fillets at 1↔(2–5) and 1↔6 junctions.
struct MWReactorIF
{
  InfiniteCylinderSDF c1;
  CappedCylinderSDF c2, c3, c4, c5, c6;
  amrex::Real k15; // fillet radius between cylinder1 and cylinders2–5
  amrex::Real k16; // fillet radius between cylinder1 and cylinder6

  AMREX_GPU_HOST_DEVICE
  amrex::Real operator() (AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) const noexcept
  {
    const amrex::Real f1 = c1(AMREX_D_DECL(x,y,z));

    amrex::Real f25 = amrex::min(c2(AMREX_D_DECL(x,y,z)), c3(AMREX_D_DECL(x,y,z)));
    f25 = amrex::min(f25, c4(AMREX_D_DECL(x,y,z)));
    f25 = amrex::min(f25, c5(AMREX_D_DECL(x,y,z)));

    amrex::Real f = (k15 > 0.0) ? smooth_min_poly(f1, f25, k15)
                                : amrex::min(f1, f25);

    const amrex::Real f6 = c6(AMREX_D_DECL(x,y,z));
    f = (k16 > 0.0) ? smooth_min_poly(f, f6, k16)
                    : amrex::min(f, f6);

    return f;
  }

  AMREX_GPU_HOST_DEVICE
  amrex::Real operator() (const amrex::RealArray& p) const noexcept
  {
    return (*this)(AMREX_D_DECL(p[0], p[1], p[2]));
  }
};

} // namespace eb_fillet

#endif

struct MyProbParm : public ProbParmDefault
{
  amrex::Real P_mean = 101325.0; 
  amrex::Real T_mean = 300.0;
  amrex::Real T_wall = T_mean;
  amrex::Real T_jet =  300.0;
  amrex::Real V_jet = 1.0;
  amrex::Real jet_rad = 0.25;
  amrex::Real bl_thickness = 0.01;
  amrex::Real vmax = -1.0;
  amrex::Real tanvmax = 0.0;
  amrex::Real vmean_tan = 0.0;
  amrex::Real vmean_rad = 0.0;
  amrex::Real R = 0.45;
  amrex::Real rtrans = 0.5;
  amrex::Real ztan = 0.0;
  amrex::Real rperp = 0.0;
  amrex::Real cyl_R = 0.0;
  amrex::Real T_tan = 300.0;
  amrex::Real T_rad = 300.0;
  amrex::Real zperp = 0.0;
  amrex::Real YCH4 = 0.0;
  amrex::Real rtan = 0.0;
  int ntan = 0;
  int nperp = 0;
  amrex::Real spark_power    = 0.0;
  amrex::Real spark_volume   = 0.0;
  amrex::Real spark_locationx = 0.0;
  amrex::Real spark_locationz = 0.0;
  amrex::Real spark_radiusx = 0.0;
  amrex::Real spark_radiusz = 0.0;
  amrex::Real power = 0.0;
  amrex::Real power_radiusx = 0.0;
  amrex::Real power_radiusy = 0.0;
  amrex::Real power_radiusz = 0.0;
  amrex::Real power_time = 1.0e-6;

  amrex::Real z_center_waveguide = 0.14;
  int power_shape = 1;

  #ifdef PELE_USE_MWSOLVER

  amrex::Real EM_t_start = 0.15;
  int plot_harps_int = 100;
  int harps_verbose = 1;
  bool flag_solve_EM = false;
  bool flag_optimize_reflector = false;
  bool normalize_power = true;
  amrex::GpuArray<char, 64> plot_harps_file = {"plt_harps"};

  #endif

  // ACTIVE CONTROL

  amrex::Real Time_ctrl =0.0;
  // Temperature control
  int Tctrl_on = 0;
  int Tctrl_verbose = 0;
  int Tctrl_int = 1;
  amrex::Real T_target = 3000.0;

  // Controller shapping
  amrex::Real tau = 1e-4;
  amrex::Real rz_min = 5e-3;
  amrex::Real rz_max = 5e-2;
  amrex::Real drz_max = 1e-5;

  // PI gains
  amrex::Real Kp = 0.0;
  amrex::Real Ki = 0.0;
  amrex::Real ierr = 0.0;

  // Live parameter radiuz
  amrex::Real rz = 0.0;

  // Outlet sponge
  bool sponge_on = false;
  bool do_small_radius = false;
  amrex::Real sponge_zbeg = 0.0;     // start of sponge
  amrex::Real sponge_zend = 0.0;     // end of sponge, usually near ProbHi(2)
  amrex::Real sponge_r = 1.0e30;     // radius limit; set large to affect full cross-section
  amrex::Real sponge_sigma_perp = 0.0; // x,y damping
  amrex::Real sponge_sigma_para = 0.0; // z damping
  amrex::Real sponge_T_sigma = 0.0;    // optional temperature relaxation
  amrex::Real sponge_T_ref = 300.0;    // optional target temp
  int sponge_shape = 2;  

  // Wall model
  amrex::Real hExt = 0 ; 
  amrex::Real eps = 0 ; 
  amrex::Real Ri = 0 ; 
  amrex::Real Ro = 0 ; 
  amrex::Real Rcond = 0 ; 
  amrex::Real sigma = 0 ; 
  amrex::Real Gout = 0 ; 
  amrex::Real Ggas = 0 ; 
  amrex::Real Tinf = 0 ; 
  amrex::Real wall_a = 0 ; 
  amrex::Real wall_b = 0 ; 
  amrex::Real Twall_max = 1400.0;
  amrex::Real Twall_min = 293.15;
};


struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void initdata(
    int i,
    int j,
    int k,
    int /*is_incompressible*/,
    amrex::Array4<amrex::Real> const& state,
    amrex::Array4<amrex::Real> const& /*aux*/,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real* dx = geomdata.CellSize();
    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real massfrac[NUM_SPECIES] = {0.0};
    amrex::Real rz = 0;

    amrex::Real x[3] = {
      prob_lo[0] + static_cast<amrex::Real>(i + 0.5) * dx[0],
      prob_lo[1] + static_cast<amrex::Real>(j + 0.5) * dx[1],
      prob_lo[2] + static_cast<amrex::Real>(k + 0.5) * dx[2]};

    rz = sqrt((x[0]*x[0]+x[1]*x[1]));
    
    if (rz<prob_parm.cyl_R) {
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
                  , state(i, j, k, VELZ) = -1.0);
    }
    else {
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
                  , state(i, j, k, VELZ) =0.0);
    }

    state(i, j, k, TEMP) = prob_parm.T_mean;
    amrex::Real P_cgs = prob_parm.P_mean * 10.0;
    massfrac[O2_ID]  = 0.0;
    massfrac[N2_ID]  = 1.0 - massfrac[O2_ID];

    
    // Density
    amrex::Real rho_cgs = 0.0;
    eos.PYT2R(P_cgs, massfrac, state(i, j, k, TEMP), rho_cgs);
    state(i, j, k, DENSITY) = rho_cgs * 1.0e3;
 
    // Enthalpy
    amrex::Real h_cgs = 0.0;
    eos.TY2H(state(i, j, k, TEMP), massfrac, h_cgs);
    state(i, j, k, RHOH) = h_cgs * 1.0e-4 * state(i, j, k, DENSITY);
 
    // Species mass
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i, j, k, FIRSTSPEC + n) = massfrac[n] * state(i, j, k, DENSITY);
    }
  }


  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal(
    const amrex::Real x[AMREX_SPACEDIM],
    const amrex::CellData<amrex::Real>& /*s_in*/,
    amrex::Real s_ext[NVAR],
    const int idir,
    const int sgn,
    const amrex::Real time,
    amrex::GeometryData const& geomdata,
    MyProbParm const&  prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
  const amrex::Real* prob_hi = geomdata.ProbHi();
  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  amrex::Real X_main[NUM_SPECIES] = {0.0};
  amrex::Real Y_main[NUM_SPECIES] = {0.0};
  amrex::Real X_quench[NUM_SPECIES] = {0.0};
  amrex::Real Y_quench[NUM_SPECIES] = {0.0};

  Y_main[O2_ID] = 0.0;
  Y_main[N2_ID] = 1.0 - Y_main[O2_ID];

  X_quench[O2_ID] = 1.0;
  X_quench[N2_ID] = 1.0 - X_quench[O2_ID];
  eos.X2Y(X_quench, Y_quench ); 
  
  amrex::Real time_tan = 1; // amrex::min(amrex::max(0.0, time)/0.4, 1.0);
  amrex::Real time_perp = amrex::min(amrex::max(0.0, time - 0.6)/0.4, 1.0);

  amrex::Real zseparate= 0.0;
  zseparate = prob_hi[2] + prob_parm.ztan - 4*prob_parm.rtan ;
 
  auto small_radius = prob_parm.do_small_radius; 
  amrex::Real zfour = 1.96875e-01;
  amrex::Real limit = 0.028/64*0.55;
  bool inside_four = (abs(x[2]-zfour) < limit) && (abs(x[1])< limit);
  if (idir == 0 ) {
    amrex::Real p_cgs = prob_parm.P_mean * 10.0;
    amrex::Real rho_cgs = 0.0;
    amrex::Real RhoH_temp = 0.0;

    s_ext[TEMP] = prob_parm.T_mean;
    eos.PYT2R(p_cgs, Y_main, s_ext[TEMP], rho_cgs);
    s_ext[DENSITY] = rho_cgs * 1.0e3;
    eos.TY2H(s_ext[TEMP], Y_main, RhoH_temp);
    s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
    for (int n = 0; n < NUM_SPECIES; n++) {
      s_ext[FIRSTSPEC + n] = Y_main[n] * s_ext[DENSITY];
    }

    s_ext[VELX] = sgn*prob_parm.vmean_tan*time_tan;
    s_ext[VELY] = 0.0;
    s_ext[VELZ] = 0.0;

    if (x[2] < zseparate ){
      amrex::Real p_cgs = prob_parm.P_mean * 10.0;
      amrex::Real rho_cgs = 0.0;
      amrex::Real RhoH_temp = 0.0;

      s_ext[TEMP] = prob_parm.T_mean;
      eos.PYT2R(p_cgs, Y_quench, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;
      eos.TY2H(s_ext[TEMP], Y_quench, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = Y_quench[n] * s_ext[DENSITY];
      }
      if(small_radius){
        if (inside_four){
          s_ext[VELX] = sgn*prob_parm.vmean_rad*time_perp*4.0512;
        }
        else{
          //s_ext[VELX] = 0.0; 
          s_ext[VELX] = sgn*prob_parm.vmean_rad*time_perp*0.0168;
        }
      }
      else{
        s_ext[VELX] = sgn*prob_parm.vmean_rad*time_perp;
      } 
      s_ext[VELY] = 0.0;
      s_ext[VELZ] = 0.0;

    }
  }
  else if (idir ==1){
    amrex::Real p_cgs = prob_parm.P_mean * 10.0;
    amrex::Real rho_cgs = 0.0;
    amrex::Real RhoH_temp = 0.0;

    s_ext[TEMP] = prob_parm.T_mean;
    eos.PYT2R(p_cgs, Y_main, s_ext[TEMP], rho_cgs);
    s_ext[DENSITY] = rho_cgs * 1.0e3;
    eos.TY2H(s_ext[TEMP], Y_main, RhoH_temp);
    s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
    for (int n = 0; n < NUM_SPECIES; n++) {
      s_ext[FIRSTSPEC + n] = Y_main[n] * s_ext[DENSITY];
    }

    s_ext[VELX] = 0.0;
    s_ext[VELY] = sgn*prob_parm.vmean_tan*time_tan;
    s_ext[VELZ] = 0.0;
  }
  else if (idir == 2){
    s_ext[TEMP] = prob_parm.T_mean;
  }

  }

  // ACtive control for rz based on Tmax
  AMREX_FORCE_INLINE
  static void update_rz_from_Tmax(amrex::Real Tmax,  amrex::Real dt, MyProbParm&  prob_parm)
  {
    amrex::Real dt_eff = dt * prob_parm.Tctrl_int;

    // amrex::Real alpha = dt_eff / prob_parm.Tmeas_tau;
    // alpha = amrex::min(1.0, alpha);
    // prob_parm.Tfilt = (1-alpha)*prob_parm.Tfilt + alpha*Tmax;
    
    // amrex::Real err = prob_parm.Tfilt - prob_parm.T_target;
    amrex::Real err = Tmax - prob_parm.T_target;

    // PI state
    prob_parm.ierr += err * dt_eff;

    // Additive PI on rz
    amrex::Real rz_cmd = prob_parm.rz + prob_parm.Kp * err + prob_parm.Ki * prob_parm.ierr;

    // Camp to physical/allowed range
    rz_cmd = amrex::min(prob_parm.rz_max, amrex::max(prob_parm.rz_min, rz_cmd));

    amrex::Real rz_new = rz_cmd;
    if(prob_parm.tau > 0.0){
      rz_new = prob_parm.rz + (dt_eff / prob_parm.tau) * (rz_cmd - prob_parm.rz);
    }


    amrex::Real dr = rz_new - prob_parm.rz;
    if (prob_parm.drz_max > 0.0) {
      dr = amrex::min(prob_parm.drz_max, amrex::max(-prob_parm.drz_max, dr));
    }

    prob_parm.rz += dr;
    
    if (prob_parm.Tctrl_verbose && amrex::ParallelDescriptor::IOProcessor()){
      amrex::Print() << "Tctrl: Tmax=" << Tmax << " err=" << err << " rz=" << prob_parm.rz << "\n";
    }
  }

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal_eb(
    const amrex::Real xEBface[AMREX_SPACEDIM],
    const amrex::Real* /*xEBfaceNormal[AMREX_SPACEDIM]*/,
    const amrex::CellData<const amrex::Real>& s_in,
    amrex::Real s_ext[NVAR],
    const amrex::Real time,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm)
  {
    amrex::Real Tg = s_in[TEMP];

    amrex::Real Tguess = prob_parm.wall_a * Tg + prob_parm.wall_b;
    
    amrex::Real hRad =
        prob_parm.eps * prob_parm.sigma *
        (Tguess + prob_parm.Tinf) *
        (Tguess*Tguess + prob_parm.Tinf*prob_parm.Tinf);
    amrex::Real hEff = prob_parm.hExt + hRad;
    
    amrex::Real Gout =
        1.0 / (prob_parm.Rcond + prob_parm.Ri / (prob_parm.Ro * hEff));
    
    amrex::Real Twall =
        (prob_parm.Ggas * Tg + Gout * prob_parm.Tinf) /
        (prob_parm.Ggas + Gout);
    
    //amrex::Real Twall = Tguess;
    Twall = amrex::max(prob_parm.Twall_min, amrex::min(prob_parm.Twall_max, Twall)); 
    s_ext[TEMP] = Twall;
    //    s_ext[TEMP] = prob_parm.T_mean;
  }

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bctype_eb(
    const amrex::Real xEBface[AMREX_SPACEDIM],
    int& EBflagType,
    amrex::Real& EBfaceFrac,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm)
  {
    // IsoT temperature
    EBflagType = pelelmex::BCTypeEB::wall_isot;
    EBfaceFrac = 1.0; 
  }

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void zero_visc(
    int i,
    int j,
    int k,
    amrex::Array4<amrex::Real> const& beta,
    amrex::GeometryData const& geomdata,
    amrex::Box const& domainBox,
    const int dir,
    const int beta_comp,
    const int nComp,
    MyProbParm const& /*prob_parm*/)
  {
     amrex::ignore_unused(i,j,k,beta,geomdata,domainBox,dir,beta_comp,nComp);
  }
  

static void EBUserDefined(
    const amrex::Geometry& geom,
    const int required_coarsening_level,
    const int max_coarsening_level)
  {
    // ParmParse your geometry parameters
    amrex::ParmParse pp("eb2");
    
    amrex::Real injtype = 0;
    pp.query("injtype",injtype);
  
    amrex::Real cyl_R = 0.5;
    pp.query("cyl_R",cyl_R);
  
    amrex::Real cyl_dir = 2;
    pp.query("cyl_dir",cyl_dir);

    amrex::Real cyl_x = 0.0;
    pp.query("cyl_x",cyl_x);
  
    amrex::Real cyl_y = 0.0;
    pp.query("cyl_y",cyl_y);
  
    amrex::Real cyl_z = 0.0;
    pp.query("cyl_z",cyl_z);
  
    amrex::Real rperp = 0.1;
    pp.query("rperp",rperp);
  
    amrex::Real rtan = 0.002;
    pp.query("rtan", rtan);
    amrex::Real rtanb = rtan; 
  
    amrex::Real ztan = 0.0;
    pp.query("ztan", ztan);
  
    amrex::Real zperp = 0.0;
    pp.query("zperp", zperp);
    
    amrex::Real zhi = 0.0;
    zhi = geom.ProbHi(2);
  
    amrex::Real xhi, xhi_2;
    xhi   = geom.ProbHi(0);
    xhi_2 = xhi/2.0;
  
    amrex::Real lbox_tan;
    lbox_tan = sqrt(M_PI*rtan*rtan);
    amrex::Real lbox_tan_o2;
    lbox_tan_o2 = lbox_tan/2.0;
  
    amrex::Real tanshift = 0.002; //0.004;
    amrex::Real lengthshift = -0.005;
    // Fillet radii (set to 0 to disable).  These are interpreted as a smoothing length (meters).
    amrex::Real fillet_r = 0.0;
    pp.query("fillet_r", fillet_r);        // between cylinder1 and cylinders2–5
    amrex::Real fillet_r_16 = 0.0;
    pp.query("fillet_r_16", fillet_r_16);  // between (cylinder1 ∪ cylinders2–5) and cylinder6

    // NOTE: AMReX EB implicit-function convention is: <0 fluid, >0 body, =0 boundary.
    // We build the FLUID region as the union of 6 cylinders, optionally with fillets at the junctions.
    // Using the custom SDFs below avoids the union/intersection sign confusion when using makeUnion/makeIntersection.

    const int idir_main = static_cast<int>(cyl_dir);

    eb_fillet::InfiniteCylinderSDF c1{cyl_R, idir_main, amrex::RealArray{cyl_x, cyl_y, cyl_z}};

    const amrex::Real halfL_tan = 0.5 * (1.5 * xhi);      // CylinderIF uses height/2 from center
    const amrex::Real z_tan_c   = zhi + ztan + rtanb;

    eb_fillet::CappedCylinderSDF c2{rperp, halfL_tan, 0,
                                    amrex::RealArray{-xhi,  xhi-rtanb-tanshift,  z_tan_c}};
    eb_fillet::CappedCylinderSDF c3{rperp, halfL_tan, 0,
                                    amrex::RealArray{ xhi, -xhi+rtanb+tanshift,  z_tan_c}};
    eb_fillet::CappedCylinderSDF c4{rperp, halfL_tan, 1,
                                    amrex::RealArray{-xhi+rtanb+tanshift, -xhi,  z_tan_c}};
    eb_fillet::CappedCylinderSDF c5{rperp, halfL_tan, 1,
                                    amrex::RealArray{ xhi-rtanb-tanshift,  xhi,  z_tan_c}};

    const amrex::Real halfL_perp = 0.5 * (xhi * 8.0);
    eb_fillet::CappedCylinderSDF c6{rperp, halfL_perp, 0,
                                    amrex::RealArray{0.0, 0.0, zhi + zperp}};

    eb_fillet::MWReactorIF MWreactor{c1, c2, c3, c4, c5, c6, fillet_r, fillet_r_16};

    auto gshop = amrex::EB2::makeShop(MWreactor);
    // Build geom using EB2::Build
    amrex::EB2::Build(gshop, geom, required_coarsening_level, max_coarsening_level);
  }


 static void patchFlowVariables(
    const amrex::Geometry& geom,
    MyProbParm const& prob_parm, // MyProbParm const& lprobparm,
    amrex::MultiFab& a_mf)
  {

    amrex::Print() << "\nPatching flow variables..";

    const amrex::Real* prob_lo = geom.ProbLo();
    const amrex::Real* prob_hi = geom.ProbHi();
    const amrex::Real* dx = geom.CellSize();
    for (amrex::MFIter mfi(a_mf, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box& bx = mfi.tilebox();

      const amrex::GpuArray<amrex::Real, 3> ic = {{
        (0.0- prob_lo[0]) / dx[0] - 0.5,
        (0.0- prob_lo[1]) / dx[1] - 0.5,
        (0.0  - prob_lo[2]) / dx[2] - 0.5
      }};
      auto s_arr = a_mf.array(mfi);
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          if (s_arr(i,j,k,TEMP) > 1.0e-16) {
          auto eos = pele::physics::PhysicsType::eos();
          amrex::Real xN2 = s_arr(i,j,k,FIRSTSPEC + N2_ID);
          amrex::Real xO2 = s_arr(i,j,k,FIRSTSPEC + O2_ID);
     
          // normalize as fractions
          amrex::Real xsum = xN2 + xO2;
          if (xsum > 1.0e-16) {
            xN2 /= xsum;
            xO2 /= xsum;
          } else {
            xN2 = 0.0;
            xO2 = 1.0;
          }
          amrex::Real allvar[NVAR] = {0.0};
          for (int n=0; n < NVAR; ++n){
            allvar[n] = s_arr(i, j, k, n);
          }
          amrex::Real X[NUM_SPECIES] = {0.0}; 
          X[N2_ID] = xN2;
          X[O2_ID] = xO2;
          amrex::Real Y[NUM_SPECIES] = {0.0};
          //eos.X2Y(X, Y); 
          for (int n = 0; n < NUM_SPECIES; ++n) {
            Y[n] = 0.0;
            Y[n]=X[n];
          }
          amrex::Real T = 293.15;
          T = s_arr(i,j,k,TEMP);
          amrex::Real rho_cgs = 0.0;
          
          // use mean thermodynamic pressure from the problem parameters
          amrex::Real P_cgs = prob_parm.P_mean * 10.0;
          eos.PYT2R(P_cgs, Y, T, rho_cgs);
          amrex::Real rho = rho_cgs*1.0e3;
          s_arr(i,j,k,DENSITY) = rho;
          for (int n = 0; n < NUM_SPECIES; ++n) {
            s_arr(i,j,k,FIRSTSPEC+n) = Y[n];
          }
          
          amrex::Real htemp = 0.0;
          eos.TY2H(T, Y, htemp);
          s_arr(i,j,k,RHOH) = rho * htemp * 1.0e-4;   
          s_arr(i,j,k,RHORT) = prob_parm.P_mean; //rho * htemp * 1.0e-4;   
          
          }
          else{

           for (int n=0; n<NVAR; n++){
             s_arr(i,j,k,n) = 0.0; 
           }
          }
        });
    }

    amrex::Print() << "Done\n";
  }

  static void modify_ext_sources(
    amrex::Real time,
    amrex::Real /*dt*/,
    const amrex::MultiFab& state_old,
    const amrex::MultiFab& state_new,
    std::unique_ptr<amrex::MultiFab>& ext_src,
    const amrex::GeometryData& geomdata,
    const MyProbParm* prob_parm_d,
    const int sdcIter,
    const int nSDCmax); 
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
