#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>
#include <math.h>

#ifdef PELE_USE_MWSOLVER

#include "lut_plasma_props.h"

#endif


void
PeleLM::readProbParm()
{
  amrex::ParmParse pp("prob");

  pp.query("P_mean", prob_parm->P_mean);
  pp.query("T_mean", prob_parm->T_mean);
  pp.query("T_wall", prob_parm->T_wall);
  
  pp.query("power", prob_parm->power);
  pp.query("power_time", prob_parm->power_time);
  pp.query("power_shape", prob_parm->power_shape);
  pp.query("power_radiusx", prob_parm->power_radiusx);
  pp.query("power_radiusy", prob_parm->power_radiusy);
  pp.query("power_radiusz", prob_parm->power_radiusz);
  pp.query("z_center_waveguide", prob_parm->z_center_waveguide);
  pp.query("power_shape", PeleLM::prob_parm->power_shape);

  #ifdef PELE_USE_MWSOLVER

  pp.query("EM_t_start", PeleLM::prob_parm->EM_t_start);
  pp.query("flag_solve_EM", PeleLM::prob_parm->flag_solve_EM);
  pp.query("flag_optimize_reflector", PeleLM::prob_parm->flag_optimize_reflector);
  pp.query("plot_harps_int", PeleLM::prob_parm->plot_harps_int);
  pp.query("harps_verbose", PeleLM::prob_parm->harps_verbose);
  pp.query("normalize_power", PeleLM::prob_parm->normalize_power);
  pp.query("harps_verbose", PeleLM::prob_parm->harps_verbose);

  std::string plot_harps_file_str; 
  pp.query("plot_harps_file", plot_harps_file_str); std::strncpy(prob_parm->plot_harps_file.data(), plot_harps_file_str.c_str(), 64); prob_parm->plot_harps_file[63] = '\0'; // Ensure null-termination

  #endif

  amrex::ParmParse pp2("eb2");
  pp2.query("ztan", prob_parm->ztan);
  pp2.query("rperp", prob_parm->rperp);
  pp2.query("cyl_R", prob_parm->cyl_R);
  pp2.query("zperp", prob_parm->zperp);
  pp2.query("rtan", prob_parm->rtan);
  pp2.query("ntan", prob_parm->ntan);
  pp2.query("nperp", prob_parm->nperp);

  amrex::Real tanflux_sccm = 0.0;
  amrex::Real cubicmeter_minute = 0.0;
  pp.query("tanflux_sccm", tanflux_sccm);
  if (tanflux_sccm > 0) {
    cubicmeter_minute = tanflux_sccm * 1e-6 * prob_parm->T_mean * 100000 / 273.15 / prob_parm->P_mean;
    prob_parm->vmean_tan = cubicmeter_minute / prob_parm->ntan / (prob_parm->rtan * prob_parm->rtan * M_PI) / 60.0;
  }

  amrex::Real perpflux_sccm = 0.0;
  pp.query("perpflux_sccm", perpflux_sccm);
  if (perpflux_sccm > 0) {
    cubicmeter_minute = perpflux_sccm * 1e-6 * prob_parm->T_mean * 100000 / 273.15 / prob_parm->P_mean;
    prob_parm->vmean_rad = cubicmeter_minute / prob_parm->nperp / (M_PI * prob_parm->rperp * prob_parm->rperp) / 60.0;
  }

  amrex::ParmParse ppc("control");
  pp.query("power_radiusz", prob_parm->rz);
  ppc.query("Time_ctrl", prob_parm->Time_ctrl);
  ppc.query("Tctrl_on", prob_parm->Tctrl_on);
  ppc.query("Tctrl_verbose", prob_parm->Tctrl_verbose);
  ppc.query("Tctrl_int", prob_parm->Tctrl_int);
  ppc.query("T_target", prob_parm->T_target);
  ppc.query("tau", prob_parm->tau);
  ppc.query("rz_min", prob_parm->rz_min);
  ppc.query("rz_max", prob_parm->rz_max);
  ppc.query("drz_max", prob_parm->drz_max);
  ppc.query("Kp", prob_parm->Kp);
  ppc.query("Ki", prob_parm->Ki);

  amrex::ParmParse psp("sponge");
  psp.query("on", prob_parm->sponge_on);
  psp.query("zbeg", prob_parm->sponge_zbeg);
  psp.query("zend", prob_parm->sponge_zend);
  psp.query("r", prob_parm->sponge_r);
  psp.query("sigma_para", prob_parm->sponge_sigma_para);
  psp.query("sigma_perp", prob_parm->sponge_sigma_perp);
  psp.query("shape", prob_parm->sponge_shape);

  amrex::Real Ri = 0.013;
  amrex::Real Ro = 0.015;
  amrex::Real ks = 1.4;
  prob_parm->hExt = 10.0;

  amrex::Real Tinf = 300.0;
  amrex::Real kg = 0.03;

  prob_parm->sigma = 5.670374419e-8;
  prob_parm->eps = 0.85;

  amrex::Real dxmin = 0.0004375;
  amrex::Real dn = 0.5 * dxmin;

  amrex::Real Rcond = Ri * std::log(Ro / Ri) / ks;
  amrex::Real Rconv = Ri / (Ro * prob_parm->hExt);

  prob_parm->Gout = 1.0 / (Rcond + Rconv);
  prob_parm->Ggas = kg / dn;
  prob_parm->Tinf = Tinf;
  prob_parm->Ri = Ri;
  prob_parm->Ro = Ro;
  prob_parm->Rcond = Rcond;
  prob_parm->wall_a = prob_parm->Ggas / (prob_parm->Ggas + prob_parm->Gout);
  prob_parm->wall_b = prob_parm->Gout * prob_parm->Tinf / (prob_parm->Ggas + prob_parm->Gout);
}


void ProblemSpecificFunctions::modify_ext_sources(
  amrex::Real time,
  amrex::Real /*dt*/,
  const amrex::MultiFab& state_old,
  const amrex::MultiFab& state_new,
  std::unique_ptr<amrex::MultiFab>& ext_src,
  const amrex::GeometryData& geomdata,
  const MyProbParm* prob_parm_d,
  const int sdcIter,
  const int nSDCmax)
{
  const auto prob_lo = geomdata.ProbLo();
  const amrex::Real* dx = geomdata.CellSize();
  auto ext_src_rhoh_a = ext_src->arrays();
  auto ext_src_a = ext_src->arrays();
  auto const& state_old_a = state_old.const_arrays();
 
  const int power_shape = prob_parm_d->power_shape;
  const amrex::Real power = prob_parm_d->power;
  const amrex::Real power_time = prob_parm_d->power_time;
  amrex::Real total_power = 0.0;
  const amrex::Real radiusx = prob_parm_d->power_radiusx;
  const amrex::Real radiusy = prob_parm_d->power_radiusy;
  amrex::Real radiusz = prob_parm_d->power_radiusz;
  const amrex::Real center = prob_parm_d->power_center;
  
  
  const amrex::GpuArray<amrex::Real, 3> ic = {{
    (0.0- prob_lo[0]) / dx[0] - 0.5,
    (0.0- prob_lo[1]) / dx[1] - 0.5,
    (center  - prob_lo[2]) / dx[2] - 0.5
  }};

  auto addsponge = prob_parm_d->sponge_on;                                                                                                                                               
  const amrex::Real zbeg = prob_parm_d->sponge_zbeg;
  const amrex::Real zend = prob_parm_d->sponge_zend;
  const amrex::Real sponge_r = prob_parm_d->sponge_r;
  const amrex::Real sponge_sigma_perp = prob_parm_d->sponge_sigma_perp;
  const amrex::Real sponge_sigma_para = prob_parm_d->sponge_sigma_para;
  const amrex::Real sponge_T_sigma = prob_parm_d->sponge_T_sigma;
  const amrex::Real sponge_T_ref = prob_parm_d->sponge_T_ref;
  const int sponge_shape = prob_parm_d->sponge_shape;

  if (addsponge) {
    ParallelFor(
      *ext_src,
      [=, dx=dx, prob_lo=prob_lo,
       state_old_a=state_old_a, ext_src_a=ext_src_a]
      AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
      {    
        amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
        amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
        amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
        amrex::Real r2 = x * x + y * y;
        bool in_r = (r2 <= sponge_r * sponge_r);
        bool in_z = (z >= zbeg && z <= zend);
        if (in_r && in_z) {
          amrex::Real eta = (zend - z) / amrex::max(zend - zbeg, 1.0e-12);
  //          amrex::Real eta = (z - zbeg) / amrex::max(zend - zbeg, 1.0e-12);
  //          eta = amrex::max(0.0, amrex::min(1.0, eta));
          amrex::Real ramp = eta;
          if (sponge_shape == 2) {
            ramp = eta * eta;
          } else if (sponge_shape == 3) {
            ramp = eta * eta * eta;
          }
          amrex::Real rho = state_old_a[box_no](i, j, k, DENSITY);
          amrex::Real u   = state_old_a[box_no](i, j, k, VELX);
          amrex::Real v   = state_old_a[box_no](i, j, k, VELY);
          amrex::Real w   = state_old_a[box_no](i, j, k, VELZ);
          amrex::Real T   = state_old_a[box_no](i, j, k, TEMP);
          ext_src_a[box_no](i, j, k, VELX) += -rho * ramp * sponge_sigma_perp * u;
          ext_src_a[box_no](i, j, k, VELY) += -rho * ramp * sponge_sigma_perp * v;
          ext_src_a[box_no](i, j, k, VELZ) += -rho * ramp * sponge_sigma_para * amrex::max(0.0, w);
          if (sponge_T_sigma > 0.0) {
            ext_src_a[box_no](i, j, k, TEMP) +=
              -ramp * sponge_T_sigma * (T - sponge_T_ref);
          }
        }  
      });  
  }  

  if (time > prob_parm_d->Time_ctrl) {
    radiusz = prob_parm_d->rz;
  }
  total_power = power * amrex::min(time/power_time, 1.0);
  if (radiusx <= 0.0|| radiusy <= 0.0|| radiusz <= 0.0||
      total_power == 0.0) {
    amrex::ParallelFor(
      *ext_src,
      [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
      {
        ext_src_rhoh_a[box_no](i, j, k, RHOH) = 0.0;
      });
    amrex::Gpu::streamSynchronize();
    return;
  }

  const amrex::Box bbox = ext_src->boxArray().minimalBox();

  const int ilo = bbox.smallEnd(0);
  const int ihi = bbox.bigEnd(0);
  const int jlo = bbox.smallEnd(1);
  const int jhi = bbox.bigEnd(1);
  const int klo = bbox.smallEnd(2);
  const int khi = bbox.bigEnd(2);

  if (power_shape == 1) {
    // Exact discrete normalization for
    // max(1-x^2/rx^2,0) * max(1-y^2/ry^2,0) * max(1-z^2/rz^2,0)

    amrex::Real Sx = 0.0;
    amrex::Real Sy = 0.0;
    amrex::Real Sz = 0.0;

    for (int i = ilo; i <= ihi; ++i) {
      const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
      Sx += amrex::max(1.0- (x * x) / (radiusx * radiusx), 0.0);
    }

    for (int j = jlo; j <= jhi; ++j) {
      const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
      Sy += amrex::max(1.0- (y * y) / (radiusy * radiusy), 0.0);
    }

    for (int k = klo; k <= khi; ++k) {
      const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];
      Sz += amrex::max(1.0- (z * z) / (radiusz * radiusz), 0.0);
    }

    const amrex::Real volume = dx[0] * dx[1] * dx[2] * Sx * Sy * Sz;

    amrex::ParallelFor(
      *ext_src,
      [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
      {
        const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
        const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
        const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];

        const amrex::Real fx =
          amrex::max(1.0- (x * x) / (radiusx * radiusx), 0.0);
        const amrex::Real fy =
          amrex::max(1.0- (y * y) / (radiusy * radiusy), 0.0);
        const amrex::Real fz =
          amrex::max(1.0- (z * z) / (radiusz * radiusz), 0.0);

        const amrex::Real src =
          (volume > 0.0) ? (total_power / volume) * fx * fy * fz : 0.0;

        ext_src_rhoh_a[box_no](i, j, k, RHOH) = src;
      });
  }
  else if (power_shape == 2) {
    amrex::Real Sx = 0.0;
  amrex::Real Sy = 0.0;
  amrex::Real Sz = 0.0;

  for (int i = ilo; i <= ihi; ++i) {
    const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
    Sx += amrex::max(1.0 - (x * x) / (radiusx * radiusx), 0.0);
  }

  for (int j = jlo; j <= jhi; ++j) {
    const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
    Sy += amrex::max(1.0 - (y * y) / (radiusy * radiusy), 0.0);
  }

  // Flat top (constant) in z: 1.0 if |z| <= radiusz, else 0.0
  for (int k = klo; k <= khi; ++k) {
    const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];
    if (std::abs(z) <= radiusz) Sz += 1.0;
  }

  const amrex::Real volume = dx[0] * dx[1] * dx[2] * Sx * Sy * Sz;

  amrex::ParallelFor(
      *ext_src,
      [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
      {
          const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
          const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
          const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];

          const amrex::Real fx = amrex::max(1.0 - (x * x) / (radiusx * radiusx), 0.0);
          const amrex::Real fy = amrex::max(1.0 - (y * y) / (radiusy * radiusy), 0.0);
          const amrex::Real fz = (amrex::Math::abs(z) <= radiusz) ? 1.0 : 0.0;

          const amrex::Real src = (volume > 0.0) ? (total_power / volume) * fx * fy * fz : 0.0;

          ext_src_rhoh_a[box_no](i, j, k, RHOH) = src;
      }
    );
  }
  else if (power_shape == 3){
    // crazy profile
    amrex::ParallelFor(
      *ext_src,
      [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
      {
          const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
          const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
          const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];
          
          const amrex::Real radius = sqrt(x*x + y*y);
          const amrex::Real R_r_sq = radiusx * radiusy;
          const amrex::Real R_z_sq = radiusz * radiusz;

          // Unified ellipsoidal distance metric
          const amrex::Real metric = (radius * radius) / R_r_sq + (z * z) / R_z_sq;

          // Peaks at metric = 1.0 (ellipsoidal shell) with smooth Gaussian roll-off
          const amrex::Real src = std::exp(-(metric - 1.0) * (metric - 1.0));
          ext_src_rhoh_a[box_no](i, j, k, RHOH) = src;
      }
    );

    // Normalize
    amrex::Real total_src = ext_src->sum(RHOH, false) * dx[0] * dx[1] * dx[2];
    amrex::Real norm_factor = (total_src > 0.0) ? (total_power / total_src) : 0.0;
    amrex::ParallelFor(
      *ext_src,
      [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
      {
          ext_src_rhoh_a[box_no](i, j, k, RHOH) *= norm_factor;
      }
    );
  }
  else if (power_shape == 0) {
    // Exact discrete normalization for support:
    // (x/rx)^2 + (y/ry)^2 <= 1  AND  (z/rz)^2 <= 1

    long long ncell = 0;

    for (int k = klo; k <= khi; ++k) {
      const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];
      const amrex::Real rz = (z * z) / (radiusz * radiusz);

      if (rz <= 1.0) {
        for (int j = jlo; j <= jhi; ++j) {
          const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
          const amrex::Real yy = (y * y) / (radiusy * radiusy);

          for (int i = ilo; i <= ihi; ++i) {
            const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
            const amrex::Real xx = (x * x) / (radiusx * radiusx);

            if (xx + yy <= 1.0) {
              ++ncell;
            }
          }
        }
      }
    }

    const amrex::Real volume =
      static_cast<amrex::Real>(ncell) * dx[0] * dx[1] * dx[2];

    amrex::ParallelFor(
      *ext_src,
      [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
      {
        const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
        const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
        const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];

        const amrex::Real rxy =
          (x * x) / (radiusx * radiusx) +
          (y * y) / (radiusy * radiusy);
        const amrex::Real rz =
          (z * z) / (radiusz * radiusz);

        const amrex::Real inside =
          (rxy <= 1.0&& rz <= 1.0) ? 1.0: 0.0;

        const amrex::Real src =
          (volume > 0.0) ? (total_power / volume) * inside : 0.0;

        ext_src_rhoh_a[box_no](i, j, k, RHOH) = src;
      });
  }
  else {
    amrex::ParallelFor(
      *ext_src,
      [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
      {
        ext_src_rhoh_a[box_no](i, j, k, RHOH) = 0.0;
      });
  }

  amrex::Gpu::streamSynchronize();

  // Diagnostic check: total deposited power = sum(src * dV)
  const amrex::Real cell_vol = dx[0] * dx[1] * dx[2];
  const amrex::Real deposited_power = ext_src->sum(RHOH, false) * cell_vol;

  if (amrex::ParallelDescriptor::IOProcessor()) {
    amrex::Print()
      << "Deposited power = " << deposited_power
      << " ; target = " << total_power
      << " ; rel error = "
      << std::abs(deposited_power - total_power) /
         (std::abs(total_power) + 1.0e-50)
      << "\n";
  }
  amrex::Real max_T = state_old.max(TEMP, 0);

  // 2. Print it safely from Processor 0 only
  amrex::Print() << "=== Global Maximum Temperature: " << max_T << " K ===\n";
}





void
PeleLM::freeProbParm()
{
}