// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "ActiveStress.h"

bool supports_active_stress(const consts::EquationType eq_type) {
  return eq_type == consts::EquationType::phys_struct ||
         eq_type == consts::EquationType::phys_ustruct ||
         eq_type == consts::EquationType::phys_FSI;
}

void ActiveStress::read_parameters(const ActiveStressParameters &params) {
  eta_f = params.get_eta_f();
  eta_s = params.get_eta_s();
  eta_n = params.get_eta_n();

  implicit_coupling_ = params.get_implicit_coupling();
  relaxation_coefficient = params.get_relaxation_coefficient();
  aitken_relaxation_enabled_ = params.get_aitken_relaxation_enabled();

  svmp::check<svmp::ParseException>(
      relaxation_coefficient > 0.0 && relaxation_coefficient <= 1.0,
      "Active stress relaxation coefficient must be in (0, 1], but got " +
          std::to_string(relaxation_coefficient) + ".");

  // With explicit coupling the model is updated once per time step, so a
  // relaxation coefficient below 1 would low-pass filter the active tension in
  // time instead of damping a fixed-point iteration.
  svmp::check<svmp::ParseException>(
      implicit_coupling_ || relaxation_coefficient == 1.0,
      "Active stress relaxation coefficient must be 1 when Implicit_coupling "
      "is disabled, but got " +
          std::to_string(relaxation_coefficient) + ".");

  // Aitken's method estimates the relaxation coefficient from the residuals of
  // two consecutive fixed-point iterations, which only exist when the coupling
  // is implicit.
  svmp::check<svmp::ParseException>(
      implicit_coupling_ || !aitken_relaxation_enabled_,
      "Active stress Aitken relaxation requires Implicit_coupling to be "
      "enabled.");

  read_model_specific_parameters(
      params.get_parameters(params.get_model_name()));
}

void ActiveStress::distribute_parameters(const CmMod &cm_mod,
                                         const cmType &cm) {
  cm.bcast(cm_mod, &eta_f);
  cm.bcast(cm_mod, &eta_s);
  cm.bcast(cm_mod, &eta_n);

  cm.bcast(cm_mod, &implicit_coupling_);
  cm.bcast(cm_mod, &relaxation_coefficient);
  cm.bcast(cm_mod, &aitken_relaxation_enabled_);

  distribute_model_specific_parameters(cm_mod, cm);
}

void ActiveStress::init(const unsigned int tnNo) {
  states.resize(n_states, tnNo);

  if (n_states > 0) {
    Vector<double> state_loc(n_states);
    init_local(state_loc);

    for (unsigned int i = 0; i < tnNo; ++i)
      for (unsigned int j = 0; j < n_states; ++j)
        states(j, i) = state_loc(j);
  }

  states_at_time_step_start.resize(n_states, tnNo);
  states_at_time_step_start = states;

  active_tension.resize(tnNo);

  if (aitken_relaxation_enabled_) {
    aitken_relaxation.resize(tnNo);
    previous_residual.resize(tnNo);
  }
}

void ActiveStress::time_advance() {
  states_at_time_step_start = states;

  if (aitken_relaxation_enabled_)
    aitken_relaxation = relaxation_coefficient;

  previous_residual_available = false;
}

void ActiveStress::update(const double t, const double dt,
                          const Vector<double> &calcium,
                          const Vector<double> &fiber_stretch,
                          const Vector<double> &fiber_stretch_rate) {
  time = t;

  for (int i = 0; i < active_tension.size(); ++i) {
    Vector<double> state_loc = states_at_time_step_start.col(i);
    advance_time_step_local(t, dt, calcium[i], fiber_stretch[i],
                            fiber_stretch_rate[i], state_loc);
    states.set_col(i, state_loc);

    // Residual of the fixed-point iteration on the active tension.
    const double residual =
        compute_active_tension_local(state_loc, fiber_stretch[i]) -
        active_tension[i];

    double omega = relaxation_coefficient;

    if (aitken_relaxation_enabled_) {
      // Node-wise Aitken estimate. The formula divides by the difference
      // between the two residuals, so the coefficient of the previous
      // iteration is kept where they coincide exactly, which is the case at
      // every node whose active tension has stopped changing.
      if (previous_residual_available && residual != previous_residual[i])
        aitken_relaxation[i] *=
            -previous_residual[i] / (residual - previous_residual[i]);

      omega = aitken_relaxation[i];
      previous_residual[i] = residual;
    }

    active_tension[i] += omega * residual;
  }

  previous_residual_available = true;
}