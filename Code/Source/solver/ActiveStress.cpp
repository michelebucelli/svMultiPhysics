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
  global_aitken_relaxation_enabled_ =
      params.get_global_aitken_relaxation_enabled();

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

  svmp::check<svmp::ParseException>(
      aitken_relaxation_enabled_ || !global_aitken_relaxation_enabled_,
      "Active stress Global_Aitken_relaxation requires Aitken_relaxation to be "
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
  cm.bcast(cm_mod, &global_aitken_relaxation_enabled_);

  distribute_model_specific_parameters(cm_mod, cm);
}

void ActiveStress::init(const unsigned int tnNo,
                        const Vector<double> &owned_nodes_) {
  owned_nodes = owned_nodes_;

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
  relaxation.resize(tnNo);
  residual.resize(tnNo);

  if (aitken_relaxation_enabled_)
    previous_residual.resize(tnNo);
}

void ActiveStress::time_advance() {
  states_at_time_step_start = states;

  relaxation = relaxation_coefficient;
  global_relaxation = relaxation_coefficient;
  previous_residual_available = false;
}

void ActiveStress::update(const CmMod &cm_mod, const cmType &cm, const double t,
                          const double dt, const Vector<double> &calcium,
                          const Vector<double> &fiber_stretch,
                          const Vector<double> &fiber_stretch_rate) {
  time = t;

  // Advance the state from the beginning of the time step, and compute the
  // residual of the fixed-point iteration on the active tension.
  for (int i = 0; i < residual.size(); ++i) {
    Vector<double> state_loc = states_at_time_step_start.col(i);
    advance_time_step_local(t, dt, calcium[i], fiber_stretch[i],
                            fiber_stretch_rate[i], state_loc);
    states.set_col(i, state_loc);

    residual[i] = compute_active_tension_local(state_loc, fiber_stretch[i]) -
                  active_tension[i];
  }

  update_relaxation(cm_mod, cm);

  for (int i = 0; i < residual.size(); ++i)
    active_tension[i] += relaxation[i] * residual[i];

  if (aitken_relaxation_enabled_) {
    previous_residual = residual;
    previous_residual_available = true;
  }
}

void ActiveStress::update_relaxation(const CmMod &cm_mod, const cmType &cm) {
  // Aitken's method needs the residuals of two consecutive iterations, so the
  // first iteration of a time step keeps the coefficient set by time_advance.
  if (!aitken_relaxation_enabled_ || !previous_residual_available)
    return;

  if (!global_aitken_relaxation_enabled_) {
    // Node-wise: the scalar form of Aitken's method at every node. The formula
    // divides by the difference between the two residuals, so the coefficient
    // of the previous iteration is kept where they coincide exactly, which is
    // the case at every node whose active tension has stopped changing.
    for (int i = 0; i < residual.size(); ++i)
      if (residual[i] != previous_residual[i])
        relaxation[i] *=
            -previous_residual[i] / (residual[i] - previous_residual[i]);

    return;
  }

  // Global: the vector form of Aitken's method, giving one coefficient for the
  // whole mesh. The difference between the two residuals is zeroed at the nodes
  // another process contributes, which is enough for both inner products below
  // because the difference appears in each of them, and because the mask is
  // made of zeros and ones and is therefore left unchanged by squaring.
  Vector<double> difference(residual.size());
  for (int i = 0; i < residual.size(); ++i)
    difference[i] = owned_nodes[i] * (residual[i] - previous_residual[i]);

  Vector<double> inner_products(2);
  inner_products(0) = previous_residual * difference;
  inner_products(1) = difference * difference;
  inner_products = cm.reduce(cm_mod, inner_products);

  if (inner_products(1) != 0.0)
    global_relaxation *= -inner_products(0) / inner_products(1);

  relaxation = global_relaxation;
}