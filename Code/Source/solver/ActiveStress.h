// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#ifndef ACTIVE_STRESS_H
#define ACTIVE_STRESS_H

#include "Array.h"
#include "Parameters.h"
#include "Vector.h"
#include "consts.h"
#include "factory.h"

#include "CmMod.h"

#include <memory>

/**
 * @brief Return whether a certain equation type can be solved with active
 * stress.
 */
bool supports_active_stress(const consts::EquationType eq_type);

/**
 * @brief Abstract active stress class.
 *
 * This class provides an interface for defining active stress models, i.e.
 * models that, in the context of structural mechanics of muscular tissue,
 * compute an active tension representing the contribution of muscular
 * contraction to the constitiutive law.
 *
 * The class assumes that the active tension can be expressed as
 * @f[
 *   \Tact = \Tact(t, \calcium, \fiberstretch, \fiberstretchrate,
 *                 \astressstate),
 * @f]
 * where @f$\calcium@f$ is the intracellular calcium concentration,
 * @f$\fiberstretch@f$ is the fiber stretch, @f$\fiberstretchrate@f$ is the
 * fiber stretch rate, and @f$\astressstate@f$ is a vector of internal state
 * variables, representing the state of contraction.
 *
 * The expression assumed above implies that the active tension is a local
 * function of the variables it depends on, that is the active tension at a
 * given point only depends on the value of other variables at that same point.
 * Accordingly, this class works nodally, by evaluating the active tension at
 * every mesh node and storing it in a vector, whose values can be accessed
 * through @ref ActiveStress::get_tension_fibers.
 *
 * ### Directional distribution of active stress
 *
 * In muscular mechanics models, active stress normally acts only along the
 * direction of fibers @f$\fiberdirection@f$, reflecting the fact that
 * contractile units are aligned with fibers. However, one might want to account
 * for fiber dispersion, i.e. the fact that fibers are not perfectly and
 * regularly aligned, but rather have a certain distribution of orientations
 * centered around the principal direction @f$\fiberdirection@f$.
 *
 * This can be surrogated by defining the active stress tensor as
 * @f[
 *   S_\text{act} = \Tact \left(
 *     \eta_f \fiberdirection \otimes \fiberdirection +
 *     \eta_s \sheetdirection \otimes \sheetdirection +
 *     \eta_n \sheetnormaldirection \otimes \sheetnormaldirection
 *   \right),
 * @f]
 * where @f$\sheetdirection@f$ and @f$\sheetnormaldirection@f$ are the sheet and
 * sheet-normal directions, respectively, and @f$\eta_f@f$, @f$\eta_s@f$, and
 * @f$\eta_n@f$ are coefficients that define the distribution the active tension
 * along the three principal directions. The coefficients must be such that
 * @f$\eta_f + \eta_s + \eta_n = 1@f$.
 *
 * This class stores the values of @f$\eta_f@f$, @f$\eta_s@f$ and @f$\eta_n@f$,
 * and provides the functions @ref ActiveStress::get_tension_fibers,
 * @ref ActiveStress::get_tension_sheets and @ref
 * ActiveStress::get_tension_sheet_normals to access @f$\eta_f \Tact@f$,
 * @f$\eta_s \Tact@f$ and @f$\eta_n \Tact@f$, respectively.
 *
 * ### Implementing concrete active stress models
 *
 * To implement a new active stress model, the following steps need to be taken:
 *
 * 1. Create a new class derived from @ref ActiveStress.
 * 2. Override the methods @ref init_local, @ref advance_time_step_local and
 *    @ref compute_active_tension_local, defining the initial condition,
 *    time evolution and active tension computation, respectively, for a single
 *    node.
 * 3. Create a new class derived from @ref ActiveStressModelParameters to store
 *    the parameters specific to the new active stress model.
 * 4. Override the methods @ref get_parameters,
 *    @ref read_model_specific_parameters and
 *    @ref distribute_model_specific_parameters to manage the parameters of the
 *    new active stress model.
 * 5. Register the new class into the active stress model factory by using the
 *    macro @ref REGISTER_ACTIVE_STRESS_MODEL. The macro should be called in a
 *    `.cpp` file, not in a header file.
 *
 * Notice that if the model is expressed in terms of a system of ODEs, it can
 * be implemented by deriving from @ref ActiveStressODE, which already addresses
 * some of the points above.
 *
 * ### Coupling with the mechanics problem
 *
 * The active tension depends on the fiber stretch both directly, through the
 * expression of @f$\Tact@f$, and indirectly, through the state
 * @f$\astressstate@f$, which is itself driven by the fiber stretch. The
 * mechanics problem, in turn, depends on the active tension.
 *
 * Every time step begins with a call to @ref time_advance, which stores the
 * state as the initial condition of the step. The state and the active tension
 * are then computed by @ref update, which can be called any number of times
 * within the step, always restarting from that stored state.
 *
 * By default the two-way coupling is treated explicitly: @ref update is called
 * once per time step, before the nonlinear iterations of the mechanics problem,
 * with the fiber stretch of the previous time step. If @c Implicit_coupling is
 * enabled, @ref update is called again at every nonlinear iteration with the
 * fiber stretch of the current displacement iterate, so that at convergence the
 * active tension and the displacement satisfy the coupled problem at the same
 * time level. The coupling is closed by a fixed-point iteration rather than by
 * including the derivative of the active tension with respect to the fiber
 * stretch in the tangent matrix. That iteration is generally not contractive on
 * its own, so the active tension is relaxed with the user-specified coefficient
 * @ref relaxation_coefficient.
 *
 * A relaxation coefficient that is small enough to converge everywhere is
 * usually far smaller than needed at most nodes. Enabling @c Aitken_relaxation
 * re-estimates it at every node and every iteration with Aitken's method, using
 * @ref relaxation_coefficient only as the value of the first iteration of each
 * time step. See @ref update for the formula.
 */
class ActiveStress {
public:
  /**
   * @brief Constructor.
   *
   * @param n_states_ Number of state variables for this model.
   * @param needs_fiber_stretch Whether this model uses the fiber stretch
   *   passed to @ref update. This flag can be used to determine whether fiber
   *   stretch computation can be skipped for efficiency.
   * @param needs_fiber_stretch_rate Whether this model uses the fiber stretch
   *   rate passed to @ref update. This flag can be used to determine whether
   *   fiber stretch rate computation can be skipped for efficiency.
   */
  ActiveStress(const unsigned int n_states_, const bool needs_fiber_stretch,
               const bool needs_fiber_stretch_rate)
      : n_states(n_states_), needs_fiber_stretch_(needs_fiber_stretch),
        needs_fiber_stretch_rate_(needs_fiber_stretch_rate) {}

  /**
   * @brief Virtual destructor.
   */
  virtual ~ActiveStress() = default;

  /**
   * @brief Construct an instance of model parameters for this model.
   */
  virtual std::unique_ptr<ActiveStressModelParameters>
  get_parameters() const = 0;

  /**
   * @brief Read model parameters from a parameter object.
   */
  void read_parameters(const ActiveStressParameters &params);

  /**
   * @brief Distribute model parameters to all parallel processes.
   */
  void distribute_parameters(const CmMod &cm_mod, const cmType &cm);

  /**
   * @brief Get the tension along fibers @f$\eta_f \Tact@f$ at a given point.
   */
  double get_tension_fibers(const int idx) const {
    return eta_f * active_tension[idx];
  }

  /**
   * @brief Get the tension along sheets @f$\eta_s \Tact@f$ at a given point.
   */
  double get_tension_sheets(const int idx) const {
    return eta_s * active_tension[idx];
  }

  /**
   * @brief Get the tension along sheet normals @f$\eta_n \Tact@f$ at a given
   * point.
   */
  double get_tension_sheet_normals(const int idx) const {
    return eta_n * active_tension[idx];
  }

  /**
   * @brief Initialize the model.
   *
   * Allocates the internal state vector and initializes it with the model's
   * initial conditions.
   *
   * @param[in] tnNo Total number of mesh nodes for the current rank.
   */
  virtual void init(const unsigned int tnNo);

  /**
   * @brief Begin a new time step.
   *
   * Stores the current state as the initial condition of the time step and
   * resets the Aitken relaxation coefficients to @ref relaxation_coefficient.
   * Must be called once per time step, before any call to @ref update.
   */
  virtual void time_advance();

  /**
   * @brief Update the state and the active tension over the current time step.
   *
   * Advances the state stored by @ref time_advance over one time step, using
   * the given calcium, fiber stretch and fiber stretch rate, and recomputes the
   * active tension at every node.
   *
   * This function may be called more than once per time step: every call
   * restarts from the state stored by @ref time_advance, so the resulting state
   * depends only on the arguments of the last call. The implicit coupling uses
   * this to run a fixed-point iteration, calling this function once per
   * nonlinear iteration of the mechanics problem with an updated fiber stretch.
   *
   * The active tension is relaxed against the value it had before the call. In
   * terms of the fixed-point residual at node @f$i@f$,
   * @f[
   *   r_i^k = \Tact(\astressstate_i^{k+1}, \fiberstretch_i^{k}) - {\Tact}_i^k\;,
   * @f]
   * the update reads
   * @f[
   *   {\Tact}_i^{k+1} = {\Tact}_i^k + \omega_i^k \, r_i^k\;.
   * @f]
   * At the first call of a time step @f${\Tact}_i^k@f$ is the converged active
   * tension of the previous time step.
   *
   * Without Aitken relaxation @f$\omega_i^k@f$ is the constant
   * @ref relaxation_coefficient. With Aitken relaxation enabled it is instead
   * re-estimated at every node from the last two residuals,
   * @f[
   *   \omega_i^{k} = -\omega_i^{k-1} \,
   *     \frac{r_i^{k-1}}{r_i^{k} - r_i^{k-1}}\;,
   * @f]
   * which is the node-wise (scalar) form of Aitken's @f$\Delta^2@f$ method: it
   * is the relaxation that would land exactly on the fixed point if the map
   * were affine at that node. The estimate is kept unchanged where the residual
   * difference is too small to be meaningful, and is clamped to a positive
   * range. @ref relaxation_coefficient provides @f$\omega_i^0@f$, which is reset
   * at the beginning of every time step by @ref time_advance.
   *
   * @param[in] t Current time (i.e. the time instant being advanced to).
   * @param[in] dt Time step size.
   * @param[in] calcium Calcium concentration at every node.
   * @param[in] fiber_stretch Fiber stretch at every node. This is usually
   *   computed with post::fib_stretch.
   * @param[in] fiber_stretch_rate Fiber stretch rate at every node. This is
   *   usually computed with post::fib_stretch_rate.
   */
  virtual void update(const double t, const double dt,
                      const Vector<double> &calcium,
                      const Vector<double> &fiber_stretch,
                      const Vector<double> &fiber_stretch_rate);

  /**
   * @brief Whether this model is updated within the nonlinear iterations of the
   * mechanics problem, i.e. whether the coupling is implicit.
   */
  bool implicit_coupling() const { return implicit_coupling_; }

  /// Number of state variables for this model.
  const unsigned int n_states;

  /**
   * @brief Whether this model uses the fiber stretch passed to @ref update.
   * This flag can be used to determine whether fiber stretch computation can be
   * skipped for efficiency.
   */
  bool needs_fiber_stretch() const { return needs_fiber_stretch_; }

  /**
   * @brief Whether this model uses the fiber stretch rate passed to
   * @ref update. This flag can be used to determine whether fiber stretch rate
   * computation can be skipped for efficiency.
   */
  bool needs_fiber_stretch_rate() const { return needs_fiber_stretch_rate_; }

protected:
  /**
   * @brief Backing store for @ref needs_fiber_stretch.
   */
  bool needs_fiber_stretch_;

  /**
   * @brief Backing store for @ref needs_fiber_stretch_rate.
   */
  bool needs_fiber_stretch_rate_;

  /**
   * @brief Read model parameters from a parameter object.
   *
   * This method needs to be overridden by derived classes to read the
   * parameters specific to the concrete model they implement.
   */
  virtual void
  read_model_specific_parameters(const ActiveStressModelParameters &params) = 0;

  /**
   * @brief Distribute model parameters to all parallel processes.
   *
   * This method needs to be overridden by derived classes to distribute the
   * parameters specific to the concrete model they implement to all parallel
   * processes.
   */
  virtual void distribute_model_specific_parameters(const CmMod &cm_mod,
                                                    const cmType &cm) = 0;

  /**
   * @brief Initialize the state vector for a single node.
   *
   * @param[out] state State vector for a single node, to be initialized by
   *   this function.
   */
  virtual void init_local(Vector<double> &state) const = 0;

  /**
   * @brief Advance in time for a single node.
   *
   * @param[in] t Current time (i.e. the time instant being advanced to).
   * @param[in] dt Time step size.
   * @param[in] calcium Calcium concentration at the current node.
   * @param[in] fiber_stretch Fiber stretch at the current node.
   * @param[in] fiber_stretch_rate Fiber stretch rate at the current node.
   * @param[in,out] state State vector for a single node, to be updated by
   *   this function.
   */
  virtual void advance_time_step_local(const double t, const double dt,
                                       const double calcium,
                                       const double fiber_stretch,
                                       const double fiber_stretch_rate,
                                       Vector<double> &state) const = 0;

  /**
   * @brief Compute the active tension for a single node.
   *
   * @param[in] state State vector for a single node.
   * @param[in] fiber_stretch Fiber stretch at the current node.
   */
  virtual double
  compute_active_tension_local(const Vector<double> &state,
                               const double fiber_stretch) const = 0;

  /// Time instant being advanced to. Set by @ref update.
  double time = 0.0;

  /// State variables for the model.
  Array<double> states;

  /**
   * @brief State variables at the beginning of the current time step.
   *
   * Set by @ref time_advance and used by @ref update as the initial condition
   * of every call within the time step.
   */
  Array<double> states_at_time_step_start;

  /// Active tension at every node.
  Vector<double> active_tension;

  /**
   * @brief Whether this model is updated within the nonlinear iterations of the
   * mechanics problem.
   */
  bool implicit_coupling_;

  /**
   * @brief Relaxation coefficient @f$\omega \in (0, 1]@f$ applied to the active
   * tension by @ref update.
   *
   * With Aitken relaxation enabled this is only the value used at the first
   * call to @ref update of every time step.
   */
  double relaxation_coefficient;

  /**
   * @brief Whether @ref update re-estimates the relaxation coefficient at every
   * node with Aitken's method.
   */
  bool aitken_relaxation_enabled_;

  /**
   * @brief Aitken relaxation coefficient at every node.
   *
   * Reset to @ref relaxation_coefficient by @ref time_advance and re-estimated
   * by every subsequent call to @ref update. Unused when Aitken relaxation is
   * disabled.
   */
  Vector<double> aitken_relaxation;

  /**
   * @brief Fixed-point residual of the active tension at every node, as
   * computed by the previous call to @ref update within the current time step.
   *
   * Unused when Aitken relaxation is disabled.
   */
  Vector<double> previous_residual;

  /**
   * @brief Whether @ref previous_residual holds a residual from the current
   * time step, i.e. whether @ref update has already been called since the last
   * @ref time_advance. Aitken's method needs two residuals, so the first call
   * of a time step keeps @ref aitken_relaxation at its initial value.
   */
  bool previous_residual_available = false;

  /// Active tension coefficient along the fiber direction.
  double eta_f;

  /// Active tension coefficient along the sheet direction.
  double eta_s;

  /// Active tension coefficient along the sheet-normal direction.
  double eta_n;
};

/**
 * @brief Alias for the active stress model factory.
 *
 * See the documentation for @ref Factory for more details on how this works.
 */
using ActiveStressFactory = Factory<ActiveStress>;

/**
 * @brief Macro to register an active stress model in the factory.
 */
#define REGISTER_ACTIVE_STRESS_MODEL(name, type)                               \
  REGISTER_IN_FACTORY(ActiveStress, type, name)

#endif