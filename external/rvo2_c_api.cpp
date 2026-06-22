#include <cstddef>

#include "RVO.h"

extern "C" int rvo2_compute_velocities(
    int n,
    const double *positions_xy,
    const double *velocities_xy,
    const double *preferred_xy,
    double dt_s,
    double neighbor_dist_m,
    int max_neighbors,
    double time_horizon_s,
    double time_horizon_obst_s,
    double radius_m,
    double max_speed_mps,
    double *out_velocities_xy) {
  if (n <= 0 || positions_xy == nullptr || velocities_xy == nullptr ||
      preferred_xy == nullptr || out_velocities_xy == nullptr) {
    return 1;
  }

  RVO::RVOSimulator simulator(
      static_cast<float>(dt_s),
      static_cast<float>(neighbor_dist_m),
      static_cast<std::size_t>(max_neighbors),
      static_cast<float>(time_horizon_s),
      static_cast<float>(time_horizon_obst_s),
      static_cast<float>(radius_m),
      static_cast<float>(max_speed_mps));

  for (int i = 0; i < n; ++i) {
    const int k = 2 * i;
    const std::size_t agent_no = simulator.addAgent(
        RVO::Vector2(static_cast<float>(positions_xy[k]),
                     static_cast<float>(positions_xy[k + 1])),
        static_cast<float>(neighbor_dist_m),
        static_cast<std::size_t>(max_neighbors),
        static_cast<float>(time_horizon_s),
        static_cast<float>(time_horizon_obst_s),
        static_cast<float>(radius_m),
        static_cast<float>(max_speed_mps),
        RVO::Vector2(static_cast<float>(velocities_xy[k]),
                     static_cast<float>(velocities_xy[k + 1])));
    if (agent_no == RVO::RVO_ERROR) {
      return 2;
    }
  }

  for (int i = 0; i < n; ++i) {
    const int k = 2 * i;
    simulator.setAgentPrefVelocity(
        static_cast<std::size_t>(i),
        RVO::Vector2(static_cast<float>(preferred_xy[k]),
                     static_cast<float>(preferred_xy[k + 1])));
  }

  simulator.doStep();

  for (int i = 0; i < n; ++i) {
    const int k = 2 * i;
    const RVO::Vector2 &velocity =
        simulator.getAgentVelocity(static_cast<std::size_t>(i));
    out_velocities_xy[k] = static_cast<double>(velocity.x());
    out_velocities_xy[k + 1] = static_cast<double>(velocity.y());
  }

  return 0;
}
