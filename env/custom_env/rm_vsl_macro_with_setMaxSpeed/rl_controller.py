import numpy as np
import traci

from ..sumo_env import SumoEnv


class RLController(SumoEnv):
    """Variant: joint ramp metering + lane-1 VSL control with macro-only state.

    Combines:
        - The 14-d macro observation from the "macro with lane" variant (MLP).
        - The 42-action joint space + lane-VSL actuation from the active variant.

    Action space is 42 = 7 green-time choices x 6 VSL speeds.
        green_idx = action % 7   -> green time in {10,15,...,40} s
        vsl_idx   = action // 7  -> direct max-speed choice in m/s
    Lane control is applied to `vsl_zone_0` (mainline lane closest to the ramp).
    """

    TARGET_VSL_LANE_ID = "vsl_zone_0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.CYCLE_DURATION_SEC = 40.0

        self.green_time_actions_sec = np.array(
            [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        )
        self.vsl_speed_actions_mps = np.array([13.89, 16.67, 19.44, 22.22, 25.0, 27.78])
        self.action_space_n = len(self.green_time_actions_sec) * len(
            self.vsl_speed_actions_mps
        )  # 42

        self.green_phase_index = 0
        self.red_phase_index = 1

        self.upstream_mainline_all_detector_ids = self.get_edge_induction_loops(
            self.UPSTREAM_EDGE
        )
        self.bottleneck_edge_all_detector_ids = self.get_edge_induction_loops(
            self.MERGING_EDGE
        )
        self.downstream_mainline_all_detector_ids = self.get_edge_induction_loops(
            self.DOWNSTREAM_EDGE
        )

        self.upstream_detector_ids_state = [
            "up_stream_sens_0",
            "up_stream_sens_1",
            "up_stream_sens_2",
        ]
        self.bottleneck_detector_ids_state = [
            "bottle_neck_sens_0",
            "bottle_neck_sens_1",
            "bottle_neck_sens_2",
            "bottle_neck_sens_3",
        ]
        self.outflow_detector_ids_reward = self.downstream_mainline_all_detector_ids
        self.ramp_queue_detector_id = "queue_sens"

        # Macro-only state: 15 features (14 macro + last green + last VSL speed). No micro grid.
        self.observation_space_n = 15

        self.last_green_time_sec = self.green_time_actions_sec[0]
        self.last_vsl_speed_mps = self.vsl_speed_actions_mps[-1]
        self.lane_default_speed_mps = None  # captured at reset (post SUMO start)

        self._reset_cycle_aggregators()

        self.processed_flow_upstream_vph = 0.0
        self.processed_flow_merging_vph = 0.0
        self.processed_mainline_flow_downstream_vph = 0.0

        self.processed_occ_upstream_percent = 0.0
        self.processed_occ_bottleneck_percent = 0.0
        self.processed_occ_downstream_percent = 0.0

        self.processed_speed_bottleneck_mps = 0.0
        self.processed_speed_upstream_mps = 0.0
        self.processed_mainline_speed_downstream_mps = 0.0

        self.processed_ramp_queue_veh = 0.0
        self.sum_queue = 0.0

        self._last_detailed_info = {}
        self._initialize_last_detailed_info_placeholders()

    def _initialize_last_detailed_info_placeholders(self):
        self._last_detailed_info = {
            "mainline_flow_upstream_v/h": 0.0,
            "mainline_occ_upstream_percent": 0.0,
            "mainline_speed_upstream_km/h": 0.0,
            "mainline_flow_mergeArea_v/h": 0.0,
            "mainline_occ_mergeArea_percent": 0.0,
            "mainline_speed_mergeArea_km/h": 0.0,
            "mainline_flow_downstream_v/h": 0.0,
            "mainline_occ_downstream_percent": 0.0,
            "mainline_speed_downstream_km/h": 0.0,
            "ramp_queue_veh": 0.0,
            "current_tl_phase_index": -1,
            "current_tl_ryg_state": "N/A",
            "chosen_green_time_sec": 0.0,
            "chosen_vsl_speed_mps": 0.0,
            "reward_outflow_speed_comp": 0.0,
            "reward_throughput_comp": 0.0,
            "penalty_ramp_queue_comp": 0.0,
            "penalty_bottleneck_occ_comp": 0.0,
            "penalty_spillback_comp": 0.0,
            "sim_time": 0.0,
            "episode": 0,
            "total_running_vehicles": 0,
            "total_departed": 0,
            "total_arrived": 0,
            "l": 0,
            "r": 0.0,
            "TimeLimit.truncated": False,
            "done": False,
        }

    def _reset_cycle_aggregators(self):
        self.sum_interval_upstream_veh_count = 0
        self.sum_interval_merging_veh_count = 0
        self.list_interval_upstream_occ = []
        self.list_interval_upstream_speed = []
        self.list_interval_bottleneck_occ = []
        self.list_interval_bottleneck_speed = []
        self.list_interval_ramp_queue = []
        self.sum_interval_outflow_veh_count = 0
        self.list_interval_outflow_speed = []
        self.sum_queue = 0
        self.current_ramp_queue_veh = 0

    VSL_SPEED_MIN_MPS = 13.89  # 50 km/h – lowest VSL action
    VSL_SPEED_MAX_MPS = 27.78  # 100 km/h – free-flow

    def _speed_to_color(self, speed_mps):
        """Map speed (m/s) to an RGBA tuple via a red → yellow → green gradient."""
        lo, hi = self.VSL_SPEED_MIN_MPS, self.VSL_SPEED_MAX_MPS
        t = max(0.0, min(1.0, (speed_mps - lo) / (hi - lo)))  # clamp [0,1]
        if t < 0.5:
            s = t / 0.5  # 0→1 through red→yellow
            r, g = 255, int(255 * s)
        else:
            s = (t - 0.5) / 0.5  # 0→1 through yellow→green
            r, g = int(255 * (1 - s)), 255
        return (r, g, 0, 255)

    def _apply_lane_vsl(self, speed_mps):
        """Apply the chosen VSL max speed to the target lane via TraCI."""
        if self.lane_default_speed_mps is None:
            return
        traci.lane.setMaxSpeed(self.TARGET_VSL_LANE_ID, speed_mps)

        # VMS POI indicator – gradient color + speed text label
        poi_id = "vsl_indicator_" + self.TARGET_VSL_LANE_ID
        if poi_id not in traci.poi.getIDList():
            shape = traci.lane.getShape(self.TARGET_VSL_LANE_ID)
            x, y = shape[0]
            traci.poi.add(
                poi_id,
                x,
                y,
                (255, 255, 255, 255),
                poiType="vms",
                layer=100,
                width=8,
                height=8,
            )

        color = self._speed_to_color(speed_mps)
        traci.poi.setColor(poi_id, color)
        speed_kmh = round(speed_mps * 3.6)
        traci.poi.setType(poi_id, f"{speed_kmh} km/h")

    def _collect_data_at_cycle_end(self):
        self.processed_flow_upstream_vph = self.get_loops_flow_interval(
            self.upstream_detector_ids_state, self.CYCLE_DURATION_SEC
        )
        self.processed_flow_merging_vph = self.get_loops_flow_interval(
            self.bottleneck_detector_ids_state, self.CYCLE_DURATION_SEC
        )
        self.processed_mainline_flow_downstream_vph = self.get_loops_flow_interval(
            self.outflow_detector_ids_reward, self.CYCLE_DURATION_SEC
        )

        self.processed_occ_upstream_percent = self.get_loops_occupancy_interval(
            self.upstream_detector_ids_state
        )
        self.processed_occ_bottleneck_percent = self.get_loops_occupancy_interval(
            self.bottleneck_detector_ids_state
        )
        self.processed_occ_downstream_percent = self.get_loops_occupancy_interval(
            self.outflow_detector_ids_reward
        )

        self.processed_speed_upstream_mps = self.get_loops_mean_speed_interval(
            self.upstream_detector_ids_state
        )
        self.processed_speed_bottleneck_mps = self.get_loops_mean_speed_interval(
            self.bottleneck_detector_ids_state
        )
        self.processed_mainline_speed_downstream_mps = (
            self.get_loops_mean_speed_interval(self.outflow_detector_ids_reward)
        )

        self.processed_ramp_queue_veh = (
            self.sum_queue * self.sim_step_length / self.CYCLE_DURATION_SEC
            if self.CYCLE_DURATION_SEC > 0
            else 0.0
        )

        self.processed_flow_lane_0_merging_vph = self.get_loops_flow_interval(
            [self.bottleneck_detector_ids_state[0]], self.CYCLE_DURATION_SEC
        )
        self.processed_occ_lane_0_bottleneck_percent = (
            self.get_loops_occupancy_interval([self.bottleneck_detector_ids_state[0]])
        )
        self.processed_speed_lane_0_bottleneck_mps = self.get_loops_mean_speed_interval(
            [self.bottleneck_detector_ids_state[0]]
        )

        self.processed_flow_lane_0_upstream_vph = self.get_loops_flow_interval(
            [self.upstream_detector_ids_state[1]], self.CYCLE_DURATION_SEC
        )
        self.processed_occ_lane_0_upstream_percent = self.get_loops_occupancy_interval(
            [self.upstream_detector_ids_state[1]]
        )
        self.processed_speed_lane_0_upstream_mps = self.get_loops_mean_speed_interval(
            [self.upstream_detector_ids_state[1]]
        )

    def reset(self):
        self.simulation_reset()
        self._reset_cycle_aggregators()
        self.last_green_time_sec = self.green_time_actions_sec[0]
        self.last_vsl_speed_mps = self.vsl_speed_actions_mps[-1]
        self._initialize_last_detailed_info_placeholders()
        self._last_detailed_info.update(super().log_info())

        # Capture the lane's default max speed once the network is loaded.
        try:
            self.lane_default_speed_mps = traci.lane.getMaxSpeed(
                self.TARGET_VSL_LANE_ID
            )
        except traci.TraCIException:
            self.lane_default_speed_mps = self.FREEFLOW_SPEED_MPS
        self.last_vsl_speed_mps = self.lane_default_speed_mps

        if self.ramp_meter_id and self.red_phase_index != -1:
            self.set_phase(self.ramp_meter_id, self.red_phase_index)
            self.set_phase_duration(self.ramp_meter_id, self.CYCLE_DURATION_SEC)

        if self.sim_step_length > 0:
            num_init_steps = int(round(max(1.0, 5.0 / self.sim_step_length)))
        else:
            num_init_steps = 5

        for _ in range(num_init_steps):
            if self.is_simulation_end():
                break
            self.simulation_step()

        self._collect_data_at_cycle_end()

        current_phase_index_init = -1
        current_ryg_state_init = "N/A"
        if self.ramp_meter_id:
            try:
                current_phase_index_init = self.get_phase(self.ramp_meter_id)
                current_ryg_state_init = self.get_ryg_state(self.ramp_meter_id)
            except Exception:
                pass

        self._last_detailed_info.update(
            {
                "mainline_flow_upstream_v/h": self.processed_flow_upstream_vph,
                "mainline_occ_upstream_percent": self.processed_occ_upstream_percent,
                "mainline_speed_upstream_km/h": self.processed_speed_upstream_mps,
                "mainline_flow_mergeArea_v/h": self.processed_flow_merging_vph,
                "mainline_occ_mergeArea_percent": self.processed_occ_bottleneck_percent,
                "mainline_speed_mergeArea_km/h": self.processed_speed_bottleneck_mps,
                "mainline_flow_downstream_v/h": self.processed_mainline_flow_downstream_vph,
                "mainline_speed_downstream_km/h": self.processed_mainline_speed_downstream_mps,
                "mainline_occ_downstream_percent": self.processed_occ_downstream_percent,
                "ramp_queue_veh": self.processed_ramp_queue_veh,
                "current_tl_phase_index": current_phase_index_init,
                "current_tl_ryg_state": current_ryg_state_init,
                "chosen_green_time_sec": self.last_green_time_sec,
                "chosen_vsl_speed_mps": self.last_vsl_speed_mps,
            }
        )
        self._last_detailed_info.update(super().log_info())

        return self._get_current_observation()

    def step(self, action_index):
        if not (0 <= action_index < self.action_space_n):
            action_index = np.clip(action_index, 0, self.action_space_n - 1).item()

        green_idx = int(action_index) % len(self.green_time_actions_sec)
        vsl_idx = int(action_index) // len(self.green_time_actions_sec)

        chosen_green_time_sec = self.green_time_actions_sec[green_idx]
        chosen_vsl_speed_mps = self.vsl_speed_actions_mps[vsl_idx]
        self.last_green_time_sec = chosen_green_time_sec
        self.last_vsl_speed_mps = chosen_vsl_speed_mps

        red_time_sec = self.CYCLE_DURATION_SEC - chosen_green_time_sec
        if red_time_sec < 0:
            red_time_sec = 0.0

        self._reset_cycle_aggregators()

        # Apply the green phase first. The lane VSL is only active during this
        # sub-phase so that ramp vehicles released by the green light get a
        # clear merge corridor; the lane reopens for the red sub-phase below.
        if (
            self.ramp_meter_id
            and self.green_phase_index != -1
            and chosen_green_time_sec > 0
        ):
            self._apply_lane_vsl(chosen_vsl_speed_mps)
            self.set_phase(self.ramp_meter_id, self.green_phase_index)
            self.set_phase_duration(self.ramp_meter_id, chosen_green_time_sec)
            if self.sim_step_length > 0:
                num_steps_green = int(
                    round(chosen_green_time_sec / self.sim_step_length)
                )
            else:
                num_steps_green = 0

            for _ in range(num_steps_green):
                if self.is_simulation_end():
                    break
                self.simulation_step()
                self.sum_queue += self.get_edge_ls_queue_length_vehicles(
                    self.ON_RAMP_EDGE
                )

        # Reopen the controlled lane before running the red sub-phase, so that
        # mainline traffic flows freely while no ramp vehicles are released.
        self._apply_lane_vsl(self.lane_default_speed_mps)

        # Apply the red phase next
        if self.ramp_meter_id and self.red_phase_index != -1 and red_time_sec > 0:
            self.set_phase(self.ramp_meter_id, self.red_phase_index)
            self.set_phase_duration(self.ramp_meter_id, red_time_sec)
            if self.sim_step_length > 0:
                num_steps_red = int(round(red_time_sec / self.sim_step_length))
            else:
                num_steps_red = 0

            for _ in range(num_steps_red):
                if self.is_simulation_end():
                    break
                self.simulation_step()
                self.sum_queue += self.get_edge_ls_queue_length_vehicles(
                    self.ON_RAMP_EDGE
                )

        self._collect_data_at_cycle_end()

        new_observation = self._get_current_observation()
        reward = self._calculate_reward()
        is_done = (
            self.is_simulation_end() or self.get_current_time() >= self.args["steps"]
        )

        current_phase_index = -1
        current_ryg_state = "N/A"
        if self.ramp_meter_id:
            try:
                current_phase_index = self.get_phase(self.ramp_meter_id)
                current_ryg_state = self.get_ryg_state(self.ramp_meter_id)
            except Exception:
                pass

        info_for_this_step = {
            "mainline_flow_upstream_v/h": self.processed_flow_upstream_vph,
            "mainline_occ_upstream_percent": self.processed_occ_upstream_percent,
            "mainline_speed_upstream_km/h": self.processed_speed_upstream_mps,
            "mainline_flow_mergeArea_v/h": self.processed_flow_merging_vph,
            "mainline_occ_mergeArea_percent": self.processed_occ_bottleneck_percent,
            "mainline_speed_mergeArea_km/h": self.processed_speed_bottleneck_mps,
            "mainline_flow_downstream_v/h": self.processed_mainline_flow_downstream_vph,
            "mainline_speed_downstream_km/h": self.processed_mainline_speed_downstream_mps,
            "mainline_occ_downstream_percent": self.processed_occ_downstream_percent,
            "ramp_queue_veh": self.processed_ramp_queue_veh,
            "current_tl_phase_index": current_phase_index,
            "current_tl_ryg_state": current_ryg_state,
            "chosen_green_time_sec": chosen_green_time_sec,
            "chosen_vsl_speed_mps": chosen_vsl_speed_mps,
            "reward_outflow_speed_comp": self._reward_outflow_speed(),
            "reward_throughput_comp": self._reward_throughput(),
            "penalty_ramp_queue_comp": self._penalty_ramp_queue(),
            "penalty_bottleneck_occ_comp": self._penalty_bottleneck_occ(),
            "penalty_spillback_comp": self._penalty_spillback(),
        }

        info_for_this_step.update(super().log_info())
        self._last_detailed_info = info_for_this_step.copy()

        return new_observation, reward, is_done, info_for_this_step

    def _get_current_observation(self):
        norm_flow_upstream = np.clip(
            self.processed_flow_upstream_vph / self.MAX_FLOW_UPSTREAM_VPH, 0, 1
        )
        norm_flow_merging = np.clip(
            self.processed_flow_merging_vph / self.MAX_FLOW_MERGING_VPH, 0, 1
        )
        norm_occ_upstream = np.clip(
            self.processed_occ_upstream_percent / self.MAX_OCCUPANCY_PERCENT, 0, 1
        )
        norm_speed_upstream = np.clip(
            self.processed_speed_upstream_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )
        norm_occ_bottleneck = np.clip(
            self.processed_occ_bottleneck_percent / self.MAX_OCCUPANCY_PERCENT, 0, 1
        )
        norm_speed_bottleneck = np.clip(
            self.processed_speed_bottleneck_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )
        norm_ramp_queue = np.clip(
            self.processed_ramp_queue_veh
            / (self.MAX_RAMP_QUEUE_VEH if self.MAX_RAMP_QUEUE_VEH > 0 else 1.0),
            0,
            1,
        )
        norm_flow_lane_0_bottleneck = np.clip(
            self.processed_flow_lane_0_merging_vph
            / (self.MAX_LANE_FLOW_VPH if self.MAX_LANE_FLOW_VPH > 0 else 1.0),
            0,
            1,
        )
        norm_flow_lane_0_upstream = np.clip(
            self.processed_flow_lane_0_upstream_vph
            / (self.MAX_LANE_FLOW_VPH if self.MAX_LANE_FLOW_VPH > 0 else 1.0),
            0,
            1,
        )
        norm_occ_lane_0_bottleneck = np.clip(
            self.processed_occ_lane_0_bottleneck_percent
            / (self.MAX_OCCUPANCY_PERCENT if self.MAX_OCCUPANCY_PERCENT > 0 else 0.0),
            0,
            1,
        )
        norm_speed_lane_0_bottleneck = np.clip(
            self.processed_speed_lane_0_bottleneck_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )
        norm_occ_lane_0_upstream = np.clip(
            self.processed_occ_lane_0_upstream_percent
            / (self.MAX_OCCUPANCY_PERCENT if self.MAX_OCCUPANCY_PERCENT > 0 else 0.0),
            0,
            1,
        )
        norm_speed_lane_0_upstream = np.clip(
            self.processed_speed_lane_0_upstream_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )
        norm_last_green_time = np.clip(
            self.last_green_time_sec
            / (self.CYCLE_DURATION_SEC if self.CYCLE_DURATION_SEC > 0 else 1.0),
            0,
            1,
        )
        norm_last_vsl_speed = np.clip(
            self.last_vsl_speed_mps
            / (
                self.vsl_speed_actions_mps[-1]
                if self.vsl_speed_actions_mps[-1] > 0
                else 1.0
            ),
            0,
            1,
        )

        state = np.array(
            [
                # main macro features
                norm_flow_upstream,
                norm_flow_merging,
                norm_occ_upstream,
                norm_speed_upstream,
                norm_occ_bottleneck,
                norm_speed_bottleneck,
                norm_ramp_queue,
                # lane-0 features
                norm_flow_lane_0_bottleneck,
                norm_flow_lane_0_upstream,
                norm_occ_lane_0_bottleneck,
                norm_speed_lane_0_bottleneck,
                norm_occ_lane_0_upstream,
                norm_speed_lane_0_upstream,
                # last action features
                norm_last_green_time,
                norm_last_vsl_speed,
            ],
            dtype=np.float32,
        )
        return state

    def _reward_outflow_speed(self):
        return np.clip(
            self.processed_mainline_speed_downstream_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )

    def _reward_upstream_speed(self):
        return np.clip(
            self.processed_speed_upstream_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )

    def _reward_merging_speed(self):
        return np.clip(
            self.processed_speed_bottleneck_mps
            / (self.FREEFLOW_SPEED_MPS if self.FREEFLOW_SPEED_MPS > 0 else 1.0),
            0,
            1,
        )

    def _penalty_bottleneck_occ(self):
        norm_occ = np.clip(
            self.processed_occ_bottleneck_percent
            / (self.MAX_OCCUPANCY_PERCENT if self.MAX_OCCUPANCY_PERCENT > 0 else 1.0),
            0,
            1,
        )
        return -1.0 * norm_occ

    def _penalty_upstream_occ(self):
        norm_occ = np.clip(
            self.processed_occ_upstream_percent
            / (self.MAX_OCCUPANCY_PERCENT if self.MAX_OCCUPANCY_PERCENT > 0 else 1.0),
            0,
            1,
        )
        return -1.0 * norm_occ

    def _reward_throughput(self):
        if self.get_edge_lane_n(self.DOWNSTREAM_EDGE) > 0:
            max_possible_throughput = self.MAX_LANE_FLOW_VPH * self.get_edge_lane_n(
                self.DOWNSTREAM_EDGE
            )
        else:
            max_possible_throughput = self.MAX_LANE_FLOW_VPH
        return np.clip(
            self.processed_mainline_flow_downstream_vph
            / (max_possible_throughput if max_possible_throughput > 0 else 1.0),
            0,
            1,
        )

    def _penalty_ramp_queue(self):
        norm_queue = np.clip(
            self.processed_ramp_queue_veh
            / (self.MAX_RAMP_QUEUE_VEH if self.MAX_RAMP_QUEUE_VEH > 0 else 1.0),
            0,
            1,
        )
        return -1.0 * norm_queue

    def _penalty_spillback(self):
        spillback_threshold_veh = 0.9 * self.MAX_RAMP_QUEUE_VEH
        if self.processed_ramp_queue_veh > spillback_threshold_veh:
            denominator = self.MAX_RAMP_QUEUE_VEH - spillback_threshold_veh
            if denominator < 1e-6:
                denominator = 1e-6
            spill_amount = (
                self.processed_ramp_queue_veh - spillback_threshold_veh
            ) / denominator
            return -1.0 * np.clip(spill_amount, 0, 1)
        return 0.0

    def _calculate_reward(self):

        # reward weights
        w_speed_merge = 1.5
        w_speed_up = 1.0
        w_speed_down = 0.5
        w_occ_bottle = 2.0
        w_occ_upstream = 1.0
        w_queue = 1.0
        w_spillback = 20.0

        # reward components
        r_speed_merge = self._reward_merging_speed()
        r_speed_up = self._reward_upstream_speed()
        r_speed_down = self._reward_outflow_speed()
        p_occ_bottle = self._penalty_bottleneck_occ()
        p_occ_upstream = self._penalty_upstream_occ()
        p_queue = self._penalty_ramp_queue()
        p_spillback = self._penalty_spillback()

        # reward calculation: weighted sum of the components
        reward = (
            (w_speed_merge * r_speed_merge)
            + (w_speed_up * r_speed_up)
            + (w_speed_down * r_speed_down)
            + (w_occ_bottle * p_occ_bottle)
            + (w_occ_upstream * p_occ_upstream)
            + (w_queue * p_queue)
            + (w_spillback * p_spillback)
        )
        return float(reward)

    def obs(self):
        return self._get_current_observation()

    def rew(self):
        return self._calculate_reward()

    def done(self):
        return self.is_simulation_end() or self.get_current_time() >= self.args["steps"]

    def info(self):
        return self._last_detailed_info
