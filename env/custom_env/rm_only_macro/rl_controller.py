import numpy as np

from ..sumo_env import SumoEnv


class RLController(SumoEnv):
    """Variant: ramp metering only (no lane control), macro-only state.

    Designed as an equitable ablation control for
    `rm_lcc_macro_with_setMaxSpeed`:
        - Same 7 aggregate macro features + `last_green_time` in the state.
        - Same reward formula and weights.
        - Same end-of-cycle aggregation (mean speed interval, queue formula).
    Only the actuator differs: this variant has NO VSL/lane control, so the
    action space is 8 (green-time choices) instead of 16, and the state has
    no `last_lane_action` feature (it would be a constant 0).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.CYCLE_DURATION_SEC = 40.0
        self.ty = 3

        self.green_time_actions_sec = np.array(
            [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        )
        self.action_space_n = len(self.green_time_actions_sec)  # 8

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

        # Macro-only state: 8 features (7 aggregate macro + last_green_time).
        # Matches `rm_lcc_macro_with_setMaxSpeed` minus the `last_lane_action`
        # bit (which would be a constant 0 here).
        self.observation_space_n = 8

        self.last_action_value_sec = self.green_time_actions_sec[0]

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

    def reset(self):
        self.simulation_reset()
        self._reset_cycle_aggregators()
        self.last_action_value_sec = self.green_time_actions_sec[0]
        self._initialize_last_detailed_info_placeholders()
        self._last_detailed_info.update(super().log_info())

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
                "chosen_green_time_sec": self.last_action_value_sec,
            }
        )
        self._last_detailed_info.update(super().log_info())

        return self._get_current_observation()

    def step(self, action_index):
        if not (0 <= action_index < self.action_space_n):
            action_index = np.clip(action_index, 0, self.action_space_n - 1).item()

        chosen_green_time_sec = self.green_time_actions_sec[int(action_index)]
        self.last_action_value_sec = chosen_green_time_sec

        red_time_sec = self.CYCLE_DURATION_SEC - chosen_green_time_sec
        if red_time_sec < 0:
            red_time_sec = 0.0

        self._reset_cycle_aggregators()

        if (
            self.ramp_meter_id
            and self.green_phase_index != -1
            and chosen_green_time_sec > 0
        ):
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
        norm_last_action = np.clip(
            self.last_action_value_sec
            / (self.CYCLE_DURATION_SEC if self.CYCLE_DURATION_SEC > 0 else 1.0),
            0,
            1,
        )

        state = np.array(
            [
                # loop detector features (normalized flow, occupancy, speed)
                norm_flow_upstream,
                norm_flow_merging,
                norm_occ_upstream,
                norm_speed_upstream,
                norm_occ_bottleneck,
                norm_speed_bottleneck,
                norm_ramp_queue,
                # last action (normalized green time)
                norm_last_action,
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
        w_speed_merge = 1.5
        w_speed_up = 1.0
        w_speed_down = 0.5

        w_occ_bottle = 2.0
        w_occ_upstream = 1.0
        w_queue = 5.0
        w_spillback = 20.0

        r_speed_merge = self._reward_merging_speed()
        r_speed_up = self._reward_upstream_speed()
        r_speed_down = self._reward_outflow_speed()

        p_occ_bottle = self._penalty_bottleneck_occ()
        p_occ_upstream = self._penalty_upstream_occ()
        p_queue = self._penalty_ramp_queue()
        p_spillback = self._penalty_spillback()

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
