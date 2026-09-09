from .custom_env import SUMO_PARAMS, Baselines, RLController


class DqnEnv:
    """Wrapper for linking RL agents with SUMO environment."""

    def min_max_scale(self, x, feature):
        """Standardizes input features using min-max scaling."""
        return (x - self.min_max[feature][0]) / (
            self.min_max[feature][1] - self.min_max[feature][0]
        )

    def __init__(self, m, p=None, gui_override=None):
        """Initializes environment mode and controller.

        Args:
            m: Mode (train, observe, or play)
            p: Player strategy name (for play mode)
            gui_override: Optional GUI setting to override SUMO_PARAMS["gui"]
        """
        self.mode = {"train": False, "observe": False, "play": False, m: True}
        self.player = p if self.mode["play"] else None

        # Determine GUI setting, prioritizing override
        gui_setting = gui_override if gui_override is not None else SUMO_PARAMS["gui"]

        # """CHANGE ENV CONSTRUCT HERE""" ##############################################################################
        if self.mode["train"]:
            self.sumo_env = RLController(gui=False, log=False, rnd=(False, False))
        elif self.mode["observe"]:
            self.sumo_env = RLController(
                gui=gui_setting, log=True, rnd=SUMO_PARAMS["rnd"]
            )
        elif self.mode["play"]:
            if p == "Test":
                self.sumo_env = RLController(
                    gui=gui_setting, log=SUMO_PARAMS["log"], rnd=SUMO_PARAMS["rnd"]
                )
            else:
                self.sumo_env = getattr(Baselines, p)(
                    gui=gui_setting, log=SUMO_PARAMS["log"], rnd=SUMO_PARAMS["rnd"]
                )
        ################################################################################################################

        # """CHANGE FEATURE SCALING HERE""" ############################################################################
        self.min_max = {}
        ################################################################################################################

        # """CHANGE ACTION AND OBSERVATION SPACE SIZES HERE""" #########################################################
        self.action_space_n = self.sumo_env.action_space_n
        self.observation_space_n = self.sumo_env.observation_space_n
        ################################################################################################################

    def obs(self):
        """Retrieves current state observation."""
        # """CHANGE OBSERVATION HERE""" ################################################################################
        obs = self.sumo_env.obs()
        ################################################################################################################
        return obs

    def rew(self):
        """Calculates reward for current state."""
        # """CHANGE REWARD HERE""" #####################################################################################
        rew = self.sumo_env.rew()
        ################################################################################################################
        return rew

    def done(self):
        """Checks if episode has terminated."""
        # """CHANGE DONE HERE""" #######################################################################################
        done = self.sumo_env.done()
        ################################################################################################################
        return done

    def info(self):
        """Returns diagnostic information."""
        # """CHANGE INFO HERE""" #######################################################################################
        info = self.sumo_env.info()
        ################################################################################################################
        return info

    def reset(self):
        """Resets environment to initial state."""
        # """CHANGE RESET HERE""" ######################################################################################
        self.sumo_env.reset()
        ################################################################################################################

    def step(self, action):
        """Executes action in the environment."""
        # """CHANGE STEP HERE""" #######################################################################################
        self.sumo_env.step(action)
        ################################################################################################################

    # In dqn_env.py

    def get_scenario_info(self):
        """Passes the request to the underlying sumo_env."""
        if hasattr(self.sumo_env, "get_scenario_info"):
            return self.sumo_env.get_scenario_info()
        return {}

    def close(self):
        """Passes the close command to the underlying sumo_env."""
        if hasattr(self.sumo_env, "close"):
            self.sumo_env.close()

    def reset_render(self):
        """Resets visualization state."""
        # """CHANGE RESET RENDER HERE""" ###############################################################################
        pass
        ################################################################################################################

    def step_render(self):
        """Updates visualization for current step."""
        # """CHANGE STEP RENDER HERE""" ################################################################################
        pass
        ################################################################################################################
