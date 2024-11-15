from isaacgym import gymapi
from isaacgym import gymutil

class EnvironmentLoader:
    def __init__(self, environment_path, sim_params=None):
        self.environment_path = environment_path
        self.gym = gymapi.acquire_gym()
        self.sim = self._create_sim(sim_params)

    def _create_sim(self, sim_params=None):
        if sim_params is None:
            sim_params = gymapi.SimParams()
            sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
            sim_params.up_axis = gymapi.UP_AXIS_Z
        sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        return sim

    def load_environment(self):
        print(f"Loading environment from {self.environment_path}...")
        env = self.gym.create_env(
            self.sim, 
            gymapi.Vec3(-1, -1, 0),  # Lower corner of the environment
            gymapi.Vec3(1, 1, 1),   # Upper corner of the environment
            1                        # Number of environments
        )

        # Load the USD file
        asset_options = gymapi.AssetOptions()
        asset = self.gym.load_asset(
            self.sim, 
            ".",  # Use the current directory
            self.environment_path, 
            asset_options
        )
        self.gym.create_actor(env, asset, gymapi.Transform(), "base", 0, 1)
        return env
