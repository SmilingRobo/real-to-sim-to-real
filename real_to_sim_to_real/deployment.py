# real_to_sim_to_real/deployment.py

class Deployer:
    def __init__(self, policy):
        self.policy = policy

    def deploy(self, real_env):
        print("Deploying policy to the real environment...")
        state = real_env.reset()
        while True:
            action = self.policy(torch.tensor(state, dtype=torch.float32)).argmax().item()
            state, _, done, _ = real_env.step(action)
            if done:
                break
        print("Deployment complete.")
