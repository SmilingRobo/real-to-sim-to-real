# real_to_sim_to_real/data_collection.py

class DataCollector:
    def __init__(self, env):
        self.env = env

    def collect_demos(self, num_demos=10, max_steps=100):
        print("Collecting demonstrations...")
        demos = []
        for _ in range(num_demos):
            state = self.env.reset()
            trajectory = []
            for _ in range(max_steps):
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((state, action, reward, next_state))
                state = next_state
                if done:
                    break
            demos.append(trajectory)
        return demos
