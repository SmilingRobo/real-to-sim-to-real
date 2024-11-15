from real_to_sim_to_real.environment import EnvironmentLoader
from real_to_sim_to_real.data_collection import DataCollector
from real_to_sim_to_real.trainer import Trainer
from real_to_sim_to_real.distillation import Distiller
from real_to_sim_to_real.deployment import Deployer
import yaml

# Load the environment
env_loader = EnvironmentLoader(environment_path="environments/my_robot_environment.usd")
env = env_loader.load_environment()

# Collect demonstration data
data_collector = DataCollector(env)
demos = data_collector.collect_demos(num_demos=10)

# Train the policy using RL
trainer = Trainer(env, config={'learning_rate': 0.001, 'max_steps': 100})
policy = trainer.train(num_steps=1000)

# Perform teacher-student distillation (optional)
distiller = Distiller(teacher=policy, student=policy)
distilled_policy = distiller.distill(demos)

# Deploy the policy to the real-world environment
deployer = Deployer(distilled_policy)
deployer.deploy(real_env=env)
