# Real-to-Sim-to-Real Framework
#### By SmilingRobo

The `real-to-sim-to-real` framework is designed to simplify the training, transfer, and deployment of robotic policies from simulated to real environments using NVIDIA Isaac Gym. This framework provides a structured pipeline for setting up custom environments, collecting demonstration data, training reinforcement learning models, applying teacher-student distillation, and deploying models in real-world setups. The framework is hosted and maintained by [SmilingRobo](https://platform.smilingrobo.com) and offers a GUI environment for creating realistic simulations.

This framework is built upon the RialTo system, as proposed in the paper:
> **Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation**  
> [Available on arXiv](https://arxiv.org/abs/2403.03949)  
> Authors: Marcel Torne, Anthony Simeonov, Zechu Li, April Chan, Tao Chen, Abhishek Gupta, Pulkit Agrawal, 2024  

For further information and updates, please visit the project page.

## Key Features
- **Environment Setup**: Simplified loading and configuration of Isaac Gym environments.
- **Data Collection**: Easily collect and store demonstration data in the simulation.
- **Reinforcement Learning**: Train policies using GPU-accelerated reinforcement learning methods.
- **Teacher-Student Distillation**: Transfer skills from complex teacher models to simpler student models.
- **Real-World Deployment**: Deploy trained policies to real robotic setups.

## Requirements
- **NVIDIA Isaac Gym**: GPU-based physics simulation for robotic environments.
- **PyTorch**: Deep learning framework for building and training models.
- **PyYAML**: Used for handling configuration files.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/real-to-sim-to-real.git
   cd real-to-sim-to-real
   ```

2. **Install Dependencies**:
   Make sure you have `Isaac Gym` installed on your machine. For Isaac Gym installation, refer to the [official Isaac Gym installation guide](https://developer.nvidia.com/isaac-gym).

3. **Install the Package**:
   ```bash
   pip install -e .
   ```

## Configuration
This framework uses a `config.yaml` file for easy configuration of training parameters, distillation, and deployment settings. You can find a sample configuration file in the `configs/` directory. Adjust parameters as needed for your environment and task requirements.

## Steps to Use the Framework

### 1. Create Your Environment
To get started, you first need to create a custom environment for your robot. This can be easily done using the GUI tools available on [platform.smilingrobo.com](https://platform.smilingrobo.com). Once your environment is configured, you can export it for use in this framework.

### 2. Run the Example Script
After setting up your environment, you can run the example script provided in `examples/example_usage.py` to train, distill, and deploy a policy.

Here’s an example of how to use the framework in a Python script:

```python
from real_to_sim_to_real import EnvironmentLoader, DataCollector, Trainer, Distiller, Deployer

# Initialize and load the environment
env_loader = EnvironmentLoader(config_path="configs/config.yaml")
env = env_loader.load_environment()

# Collect demonstration data
data_collector = DataCollector(env)
demos = data_collector.collect_demos(num_demos=10)

# Train policy in simulation
trainer = Trainer(env, config={"learning_rate": 0.001, "num_steps": 1000})
policy = trainer.train()

# Distill knowledge from teacher to student policy
distiller = Distiller(teacher=policy, student=policy)
distilled_policy = distiller.distill(demos)

# Deploy policy to the real-world environment
deployer = Deployer(policy=distilled_policy)
deployer.deploy(real_env=env)  # Replace with real-world environment instance
```

## Instructions to Install and Use the GUI Environment
For creating and managing environments with an intuitive GUI, please visit [platform.smilingrobo.com](https://platform.smilingrobo.com), where you can upload, configure, and export environments for use in this framework.

## Citation
If you use this codebase in your research, please cite:
```bibtex
@article{torne2024reconciling,
  title={Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation},
  author={Torne, Marcel and Simeonov, Anthony and Li, Zechu and Chan, April and Chen, Tao and Gupta, Abhishek and Agrawal, Pulkit},
  journal={arXiv preprint arXiv:2403.03949},
  year={2024}
}
```
