# Real-to-Sim-to-Real Framework

### By SmilingRobo


The `real-to-sim-to-real` framework is designed to simplify the training, transfer, and deployment of robotic policies from simulated to real environments using NVIDIA Isaac Gym. This framework provides a structured pipeline for setting up custom environments, collecting demonstration data, training reinforcement learning models, applying teacher-student distillation, and deploying models in real-world setups. The framework is hosted and maintained by [SmilingRobo](https://www.smilingrobo.com) and offers a GUI environment for creating realistic simulations.

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

## Installation

1. **Clone the Repository**:

```bash
   git clone https://github.com/your-repo/real-to-sim-to-real.git
   cd real-to-sim-to-real

   pip install -e .

   pip install -r requirements.txt
   pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --no-index
   pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --no-index

```

2. **Install Dependencies**:
   Make sure you have `Isaac Gym` installed on your machine. For Isaac Gym installation, refer to the [official Isaac Gym installation guide](https://developer.nvidia.com/isaac-gym).

Launch isaac-sim to complete the installation

3. **Install orbit [here](https://docs.omniverse.nvidia.com/isaacsim/latest/ext_omni_isaac_orbit.html)**

```bash
git clone git@github.com:NVIDIA-Omniverse/orbit.git
git checkout f2d97bdcddb3005d17d0ebd1546c7064bc7ae8bc

```

```bash
export ISAACSIM_PATH=<path to isaacsim>
```

```bash
# enter the cloned repository
cd orbit
# create a symbolic link
ln -s ${ISAACSIM_PATH} _isaac_sim
```

Create conda environment from orbit

```bash
./orbit.sh --conda isaac-sim
conda activate isaac-sim
orbit -i
orbit -e
```

Make sure orbit was correctly set up:

```bash
python -c "import omni.isaac.orbit; print('Orbit configuration is now complete.')"
```


## Steps to Use

### 1. Create Your Environment

To get started, you first need to create a custom environment for your robot. This can be easily done using the GUI tools available on [platform.smilingrobo.com](https://platform.smilingrobo.com).<br>
Once your environment is configured, you can export it for use in this framework.

### 2. Run the Example Script

After setting up your environment, you can run the example script provided in `examples/example_usage.py` to train, distill, and deploy a policy.

Here’s an example of how to use the framework in a Python script:

### Collect teleoperation data in sim

- booknshelf

```
python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=booknshelvenew --max_path_length=150 --extra_params=booknshelve,booknshelve,booknshelve_debug_mid_randomness --usd_path=/home/marcel/USDAssets/scenes --offset=1
```

- mugandshelf

```
python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=mugandshelfnew2 --max_path_length=150 --extra_params=mugandshelf,mugandshelf_mid_rot_randomness --usd_path=/home/marcel/USDAssets/scenes
```

- cabinet

```
python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=cabinet --max_path_length=150 --extra_params=cabinet,cabinet_mid_randomness --usd_path=/home/marcel/USDAssets/scenes
```

- drawer

```
python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=drawertras --max_path_length=150 --extra_params=wooden_drawer_bigger,drawer_debug_high_randomness --usd_path=/home/marcel/USDAssets/scenes
```

- dishinrack

```
python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=dishinrackv3 --max_path_length=200 --extra_params=dishnrackv2,dishnrack_high_randomness,no_action_rand
```

- kitchentoaster

```
python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=kitchentoaster --max_path_length=150 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand
```

### RL-finetuning

- cabinet

```
python launch_ppo.py --num_envs=4096 --n_steps=90 --max_path_length=90 --usd_path=/home/marcel/USDAssets/scenes --bc_loss --bc_coef=0.1 --filename=isaac-envcabinet --extra_params=cabinet,cabinet_mid_rot_randomness,no_action_rand --datafolder=demos --num_demos=12 --ppo_batch_size=31257 --run_path=locobot-learn/cabinet.usdppo-finetune/rw4wp46f --model_name=model_policy_567 --from_ppo
```

- mugandshelf

```
python launch_ppo.py --num_envs=4096 --n_steps=150 --max_path_length=150 --bc_loss --bc_coef=0.1 --filename=isaac-envmugandshelf --extra_params=mugandshelf,mugandshelf_mid_rot_randomness,no_action_rand --datafolder=demos --num_demos=6 --ppo_batch_size=31257 --model_name=model_policy_279 --run_path=locobot-learn/mugandshelf.usdppo-finetune/ef5nbwyb --from_ppo
```

- booknshelve

```
python launch_ppo.py --num_envs=4096 --n_steps=130 --max_path_length=130 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=booknshelvesreal2sim --datafolder=/data/pulkitag/data/marcel/data --num_demos=14 --ppo_batch_size=31257 --num_envs=2048
```

- kitchen toaster

```
python launch_ppo.py --num_envs=4096 --n_steps=130 --max_path_length=130 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=isaac-envkitchentoaster --datafolder=demos --num_demos=12 --ppo_batch_size=31257 --num_envs=2048 --usd_path=/home/marcel/USDAssets/scenes
```

- dishsinklab

```
python launch_ppo.py --num_envs=4096 --n_steps=110 --max_path_length=110 --extra_params=dishsinklab,dishsinklab_low_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=isaac-envdishsinklablow --datafolder=demos --num_demos=15 --ppo_batch_size=31257 --num_envs=2048 --usd_path=/home/marcel/USDAssets/scenes
```

### Visualize PPO

```
python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=trash --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --visualize_traj --max_demos=10 --sensors=rgb,pointcloud
```

### Distillation from synthetic pointclouds

- booknshelve

```
python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=booknshelvenewsynthetic --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj
```

- mugandshelf

```
python distillation.py --max_path_length=150 --extra_params=mugandshelf,mugandshelf_mid_rot_randomness,no_action_rand --filename=mugandshelfsynthetic --run_path=locobot-learn/mugandshelf.usdppo-finetune/ke2wi6xb --model_name=policy_finetune_step_369 --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj
```

- cabinet

```
python distillation.py --max_path_length=90 --extra_params=cabinet,cabinet_mid_rot_randomness,no_action_rand --filename=cabinetsynthetic --run_path=locobot-learn/cabinet.usdppo-finetune/uqn3jbmg --model_name=policy_finetune_step_393 --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj
```

- mugupright

```
python distillation.py --max_path_length=100 --extra_params=mugupright,mugupright_mid_randomness,no_action_rand --filename=muguprightsynthetic --run_path=locobot-learn/mugupright.usdppo-finetune/5vocshfn --model_name=model_policy_830 --from_ppo --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj --usd_path=/home/marcel/USDAssets/scenes
```

### Distillation from sim pcd

- cabinet

```
python distillation.py --max_path_length=90 --extra_params=cabinet,cabinet_mid_rot_randomness,no_action_rand --filename=cabinetsimpcd --run_path=locobot-learn/cabinet.usdppo-finetune/uqn3jbmg --model_name=policy_finetune_step_393 --datafolder=/data/pulkitag/data/marcel/data/ --max_demos=2500 --eval_freq=1 --use_state --visualize_traj --policy_batch_size=32 --num_envs=9 --run_path_student=locobot-learn/distillation_cabinet/47r0waz0 --model_name_student=policy_distill_step_71
```

- mugupright

```
python distillation.py --max_path_length=100 --extra_params=mugupright,mugupright_mid_randomness,no_action_rand --filename=muguprightsimpcd --run_path=locobot-learn/mugupright.usdppo-finetune/5vocshfn --model_name=model_policy_830 --from_ppo --datafolder=/home/marcel/data/ --max_demos=2500 --eval_freq=1 --use_state --visualize_traj --num_envs=12 --usd_path=/home/marcel/USDAssets/scenes --run_path_student=locobot-learn/distillation_mugupright/tjlds4ds --model_name_student=policy_distill_step_407
```

- mugandshelf

```
python distillation.py --max_path_length=150 --extra_params=mugandshelf,mugandshelf_mid_rot_randomness,no_action_rand --filename=mugandshelfsimpcd --run_path=locobot-learn/mugandshelf.usdppo-finetune/ke2wi6xb --model_name=policy_finetune_step_369 --datafolder=/data/pulkitag/data/marcel/data/ --max_demos=2500 --eval_freq=1 --use_state --num_envs=12 --visualize_traj --model_name_student=policy_distill_step_381 --run_path_student=locobot-learn/distillation_mugandshelf/valc8x68
```

- booknshelve

```
python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=booknshelvenewsimpcd --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --num_envs=9 --max_demos=2500 --eval_freq=1 --use_state --visualize_traj
```

### Collecting distractor data

```
python distillation.py --max_path_length=80 --extra_params=wooden_drawer_bigger,drawer_debug_extra_randomness --filename=drawerdistractorsfixedv2 --run_path=locobot-learn/drawerbiggerhandle.usdppo-finetune/cpyibf8w --model_name=model_policy_16 --from_ppo --datafolder=/data/pulkitag/data/marcel/data/ --eval_freq=1 --use_state --visualize_traj --policy_batch_size=32 --max_demos=5000 --num_envs=9 --policy_train_steps=1 --distractors=distractors_fixed --seed=2
```

### Running Dagger

```
python distillation.py --max_path_length=65 --extra_params=wooden_drawer_bigger,drawer_debug_high_randomness,no_action_rand --filename=drawerfromrealdistractors,drawerbiggerreal --run_path=locobot-learn/drawerbiggerhandle.usdppo-finetune/l3zbpqta --model_name=model_policy_196 --from_ppo --datafolder=/data/scratch-oc40/pulkitag/marcel --eval_freq=1 --use_state --visualize_traj --policy_batch_size=32 --policy_train_steps=10000 --eval_freq=3 --max_demos=500 --num_envs=5 --run_path_student=locobot-learn/distillation_drawer_bigger/7vl0dkb3 --model_name_student=policy_distill_step_17 --dagger --sampling_expert=0 --policy_batch_size=8
```

## Running in the real world

### Install environment

We use [Polymetis](https://facebookresearch.github.io/fairo/polymetis/). You would need to install polymetis on the robot side. We give the instructions on how to install and create the environments on the GPU side.

```
conda create -n franka-env-new-cuda python=3.8.15
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis
```

- Clone and install [airobot](https://github.com/Improbable-AI/airobot)
- Clone and install [improbable_rdt]()
  - git clone --recurse git@github.com:anthonysimeonov/improbable_rdt.git
  - git checkout lightweight-marcel

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --no-index
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --no-index
```

```
pip install -r franka_requirements.txt
```

### Evaluation

```
conda activate franka-env-new-cuda
export RDT_SOURCE_DIR=~/improbable_rdt/src/rdt
export PB_PLANNING_SOURCE_DIR=~/improbable_rdt/pybullet-planning/
```

- dishinrack

```
python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/zmglvp19 --model_name=policy_distill_step_67 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3
```

### Collect teleop data in the real world

```
conda activate franka-env-new-cuda
export RDT_SOURCE_DIR=~/improbable_rdt/src/rdt
export PB_PLANNING_SOURCE_DIR=~/improbable_rdt/pybullet-planning/
```

```
python teleop_franka.py --demo_folder=mugandshelfreal --offset=0 --hz=2 --max_path_length=100 --extra_params=booknshelve,booknshelve_debug_low_randomness,booknshelve2
```

```
python teleop_franka.py --demo_folder=cupntrashreal --offset=0 --hz=2 --max_path_length=100 --extra_params=cupntrash,kitchentoaster_randomness,no_action_rand --num_demos=15 --cam_index=1
```

```
python teleop_franka.py --demo_folder=cupntrashreal --offset=0 --hz=2 --max_path_length=100 --extra_params=cupntrash,kitchentoaster_randomness,no_action_rand --num_demos=15 --cam_index=1
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
