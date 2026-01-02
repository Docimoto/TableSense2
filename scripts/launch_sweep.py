"""
Launch W&B sweep for hyperparameter tuning.
"""
import wandb
import yaml
import argparse


def main():
    parser = argparse.ArgumentParser(description="Launch W&B sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sweep_config.yaml",
        help="Path to sweep config YAML file"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Max number of runs"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="tablesense2",
        help="W&B project name"
    )
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Initialize sweep controller
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    
    print(f"Sweep ID: {sweep_id}")
    print(f"Start agent with: wandb agent {args.project}/{sweep_id}")
    print(f"Or run up to {args.count} runs with: wandb agent {args.project}/{sweep_id} --count {args.count}")
    
    # Optional: Start agent immediately in this process
    # wandb.agent(sweep_id, function=train_function, count=args.count)


if __name__ == "__main__":
    main()

