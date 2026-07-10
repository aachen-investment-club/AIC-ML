import argparse

from strategies.quant.optimize.optimizer import run_optimizer

"""
- This script is a command line wrapper around run_optimizer
- HOW TO RUN THE OPTIMIZER:
- replace data.csv & metadata.csv with actual file names

python strategies/quant/scripts/run_optimizer.py \
    strategies/quant/scripts/data/data.csv \
    strategies/quant/scripts/data/metadata.csv
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run the parameter optimizer."
    )

    parser.add_argument(
        "input_data",
        help="Path to the input data CSV file.",
    )

    parser.add_argument(
        "input_metadata",
        help="Path to the metadata CSV file.",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50).",
    )

    args = parser.parse_args()

    best_config = run_optimizer(
        input_data_path=args.input_data,
        input_metadata_path=args.input_metadata,
        n_trials=args.trials,
    )

    print("\nOptimization finished.")
    print(best_config)


if __name__ == "__main__":
    main()
