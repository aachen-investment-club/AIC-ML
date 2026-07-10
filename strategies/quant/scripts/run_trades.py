import argparse

from strategies.quant.trade.trade import trade

"""
- This script is a command line wrapper around trades
- HOW TO RUN THE TRADES:
- replace data.csv & metadata.csv with actual file names

python strategies/quant/scripts/run_trades.py \
    strategies/quant/scripts/data/data.csv \
    strategies/quant/scripts/data/metadata.csv
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run the trading pipeline."
    )

    parser.add_argument(
        "input_data",
        help="Path to the input data CSV file.",
    )

    parser.add_argument(
        "input_metadata",
        help="Path to the metadata CSV file.",
    )

    args = parser.parse_args()

    result = trade(
        input_data_path=args.input_data,
        input_metadata_path=args.input_metadata,
    )

    print("\nTrading pipeline finished.")
    print(result)


if __name__ == "__main__":
    main()
