import pandas as pd


def print_results(results_df: pd.DataFrame):
    # log per fold
    for session_id, value in results_df.groupby("session_id")["test_acc"].mean().items():
        print(f"session_{session_id}-test_acc: {value}")
    print(f"test_acc: {results_df['test_acc'].mean()}")
    # log dataframes
    print(results_df.head())
