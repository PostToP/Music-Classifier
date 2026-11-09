import sys


def run_fetch() -> None:
    from data.database import main as fetch_videos

    fetch_videos()


def run_preprocess() -> None:
    from data.preprocess import preprocess_dataset

    preprocess_dataset()


def run_train() -> None:
    from model.train import create_model
    from model.tuning import fetch_best_params

    create_model(fetch_best_params())


def run_optuna() -> None:
    from model.tuning import main as optuna_main

    optuna_main()


def run_feature() -> None:
    from model.feature_extraction import prepare_datasets_and_pipelines

    prepare_datasets_and_pipelines()


def run_split() -> None:
    from data.split_dataset import split_dataset2

    split_dataset2()


def run_label_errors() -> None:
    from model.label_errors import find_label_error_candidates

    find_label_error_candidates()


def compile_model() -> None:
    from model.compile import create_model_wrapper

    create_model_wrapper()


def main() -> None:
    COMMANDS = {
        "fetch": run_fetch,
        "preprocess": run_preprocess,
        "split": run_split,
        "feature": run_feature,
        "train": run_train,
        "compile": compile_model,
        "optuna": run_optuna,
        "labelerrors": run_label_errors,
    }

    if not sys.argv[1:]:
        print("Usage: python cli.py <operations>")
        print(f"Available operations: {', '.join(COMMANDS.keys())}")
        return

    commands = sys.argv[1:]

    for i in commands:
        if i not in COMMANDS:
            print(f"Unknown command: {i}")
            return

    while commands:
        command = commands.pop(0)
        COMMANDS[command]()


if __name__ == "__main__":
    main()
