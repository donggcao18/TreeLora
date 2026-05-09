from datasets import load_dataset


CODETASK_HF_REPO = "dongg18/CODETASK_with_instruction_pool"
CODETASK_NAMES = {"CONCODE", "CodeTrans", "CodeSearchNet", "BFP"}


def create_codetask_dataset(dataset_name, seed, num_train=-1, num_eval=-1, num_test=-1):
    if dataset_name not in CODETASK_NAMES:
        raise ValueError(
            f"Unsupported CodeTask dataset '{dataset_name}'. "
            f"Expected one of: {', '.join(sorted(CODETASK_NAMES))}"
        )

    data_dict = {}
    split_sizes = {
        "train": num_train,
        "validation": num_eval,
        "test": num_test,
    }

    for split, size in split_sizes.items():
        dataset = load_dataset(
            CODETASK_HF_REPO,
            data_files={split: f"{dataset_name}/{split}-*.parquet"},
            split=split,
        )
        dataset = dataset.remove_columns([
            column for column in dataset.column_names
            if column not in ("input", "output")
        ])
        dataset = dataset.rename_column("input", "prompt")
        dataset = dataset.rename_column("output", "answer")

        size = int(size)
        if size != -1:
            dataset = dataset.shuffle(seed=seed).select(range(size))

        data_dict[split] = dataset

    return data_dict["train"], data_dict["validation"], data_dict["test"]
