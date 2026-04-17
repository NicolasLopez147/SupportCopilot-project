from pathlib import Path

import yaml
from datasets import load_dataset

from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT, RAW_DATA_DIR


def load_data_config() -> dict:
    config_path = CONFIGS_DIR / "data.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_hf_dataset(dataset_name: str, hf_name: str, revision: str | None = None) -> None:
    output_dir = RAW_DATA_DIR / dataset_name

    if output_dir.exists():
        print(f"[skip] {dataset_name} already exists at {output_dir}")
        return

    if revision:
        print(f"[download] {dataset_name} from Hugging Face: {hf_name} (revision={revision})")
        dataset = load_dataset(hf_name, revision=revision)
    else:
        print(f"[download] {dataset_name} from Hugging Face: {hf_name}")
        dataset = load_dataset(hf_name)

    dataset.save_to_disk(str(output_dir))
    print(f"[saved] {dataset_name} -> {output_dir}")


def resolve_project_path(path_str: str) -> Path:
    return PROJECT_ROOT / Path(path_str)


def check_external_dataset(dataset_name: str, dataset_cfg: dict) -> None:
    print(f"[check] {dataset_name} is an external dataset")

    local_paths = dataset_cfg.get("local_paths", {})
    if not local_paths:
        print(f"[warning] {dataset_name} has no local_paths configured")
        return

    missing_files = []

    for key, relative_path in local_paths.items():
        file_path = resolve_project_path(relative_path)

        if file_path.exists():
            print(f"[found] {key}: {file_path}")
        else:
            print(f"[missing] {key}: {file_path}")
            missing_files.append(key)

    if missing_files:
        print(f"[pending] {dataset_name} is not ready yet. Missing: {missing_files}")
    else:
        print(f"[ready] {dataset_name} external files are complete")


def main() -> None:
    config = load_data_config()
    datasets_cfg = config.get("datasets", {})

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_cfg in datasets_cfg.items():
        if not dataset_cfg.get("enabled", False):
            continue

        source_type = dataset_cfg.get("source_type")

        if source_type == "huggingface":
            hf_name = dataset_cfg.get("hf_name")
            revision = dataset_cfg.get("revision")

            if not hf_name:
                print(f"[warning] {dataset_name} has no hf_name")
                continue

            download_hf_dataset(dataset_name, hf_name, revision)

        elif source_type == "external":
            check_external_dataset(dataset_name, dataset_cfg)

        else:
            print(f"[warning] {dataset_name} has unsupported source_type: {source_type}")


if __name__ == "__main__":
    main()