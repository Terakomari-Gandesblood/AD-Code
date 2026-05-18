import os
import glob
import json
import random
from torch.utils.data import Dataset


class CarlaBCDataset(Dataset):
    def __init__(self, data_dir: str,
                 max_frames: int = None,
                 shuffle_files: bool = True,
                 shuffle_records: bool = True,
                 patterns=None):
        super().__init__()
        self.records = []

        data_dir = os.path.abspath(data_dir)

        if patterns is None:
            patterns = ["data_*.json", "training_data_*.json"]

        json_files = []
        for p in patterns:
            json_files.extend(glob.glob(os.path.join(data_dir, p)))

        json_files = sorted(set(json_files))

        if shuffle_files:
            random.shuffle(json_files)

        if len(json_files) == 0:
            raise FileNotFoundError(
                f"[BC Dataset] No json files found in: {data_dir}\n"
                f"Checked patterns: {patterns}\n"
                f"Please verify your dataset directory and filenames."
            )

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                frames = json.load(f)

            if not isinstance(frames, list) or (len(frames) > 0 and not isinstance(frames[0], dict)):
                raise ValueError(
                    f"[BC Dataset] Bad file format: {jf}\n"
                    f"Expected a JSON list of dict records, got: {type(frames)}"
                )

            if max_frames is None:
                self.records.extend(frames)
            else:
                remaining = max_frames - len(self.records)
                if remaining <= 0:
                    break
                if len(frames) > remaining:
                    frames = random.sample(frames, remaining)
                self.records.extend(frames)

        if shuffle_records:
            random.shuffle(self.records)

        print(f"[BC Dataset] Loaded {len(self.records)} frames from {len(json_files)} files.")
        if len(self.records) == 0:
            raise ValueError("[BC Dataset] Loaded 0 frames. Files exist but are empty.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
