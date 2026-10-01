import datetime
import json
from pathlib import Path

import yaml


class RunDir:
    def __init__(self, base, detector, dataset, when=None, path=None):
        if path:
            self.path = Path(path)
        else:
            when = when or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = Path(base) / f"{when}_{detector}_{dataset}"
        self.path.mkdir(parents=True, exist_ok=True)

    def file(self, name):
        return self.path / name

    def save_table(self, name, frame):
        target = self.path / name
        frame.to_csv(target, index=False)
        return target

    def save_yaml(self, name, payload):
        (self.path / name).write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))

    def save_json(self, name, payload):
        (self.path / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def save_text(self, name, text):
        (self.path / name).write_text(text)
