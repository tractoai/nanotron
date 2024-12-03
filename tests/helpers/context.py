import shutil
import uuid
from functools import lru_cache
from pathlib import Path
import yt.wrapper as yt

from nanotron.serialize import (
    Storage,
    LocalStorage,
    TractoStorage,
)


class TestContext:
    def __init__(self):
        self._random_string = str(uuid.uuid1())
        self._root_dir = Path(__file__).parent.parent / ".test_cache"
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._tracto_dir = None

    @lru_cache(maxsize=1)
    def get_auto_remove_tmp_dir(self):
        path = self._root_dir / self._random_string
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_auto_remove_tracto_dir(self):
        if not self._tracto_dir:
            self._tracto_dir = f"//tmp/{self._random_string}"
            yt.create("map_node", self._tracto_dir, recursive=True, ignore_existing=True)
        return self._tracto_dir
    
    def get_storage(self, use_tracto=False):
        if use_tracto:
            ytc = yt.YtClient(config=yt.default_config.get_config_from_env())
            return TractoStorage(ytc, self.get_auto_remove_tracto_dir())
        return LocalStorage(str(self.get_auto_remove_tmp_dir()))

    def __del__(self):
        path = self.get_auto_remove_tmp_dir()
        shutil.rmtree(path)
        if self._tracto_dir:
            yt.remove(self._tracto_dir, recursive=True)
