import io
import os
import sys
import torch

import yt.wrapper as yt

from abc import ABC, abstractmethod


class Storage(ABC):
    @abstractmethod
    def precache(self):
        ...

    @abstractmethod
    def create_directory(self, path: str):
        ...

    @abstractmethod
    def write_file(self, path: str, data: bytes, metadata: dict[str, str] = {}):
        ...

    def save(self, path: str, obj: object, metadata: dict[str, str] = {}):
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        self.write_file(path, buffer.getvalue(), metadata=metadata)

    @abstractmethod
    def read_file(self, path: str) -> bytes:
        ...

    @abstractmethod
    def read_metadata(self, path: str) -> dict[str, str]:
        ...

    def load(self, path: str, map_location: str | None = None) -> object:
        return torch.load(io.BytesIO(self.read_file(path)), map_location=map_location)

    @abstractmethod
    def list_dir(self, path: str) -> list[str]:
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        ...


class LocalStorage(Storage):
    def __init__(self, base_path: str):
        self._base_path = base_path

    def precache(self):
        pass

    def create_directory(self, path: str):
        os.makedirs(self._get_path(path), exist_ok=True)

    def write_file(self, path: str, data: bytes, metadata: dict[str, str] = {}):
        with open(self._get_path(path), "wb") as f:
            f.write(data)
        with open(self._get_path(path + ".metadata"), "w") as f:
            for k, v in metadata.items():
                f.write(f"{k}: {v}\n")

    def read_file(self, path: str) -> bytes:
        with open(self._get_path(path), "rb") as f:
            return f.read()
        
    def read_metadata(self, path: str) -> dict[str, str]:
        if not self.exists(path + ".metadata"):
            return {}
        with open(self._get_path(path + ".metadata"), "r") as f:
            return dict(line.strip().split(": ", 1) for line in f)

    def list_dir(self, path: str) -> list[str]:
        return os.listdir(self._get_path(path))
    
    def exists(self, path: str) -> bool:
        return os.path.exists(self._get_path(path))

    def _get_path(self, path: str) -> str:
        return self._base_path + "/" + path


class TractoStorage(Storage):
    def __init__(self, yt_client: yt.YtClient, base_path: str):
        # there is a side effect -> directory creation
        yt_client = self._fix_client(yt_client, base_path)
        self._yt_client = yt_client
        self._base_path = base_path

    def precache(self):
        pass

    @staticmethod
    def _fix_client(yt_client: yt.YtClient, base_path: str):
        tmp_dir = "//tmp/nanotron_checkpoints_tmp"
        yt_client_config = yt.config.get_config(yt_client)
        yt_client_config["remote_temp_files_directory"] = tmp_dir
        yt_client_config["remote_temp_tables_directory"] = tmp_dir
        yt_client = yt.YtClient(config=yt_client_config)
        yt_client.create(
            "map_node",
            tmp_dir,
            recursive=True,
            ignore_existing=True,
            attributes={
                "primary_medium": "nvme",
                "media": {
                    "nvme": {
                        "replication_factor": 3,
                        "data_parts_only": False,
                    },
                },
            },
        )
        return yt_client

    def create_directory(self, path: str):
        self._yt_client.create(
            "map_node",
            self._get_path(path),
            recursive=True,
            ignore_existing=True,
        )

    def write_file(self, path: str, data: bytes, metadata: dict[str, str] = {}):
        self._yt_client.write_file(self._get_path(path), data)
        self._yt_client.set(self._get_path(path) + "/@metadata", metadata)

    def read_file(self, path: str) -> bytes:
        return self._yt_client.read_file(self._get_path(path)).read()
    
    def read_metadata(self, path: str) -> dict[str, str]:
        return self._yt_client.get(self._get_path(path) + "/@metadata")

    def list_dir(self, path):
        return self._yt_client.list(self._get_path(path))
    
    def exists(self, path: str) -> bool:
        return self._yt_client.exists(self._get_path(path))

    def _get_path(self, path: str) -> str:
        if not path:
            return self._base_path
        return self._base_path + "/" + path


class CachingTractoStorage(Storage):
    def __init__(self, yt_path: str, tmpfs_path: str | None, yt_client: yt.YtClient):
        self._yt_path = yt_path
        self._yt_client = yt_client
        self._tmpfs_path = tmpfs_path

        if tmpfs_path is not None:
            self._local_storage = LocalStorage(tmpfs_path)
        else:
            self._local_storage = None
        self._yt_storage = TractoStorage(yt_client, yt_path)

    def precache(self):
        if not self._local_storage:
            return
        def do_download(path):
            yt_path = self._yt_path
            if len(path) > 0:
                yt_path += "/" + path
            tp = self._yt_client.get(yt_path + "/@type")
            if tp == "map_node":
                self._local_storage.create_directory(path)
                for child in self._yt_client.list(yt_path):
                    new_path = child
                    if len(path) > 0:
                        new_path = path + "/" + child
                    do_download(new_path)
            else:
                print("Reading file", yt_path, file=sys.stderr)
                content = self._yt_client.read_file(yt_path).read()
                print("Writing file", yt_path, file=sys.stderr)
                metadata = {}
                if self._yt_client.exists(yt_path + "/@metadata"):
                    metadata = self._yt_client.get(yt_path + "/@metadata")
                self._local_storage.write_file(path, content, metadata)
                print("Done", yt_path, file=sys.stderr)
        do_download("")
    
    def remove(self):
        # TODO(gritukan): We always load checkpoint once per process, so we can leave it as is for now
        pass

    def create_directory(self, path: str):
        raise "Not implemented"

    def write_file(self, path: str, data: bytes, metadata: dict[str, str] = {}):
        raise "Not implemented"

    def read_file(self, path: str) -> bytes:
        if self._local_storage:
            return self._local_storage.read_file(path)
        else:
            return self._yt_storage.read_file(path)
        
    def read_metadata(self, path: str) -> dict[str, str]:
        if self._local_storage:
            return self._local_storage.read_metadata(path)
        else:
            return self._yt_storage.read_metadata(path)

    def list_dir(self, path: str) -> list[str]:
        if self._local_storage:
            return self._local_storage.list_dir(path)
        else:
            return self._yt_storage.list_dir(path)
    
    def exists(self, path: str) -> bool:
        if self._local_storage:
            return self._local_storage.exists(path)
        else:
            return self._yt_storage.exists(path)
