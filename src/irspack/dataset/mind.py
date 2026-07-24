import json
import os
import shutil
import urllib.request
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, List, Optional, Union
from urllib.error import HTTPError
from zipfile import ZipFile

import pandas as pd


class MINDDataManager:
    r"""Manage the Microsoft News Dataset (MIND).

    MIND contains timestamped news impressions as well as article categories,
    titles, abstracts, linked Wikidata entities, and knowledge-graph
    embeddings.  The ``small`` variant is the default because it is suitable
    for local experiments (the two archives are roughly 190 MB in total).

    The official training and development archives are chronologically
    separated.  ``read_interaction()`` only returns clicks from the
    timestamped impression field; it deliberately does not expand ``history``
    because timestamps for those historical clicks are not available.

    Before downloading MIND, read the Microsoft Research License Terms linked
    from https://msnews.github.io/. Setting ``force_download=True`` confirms
    that you have read and accept those terms.

    Args:
        train_zippath:
            Path to the training archive. Defaults to
            ``~/.mind/MIND{size}_train.zip``.
        dev_zippath:
            Path to the development archive. Defaults to
            ``~/.mind/MIND{size}_dev.zip``.
        size:
            Either ``"small"`` or ``"large"``.
        force_download:
            Download missing archives without an interactive confirmation.
            This also confirms acceptance of the dataset's license terms.
        download_source:
            ``"huggingface"`` (default) downloads the original archives from
            the gated ``yjw1029/MIND`` mirror. ``"azure"`` retains the legacy
            official URL, which currently returns HTTP 409 because anonymous
            blob access has been disabled.
        hf_token:
            Hugging Face read token. If omitted, ``HF_TOKEN`` and then
            ``HUGGING_FACE_HUB_TOKEN`` are checked. The token is only needed
            when downloading from Hugging Face, not when archives already
            exist locally.
    """

    AZURE_BASE_URL = "https://mind201910small.blob.core.windows.net/release"
    HUGGINGFACE_BASE_URL = "https://huggingface.co/datasets/yjw1029/MIND/resolve/main"
    DEFAULT_DIR = Path("~/.mind").expanduser()
    _VALID_SPLITS = ("train", "dev")
    _VALID_SIZES = ("small", "large")

    def __init__(
        self,
        train_zippath: Optional[Union[Path, str]] = None,
        dev_zippath: Optional[Union[Path, str]] = None,
        size: str = "small",
        force_download: bool = False,
        download_source: str = "huggingface",
        hf_token: Optional[str] = None,
    ):
        if size not in self._VALID_SIZES:
            raise ValueError("size must be either 'small' or 'large'.")
        if download_source not in ("huggingface", "azure"):
            raise ValueError("download_source must be 'huggingface' or 'azure'.")

        default_prefix = self.DEFAULT_DIR / f"MIND{size}"
        paths = {
            "train": Path(train_zippath or f"{default_prefix}_train.zip"),
            "dev": Path(dev_zippath or f"{default_prefix}_dev.zip"),
        }
        missing = [split for split, path in paths.items() if not path.exists()]
        if missing:
            should_download = force_download
            if not force_download:
                answer = input(
                    "Could not find the MIND archive(s): {}.\n"
                    "Have you read and accepted the Microsoft Research License "
                    "Terms, and may I download them? [y/N]".format(
                        ", ".join(str(paths[split]) for split in missing)
                    )
                )
                should_download = answer.lower() == "y"
            if not should_download:
                raise RuntimeError("could not read the MIND archives")
            for split in missing:
                path = paths[split]
                path.parent.mkdir(parents=True, exist_ok=True)
                filename = f"MIND{size}_{split}.zip"
                base_url = (
                    self.HUGGINGFACE_BASE_URL
                    if download_source == "huggingface"
                    else self.AZURE_BASE_URL
                )
                url = f"{base_url}/{filename}"
                print(f"downloading {url}...")
                self._download(
                    url,
                    path,
                    hf_token=hf_token if download_source == "huggingface" else None,
                )

        self.size = size
        self.paths = paths
        self._zipfiles = {split: ZipFile(path) for split, path in paths.items()}

    @staticmethod
    def _download(url: str, path: Path, hf_token: Optional[str] = None) -> None:
        token = (
            hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        request = urllib.request.Request(url, headers=headers)
        temporary_path: Optional[Path] = None
        try:
            with urllib.request.urlopen(request) as response:
                with NamedTemporaryFile(
                    "wb", delete=False, dir=path.parent, prefix=f".{path.name}."
                ) as output:
                    temporary_path = Path(output.name)
                    shutil.copyfileobj(response, output)
            temporary_path.replace(path)
        except HTTPError as err:
            if err.code in (401, 403):
                raise RuntimeError(
                    "MIND's Hugging Face mirror is gated. Visit "
                    "https://huggingface.co/datasets/yjw1029/MIND, accept its "
                    "access conditions, then set a Hugging Face read token in "
                    "the HF_TOKEN environment variable."
                ) from err
            if err.code == 409:
                raise RuntimeError(
                    "The legacy MIND Azure storage no longer permits anonymous "
                    "access. Use download_source='huggingface' or place the two "
                    "archives manually."
                ) from err
            raise
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def close(self) -> None:
        """Close the training and development archives."""
        for zf in self._zipfiles.values():
            zf.close()

    def __enter__(self) -> "MINDDataManager":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _validate_split(self, split: str) -> None:
        if split not in self._VALID_SPLITS:
            raise ValueError("split must be either 'train' or 'dev'.")

    def _read_as_istream(self, split: str, path: str) -> BytesIO:
        self._validate_split(split)
        zf = self._zipfiles[split]
        if path in zf.namelist():
            member = path
        else:
            matches = [name for name in zf.namelist() if name.endswith(f"/{path}")]
            if len(matches) != 1:
                raise KeyError(
                    f"Could not uniquely locate {path!r} in the {split} archive."
                )
            member = matches[0]
        return BytesIO(zf.read(member))

    def _iter_splits(self, split: Optional[str]) -> Iterable[str]:
        if split is None:
            return self._VALID_SPLITS
        self._validate_split(split)
        return (split,)

    def read_behaviors(self, split: str = "train") -> pd.DataFrame:
        """Read the original impression-level behavior log."""
        with self._read_as_istream(split, "behaviors.tsv") as ifs:
            result = pd.read_csv(
                ifs,
                sep="\t",
                header=None,
                names=[
                    "impression_id",
                    "user_id",
                    "timestamp",
                    "history",
                    "impressions",
                ],
            )
        result["timestamp"] = pd.to_datetime(
            result["timestamp"], format="%m/%d/%Y %I:%M:%S %p"
        )
        return result

    def read_impressions(
        self, split: Optional[str] = None, clicked_only: bool = False
    ) -> pd.DataFrame:
        """Read candidate impressions in long form.

        Args:
            split:
                ``"train"``, ``"dev"``, or ``None`` to concatenate both.
            clicked_only:
                If true, retain only clicked candidates.

        Returns:
            A dataframe with impression, user, timestamp, item, position, and
            click-label columns.
        """
        frames: List[pd.DataFrame] = []
        for split_name in self._iter_splits(split):
            behaviors = self.read_behaviors(split_name)
            candidates = behaviors["impressions"].str.split().explode()
            labels = candidates.str.rsplit("-", n=1, expand=True)
            frame = behaviors.loc[
                candidates.index, ["impression_id", "user_id", "timestamp"]
            ].copy()
            frame["item_id"] = labels[0].to_numpy()
            frame["clicked"] = labels[1].eq("1").to_numpy()
            frame["position"] = frame.groupby("impression_id", sort=False).cumcount()
            frame["split"] = split_name
            if clicked_only:
                frame = frame[frame["clicked"]]
            frames.append(frame.reset_index(drop=True))
        return pd.concat(frames, ignore_index=True)

    def read_interaction(self, split: Optional[str] = None) -> pd.DataFrame:
        """Read positive, timestamped item interactions.

        ``history`` is intentionally excluded because MIND provides neither
        the individual timestamps nor impression context for those clicks.
        """
        return self.read_impressions(split=split, clicked_only=True).drop(
            columns=["clicked", "position"]
        )

    @staticmethod
    def _parse_entities(value: object) -> List[Dict[str, object]]:
        if not isinstance(value, str) or not value:
            return []
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("MIND entity data must be a JSON list.")
        return parsed

    def read_item_info(
        self, split: Optional[str] = None, parse_entities: bool = True
    ) -> pd.DataFrame:
        """Read article metadata, indexed by ``item_id``.

        When both splits are requested, duplicate articles are returned once.
        By default the entity JSON fields are parsed into lists of dictionaries.
        """
        frames: List[pd.DataFrame] = []
        columns = [
            "item_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ]
        for split_name in self._iter_splits(split):
            with self._read_as_istream(split_name, "news.tsv") as ifs:
                frames.append(pd.read_csv(ifs, sep="\t", header=None, names=columns))
        result = pd.concat(frames, ignore_index=True).drop_duplicates(
            "item_id", keep="first"
        )
        if parse_entities:
            for column in ("title_entities", "abstract_entities"):
                result[column] = result[column].map(self._parse_entities)
        return result.set_index("item_id")

    def _read_embedding(self, filename: str, id_name: str, split: str) -> pd.DataFrame:
        with self._read_as_istream(split, filename) as ifs:
            result = pd.read_csv(ifs, sep="\t", header=None)
        result = result.rename(columns={0: id_name}).set_index(id_name)
        result.columns = [f"embedding_{i}" for i in range(result.shape[1])]
        return result.astype("float32")

    def read_entity_embeddings(self, split: str = "train") -> pd.DataFrame:
        """Read 100-dimensional TransE entity embeddings."""
        return self._read_embedding("entity_embedding.vec", "entity_id", split)

    def read_relation_embeddings(self, split: str = "train") -> pd.DataFrame:
        """Read 100-dimensional TransE relation embeddings."""
        return self._read_embedding("relation_embedding.vec", "relation_id", split)
