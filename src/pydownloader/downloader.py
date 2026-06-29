"""
downloader.py — 可复用的多源文件下载模块

支持的下载源：
  - http/https  : 普通 HTTP 请求（requests + tqdm）
  - gdrive      : Google Drive（gdown）
  - youtube     : YouTube / 各大视频网站（yt-dlp）
  - git         : Git 仓库克隆（gitpython / subprocess）

依赖安装：
  pip install requests tqdm gdown yt-dlp gitpython pyyaml

用法示例：
  # 单文件下载
  from downloader import Downloader
  dl = Downloader(max_workers=4, rate_limit_kb=500)
  dl.download("https://example.com/file.zip", "/tmp/file.zip")

  # 批量下载（配置文件）
  dl.download_from_config("tasks.yaml")
"""

import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import yaml

# ── 日志配置 ────────────────────────────────────────────────────────────────

def _setup_logger(name: str = "downloader") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


logger = _setup_logger()


# ── 限速工具 ─────────────────────────────────────────────────────────────────

class _ThrottledWriter:
    """将写操作限速到 rate_limit_kb KB/s（0 = 不限速）。"""

    def __init__(self, file_obj, rate_limit_kb: float = 0):
        self._f = file_obj
        self._rate = rate_limit_kb * 1024  # bytes/s
        self._last = time.monotonic()
        self._written = 0

    def write(self, data: bytes) -> int:
        if self._rate > 0:
            self._written += len(data)
            elapsed = time.monotonic() - self._last
            expected = self._written / self._rate
            if expected > elapsed:
                time.sleep(expected - elapsed)
        return self._f.write(data)


# ── 源类型检测 ────────────────────────────────────────────────────────────────

def _detect_source(url: str) -> str:
    """
    根据 URL 自动判断下载源类型。
    返回: 'gdrive' | 'youtube' | 'git' | 'http'
    """
    u = url.lower()
    if "drive.google.com" in u or "docs.google.com" in u:
        return "gdrive"
    youtube_hosts = {"youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com"}
    parsed = urlparse(url)
    if parsed.hostname in youtube_hosts:
        return "youtube"
    # yt-dlp 支持的其他视频平台可在此扩展
    if u.endswith(".git") or u.startswith("git@") or "/git/" in u:
        return "git"
    # 也支持显式 git+https://
    if u.startswith("git+"):
        return "git"
    return "http"


# ── 各源下载实现 ──────────────────────────────────────────────────────────────

def _download_http(
    url: str,
    dest_: Path,
    rate_limit_kb: float = 0,
) -> None:
    """HTTP/HTTPS 下载，支持进度条和限速。"""
    import requests
    from tqdm import tqdm

    if dest_.is_dir() :
        dest = dest_ / os.path.basename(url).split('?')[0]
    else :
        dest = dest_
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with dest.open("wb") as fh, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
            leave=False,
        ) as bar:
            writer = _ThrottledWriter(fh, rate_limit_kb)
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    writer.write(chunk)
                    bar.update(len(chunk))


def _download_gdrive(url: str, dest: Path, rate_limit_kb: float = 0) -> None:
    """Google Drive 下载（gdown）。rate_limit 在 gdown 层面暂不支持，忽略。"""
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    # dest 若为目录，gdown 自动命名；若含扩展名视为文件路径
    output = str(dest)
    gdown.download(url, output=output, quiet=False)


def _download_youtube(url: str, dest: Path, rate_limit_kb: float = 0) -> None:
    """yt-dlp 下载视频，dest 可为目录或带扩展名的文件路径。"""
    import yt_dlp

    dest.parent.mkdir(parents=True, exist_ok=True)

    # 判断 dest 是目录还是文件路径
    if dest.suffix:
        outtmpl = str(dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        outtmpl = str(dest / "%(title)s.%(ext)s")

    ydl_opts: dict = {
        "outtmpl": outtmpl,
        "progress_hooks": [],
        "quiet": False,
        "no_warnings": False,
    }
    if rate_limit_kb > 0:
        ydl_opts["ratelimit"] = int(rate_limit_kb * 1024)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def _download_git(url: str, dest: Path, rate_limit_kb: float = 0) -> None:
    """Git 仓库克隆（优先用 gitpython，回退到 subprocess git clone）。"""
    # 去掉 git+ 前缀
    real_url = url[4:] if url.lower().startswith("git+") else url
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import git as gitmodule
        gitmodule.Repo.clone_from(real_url, str(dest), progress=_GitProgress())
    except ImportError:
        logger.warning("gitpython 未安装，回退到 subprocess git clone")
        subprocess.run(
            ["git", "clone", "--progress", real_url, str(dest)],
            check=True,
        )


class _GitProgress:
    """gitpython 进度回调，打印到终端。"""

    def __call__(self, op_code, cur_count, max_count=None, message=""):
        if max_count:
            pct = cur_count / max_count * 100
            print(f"\r  克隆进度: {pct:.1f}%  {message}", end="", flush=True)
    
    # gitpython RemoteProgress 接口兼容
    def update(self, op_code, cur_count, max_count=None, message=""):
        self(op_code, cur_count, max_count, message)


# ── 主模块类 ──────────────────────────────────────────────────────────────────

class Downloader:
    """
    多源文件下载器。

    Parameters
    ----------
    max_workers : int
        并发下载线程数（默认 1，即串行）。
    rate_limit_kb : float
        全局限速，单位 KB/s（0 = 不限速）。HTTP 下载生效；
        YouTube 通过 yt-dlp 限速；GDrive / Git 暂不支持限速。
    log_file : str | None
        若指定，日志同时写入该文件。
    """

    def __init__(
        self,
        max_workers: int = 1,
        rate_limit_kb: float = 0,
        log_file: Optional[str] = None,
    ):
        self.max_workers = max(1, max_workers)
        self.rate_limit_kb = rate_limit_kb
        self._logger = _setup_logger()

        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._logger.addHandler(fh)

    # ── 核心单文件下载 ──────────────────────────────────────────────────────

    def download(
        self,
        url: str,
        dest: Union[str, Path],
        source_type: Optional[str] = None,
    ) -> bool:
        """
        下载单个文件。

        Parameters
        ----------
        url         : 文件链接或 Git 仓库地址。
        dest        : 保存路径（目录或文件路径）。
        source_type : 强制指定源类型 'http'|'gdrive'|'youtube'|'git'，
                      默认自动检测。

        Returns
        -------
        bool : True = 成功，False = 失败。
        """
        dest = Path(dest)
        stype = source_type or _detect_source(url)
        t0 = time.monotonic()

        self._logger.info(f"开始下载  source={stype}  url={url}  dest={dest}")

        try:
            _HANDLERS = {
                "http": _download_http,
                "gdrive": _download_gdrive,
                "youtube": _download_youtube,
                "git": _download_git,
            }
            handler = _HANDLERS.get(stype)
            if handler is None:
                raise ValueError(f"不支持的 source_type: {stype!r}")

            handler(url, dest, self.rate_limit_kb)

            elapsed = time.monotonic() - t0
            self._logger.info(
                f"✓ 下载成功  source={stype}  dest={dest}  耗时={elapsed:.1f}s"
            )
            return True

        except Exception as exc:
            elapsed = time.monotonic() - t0
            self._logger.error(
                f"✗ 下载失败  source={stype}  url={url}  "
                f"耗时={elapsed:.1f}s  错误={exc}"
            )
            return False

    # ── 批量下载 ────────────────────────────────────────────────────────────

    def download_batch(self, tasks: list[dict]) -> dict[str, bool]:
        """
        批量下载。

        Parameters
        ----------
        tasks : list of dict，每项包含：
            - url         : str  （必填）
            - dest        : str  （必填）
            - source_type : str  （可选，自动检测）

        Returns
        -------
        dict : {url: 成功/失败(bool)}
        """
        results: dict[str, bool] = {}

        if self.max_workers == 1:
            for task in tasks:
                url = task["url"]
                results[url] = self.download(
                    url,
                    task["dest"],
                    task.get("source_type"),
                )
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_to_url = {
                    pool.submit(
                        self.download,
                        task["url"],
                        task["dest"],
                        task.get("source_type"),
                    ): task["url"]
                    for task in tasks
                }
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        results[url] = future.result()
                    except Exception as exc:
                        self._logger.error(f"并发任务异常  url={url}  错误={exc}")
                        results[url] = False

        total = len(results)
        ok = sum(results.values())
        self._logger.info(f"批量下载完成  成功={ok}/{total}")
        return results

    # ── 配置文件批量下载 ────────────────────────────────────────────────────

    def download_from_config(self, config_path: Union[str, Path]) -> dict[str, bool]:
        """
        从 JSON / YAML 配置文件批量下载。

        配置文件格式（yaml 示例）：
        ──────────────────────────────
        tasks:
          - url: https://example.com/file.zip
            dest: /data/downloads/file.zip

          - url: https://drive.google.com/file/d/xxx/view
            dest: /data/gdrive/
            source_type: gdrive        # 可选，默认自动检测

          - url: https://www.youtube.com/watch?v=xxx
            dest: /data/videos/

          - url: https://github.com/user/repo.git
            dest: /data/repos/repo

        settings:                      # 可选，覆盖构造函数参数
          max_workers: 4
          rate_limit_kb: 512
        ──────────────────────────────

        Parameters
        ----------
        config_path : JSON 或 YAML 配置文件路径。

        Returns
        -------
        dict : {url: 成功/失败(bool)}
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        suffix = config_path.suffix.lower()
        with config_path.open("r", encoding="utf-8") as f:
            if suffix == ".json":
                cfg = json.load(f)
            elif suffix in (".yaml", ".yml"):
                cfg = yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {suffix}（仅支持 .json / .yaml）")

        # 读取可选全局设置
        settings = cfg.get("settings", {})
        if "max_workers" in settings:
            self.max_workers = max(1, int(settings["max_workers"]))
        if "rate_limit_kb" in settings:
            self.rate_limit_kb = float(settings["rate_limit_kb"])

        tasks = cfg.get("tasks", [])
        if not tasks:
            self._logger.warning("配置文件中没有找到任何下载任务（tasks 为空）")
            return {}

        self._logger.info(
            f"加载配置文件  path={config_path}  任务数={len(tasks)}  "
            f"并发={self.max_workers}  限速={self.rate_limit_kb or '不限'}KB/s"
        )
        return self.download_batch(tasks)


# ── CLI 入口（可选）──────────────────────────────────────────────────────────

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="downloader.py — multi-source file downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 单文件下载
    p_single = sub.add_parser("get", help="Download a single file")
    p_single.add_argument("url", help="File URL or Git repository address")
    p_single.add_argument("dest", help="Save to path")
    p_single.add_argument("--type", dest="source_type", help="Force to specify source type")
    p_single.add_argument("--rate", type=float, default=0, help="Rate limit KB/s (0=unlimited)")

    # 配置文件批量下载
    p_batch = sub.add_parser("batch", help="Download in batch via configuration file")
    p_batch.add_argument("config", help="Path to config file (.json or .yaml)")
    p_batch.add_argument("--workers", type=int, default=1, help="Number of concurrent threads")
    p_batch.add_argument("--rate", type=float, default=0, help="Rate limit KB/s (0=unlimited)")
    p_batch.add_argument("--log", default=None, help="Path to log file")

    args = parser.parse_args()

    if args.cmd == "get":
        dl = Downloader(rate_limit_kb=args.rate)
        ok = dl.download(args.url, args.dest, args.source_type)
        raise SystemExit(0 if ok else 1)

    elif args.cmd == "batch":
        dl = Downloader(
            max_workers=args.workers,
            rate_limit_kb=args.rate,
            log_file=args.log,
        )
        results = dl.download_from_config(args.config)
        failed = [u for u, ok in results.items() if not ok]
        if failed:
            logger.warning(f"Failed to download:\n" + "\n".join(f"  {u}" for u in failed))
        raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    _cli()

