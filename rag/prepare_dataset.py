import asyncio
import json
import os
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import urllib3.exceptions
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult
from lxml import etree
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

import wandb
from utils import mdify

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rag-workshop")


async def get_urls(sitemap_url: str) -> list[str]:
    """
    Fetches and parses URLs from a sitemap.

    Args:
        sitemap_url (str): The URL of the sitemap to fetch.

    Returns:
        list: A list of unique URLs extracted from the sitemap.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(sitemap_url)
        response.raise_for_status()
        root = etree.fromstring(response.text.encode("utf-8"))
        namespaces = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = root.xpath("//ns:loc/text()", namespaces=namespaces)
        return list(set(urls))


def filter_wandb_docs_urls(urls: list[str]) -> list[str]:
    """
    Filters a list of URLs to include only those related to Weights & Biases documentation.
    sections we think are necessary.

    Args:
        urls (list): A list of URLs to filter.

    Returns:
        list: A sorted list of unique URLs that match the specified documentation paths.
    """
    all_urls = []
    for url in urls:
        url = urlparse(url)
        for path_base in ["guides", "ref", "quickstart", "tutorials"]:
            if url.path.split("/")[1] == path_base:
                all_urls.append(url.geturl())

    all_urls = sorted(list(set(all_urls)))
    return all_urls


def filter_weave_docs_urls(urls: list[str]) -> list[str]:
    """
    Filters a list of URLs to include only those related to Weave documentation.
    we think are necessary

    Args:
        urls (list): A list of URLs to filter.

    Returns:
        list: A sorted list of unique URLs that match the specified documentation paths.
    """
    all_urls = []
    for url in urls:
        url = urlparse(url)
        for path_base in ["guides", "reference", "quickstart"]:
            if url.path.split("/")[1] == path_base or url.path.split("/")[1].startswith(
                "tutorial"
            ):
                all_urls.append(url.geturl())

    all_urls = sorted(list(set(all_urls)))
    return all_urls


async def craw_parallel(urls: list[str], max_concurrent: int = 5) -> list[CrawlResult]:
    """
    Crawls multiple URLs in parallel using an asynchronous web crawler.

    Args:
        urls (list): A list of URLs to crawl.
        max_concurrent (int, optional): The maximum number of concurrent crawling tasks. Defaults to 5.

    Returns:
        list: A list of results from the crawled URLs.
    """
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(
        css_selector="article div.theme-doc-markdown.markdown",
        cache_mode=CacheMode.ENABLED,
        only_text=True,
        markdown_generator=DefaultMarkdownGenerator(
            options={
                "mark_code": True,
                "ignore_links": True,
                "ignore_images": True,
                "skip_internal_links": True,
                "single_line_break": False,
                "handle_code_in_pre": True,
            }
        ),
    )
    all_results = []
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            tasks = []

            for j, url in enumerate(batch):
                session_id = f"parallel_session_{i + j}"
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results.extend(results)
    return all_results


async def crawl_docs(sitemap_url: str, max_concurrent: int = 5) -> list[dict[str, str]]:
    """
    Crawls documentation pages from a sitemap URL and processes them.

    Args:
        sitemap_url (str): The URL of the sitemap to fetch.
        max_concurrent (int, optional): The maximum number of concurrent crawling tasks. Defaults to 5.

    Returns:
        list: A list of dictionaries containing the content and URI of each crawled document.
    """
    urls = await get_urls(sitemap_url)
    loc = urlparse(sitemap_url).netloc
    if loc == "docs.wandb.ai":
        filtered_urls = filter_wandb_docs_urls(urls)
    else:
        filtered_urls = filter_weave_docs_urls(urls)
    all_results = await craw_parallel(filtered_urls, max_concurrent=max_concurrent)
    documents = []

    for result in all_results:
        documents.append(
            {
                "content": mdify(result.html),
                "uri": result.url,
            }
        )
    return documents


async def crawl_wandb_docs(max_concurrent: int = 5) -> list[dict[str, str]]:
    """
    Crawls Weights & Biases documentation pages.

    Args:
        max_concurrent (int, optional): The maximum number of concurrent crawling tasks. Defaults to 5.

    Returns:
        list: A list of dictionaries containing the content and URI of each crawled document.
    """
    sitemap_url = "https://docs.wandb.ai/sitemap.xml"
    documents = await crawl_docs(sitemap_url, max_concurrent)
    for document in documents[:]:
        document["source"] = "wandb_docs"
        document["file_type"] = "markdown"
    return documents


async def crawl_weave_docs(max_concurrent: int = 5) -> list[dict[str, str]]:
    """
    Crawls Weave documentation pages.

    Args:
        max_concurrent (int, optional): The maximum number of concurrent crawling tasks. Defaults to 5.

    Returns:
        list: A list of dictionaries containing the content and URI of each crawled document.
    """
    sitemap_url = "https://weave-docs.wandb.ai/sitemap.xml"
    documents = await crawl_docs(sitemap_url, max_concurrent)
    for document in documents[:]:
        document["source"] = "weave_docs"
        document["file_type"] = "markdown"
    return documents


@retry(
    retry=(
        retry_if_exception_type(httpx.HTTPError)
        | retry_if_exception_type(urllib3.exceptions.HTTPError)
    ),
    stop=stop_after_attempt(10),
    wait=wait_fixed(60),
)
async def download_file(*, url: str, path: str, client: Any = None) -> str:
    """
    Atomically download a file from ``url`` to ``path``.

    If ``path`` already exists, the file will not be downloaded again.
    This means that different URLs should be saved to different paths.

    This function is meant to be used in cases where the contents of ``url``
    is immutable -- calling it more than once should always return the same bytes.

    Returns the download path.

    """
    # If the URL has already been downloaded, we can skip downloading it again.
    if os.path.exists(path):
        return path

    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        async with httpx.AsyncClient(timeout=1000, follow_redirects=True).stream(
            "GET", url
        ) as resp:
            resp.raise_for_status()
            tmp_path = f"{path}.{uuid.uuid4()}.tmp"

            with open(tmp_path, "wb") as out_file:
                async for chunk in resp.aiter_raw():
                    out_file.write(chunk)

    except Exception as exc:
        print(exc, file=sys.stderr)
        raise

    os.rename(tmp_path, path)
    return path


def load_wandb_code(repo_dir, repo_path):
    repo_dir = Path(repo_dir)
    repo_dir = list(repo_dir.glob("*"))[0]
    tests_dir = repo_dir / "tests"
    tests_files = tests_dir.rglob("*.py")
    tests_files = list(
        filter(
            lambda x: x.name not in ["__init__.py", "conftest.py", "__main__.py"],
            tests_files,
        )
    )
    out_docs = []
    base_uri = repo_path.replace("/archive/refs/tags/", "/tree/").replace(".zip", "")
    for test_file in tests_files:
        uri = test_file.relative_to(repo_dir)
        uri = f"{base_uri}/{uri}"
        doc = {
            "content": test_file.read_text(),
            "uri": uri,
            "source": "wandb_sdk",
            "file_type": "python",
        }
        out_docs.append(doc)
    return out_docs


def load_weave_code(repo_dir, repo_path):
    repo_dir = Path(repo_dir)
    repo_dir = list(repo_dir.glob("*"))[0]
    base_uri = repo_path.replace("/archive/refs/tags/", "/tree/").replace(".zip", "")
    all_files = []

    suffixes = ["md", "ipynb", "py"]
    examples_dir = repo_dir / "examples"
    for suffix in suffixes:
        all_files.extend(list(examples_dir.rglob(f"*.{suffix}")))

    tests_dir = repo_dir / "tests"
    tests_files = tests_dir.rglob("*.py")
    test_files = list(
        filter(
            lambda x: x.name not in ["__init__.py", "conftest.py", "__main__.py"],
            tests_files,
        )
    )
    all_files.extend(test_files)
    out_docs = []
    for file in all_files:
        uri = file.relative_to(repo_dir)
        uri = f"{base_uri}/{uri}"
        if file.suffix == ".ipynb":
            file_type = "notebook"
        elif file.suffix == ".py":
            file_type = "python"
        elif file.suffix == ".md":
            file_type = "markdown"
        doc = {
            "content": file.read_text(),
            "uri": uri,
            "source": "weave_sdk",
            "file_type": file_type,
        }
        out_docs.append(doc)
    return out_docs


def load_wandb_examples(repo_dir, repo_path):
    repo_dir = Path(repo_dir)
    repo_dir = list(repo_dir.glob("*"))[0]
    examples_dir = repo_dir / "examples"
    base_uri = repo_path.replace("/archive/refs/heads/", "/tree/").replace(".zip", "")
    suffixes = ["md", "ipynb", "py"]
    all_files = []
    for suffix in suffixes:
        all_files.extend(list(examples_dir.rglob(f"*.{suffix}")))
    colab_dir = repo_dir / "colabs"
    for suffix in suffixes:
        all_files.extend(list(colab_dir.rglob(f"*.{suffix}")))
    out_docs = []
    for file in all_files:
        uri = file.relative_to(repo_dir)
        uri = f"{base_uri}/{uri}"
        if file.suffix == ".ipynb":
            file_type = "notebook"
        elif file.suffix == ".py":
            file_type = "python"
        elif file.suffix == ".md":
            file_type = "markdown"
        doc = {
            "content": file.read_text(),
            "uri": uri,
            "source": "wandb_examples",
            "file_type": file_type,
        }
        out_docs.append(doc)
    return out_docs


def load_wandb_edu(repo_dir, repo_path):
    repo_dir = Path(repo_dir)
    repo_dir = list(repo_dir.glob("*"))[0]
    base_uri = repo_path.replace("/archive/refs/heads/", "/tree/").replace(".zip", "")
    suffixes = ["md", "ipynb", "py"]
    all_files = []
    for suffix in suffixes:
        all_files.extend(list(repo_dir.rglob(f"*.{suffix}")))
    out_docs = []
    for file in all_files:
        uri = file.relative_to(repo_dir)
        uri = f"{base_uri}/{uri}"
        if file.suffix == ".ipynb":
            file_type = "notebook"
        elif file.suffix == ".py":
            file_type = "python"
        elif file.suffix == ".md":
            file_type = "markdown"
        doc = {
            "content": file.read_text(),
            "uri": uri,
            "source": "wandb_edu",
            "file_type": file_type,
        }
        out_docs.append(doc)
    return out_docs


async def load_code_repos():
    code_repos = [
        (
            "https://github.com/wandb/wandb/archive/refs/tags/v0.19.3.zip",
            "data/wandb_sdk.zip",
        ),
        (
            "https://github.com/wandb/weave/archive/refs/tags/v0.51.28.zip",
            "data/weave_sdk.zip",
        ),
        (
            "https://github.com/wandb/examples/archive/refs/heads/master.zip",
            "data/wandb_examples.zip",
        ),
        (
            "https://github.com/wandb/edu/archive/refs/heads/main.zip",
            "data/wandb_edu.zip",
        ),
    ]
    code_documents = {}
    for repo, repo_path in code_repos:
        await download_file(url=repo, path=repo_path)
        repo_dir = repo_path.replace(".zip", "")
        with zipfile.ZipFile(repo_path, "r") as zip_ref:
            zip_ref.extractall(repo_dir)
        if repo_path == "data/wandb_sdk.zip":
            wandb_code = load_wandb_code(repo_dir, repo)
            code_documents["wandb_sdk"] = wandb_code
        elif repo_path == "data/weave_sdk.zip":
            weave_code = load_weave_code(repo_dir, repo)
            code_documents["weave_sdk"] = weave_code
        elif repo_path == "data/wandb_examples.zip":
            wandb_examples = load_wandb_examples(repo_dir, repo)
            code_documents["wandb_examples"] = wandb_examples
        elif repo_path == "data/wandb_edu.zip":
            wandb_edu = load_wandb_edu(repo_dir, repo)
            code_documents["wandb_edu"] = wandb_edu
    return code_documents


async def main():
    run = wandb.init(project=WANDB_PROJECT, job_type="build_dataset")
    wandb_docs = await crawl_wandb_docs(max_concurrent=5)
    weave_docs = await crawl_weave_docs(max_concurrent=5)
    code_documents = await load_code_repos()

    artifact = wandb.Artifact(
        name="documentation",
        type="dataset",
        description="Documentation for the RAG workshop",
    )
    with artifact.new_file("wandb_docs.jsonl", mode="w") as f:
        for doc in wandb_docs:
            f.write(json.dumps(doc) + "\n")
    with artifact.new_file("weave_docs.jsonl", mode="w") as f:
        for doc in weave_docs:
            f.write(json.dumps(doc) + "\n")
    for key, value in code_documents.items():
        with artifact.new_file(f"{key}_code.jsonl", mode="w") as f:
            for doc in value:
                f.write(json.dumps(doc) + "\n")

    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    asyncio.run(main())
