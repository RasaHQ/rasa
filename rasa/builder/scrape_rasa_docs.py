import json
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://rasa.com"
DOCS_ROOT = "https://rasa.com/docs"
OUTPUT_DIR = "rasa_docs_md"
MAX_PAGES = 100  # Optional limit for safety

visited = set()
to_visit = [DOCS_ROOT]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_valid_doc_url(url: str) -> bool:
    return url.startswith(DOCS_ROOT) and not any(
        [url.endswith(".pdf"), "#" in url, "mailto:" in url]
    )


def slugify_url(url: str) -> str:
    path = urlparse(url).path.strip("/").replace("/", "_")
    return path if path else "index"


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove navs, footers, and code tabs (customize if needed)
    for tag in soup(["nav", "footer", "script", "style", "form", "button"]):
        tag.decompose()

    main = soup.find("main") or soup.body
    if not main:
        return ""

    # Replace <code> with backticks
    for code in main.find_all("code"):
        code.string = f"`{code.get_text(strip=True)}`"

    text = main.get_text(separator="\n", strip=True)
    return text


def save_as_markdown(text: str, url: str) -> str:
    slug = slugify_url(url)
    file_name = f"{slug}.md"
    md_path = Path(OUTPUT_DIR) / file_name
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Saved: {md_path}")
    return file_name


pages_scraped = 0
markdown_to_url = {}

while to_visit and pages_scraped < MAX_PAGES:
    url = to_visit.pop(0)
    if url in visited:
        continue

    try:
        print(f"Scraping: {url}")
        response = requests.get(url)
        response.raise_for_status()

        html = response.text
        text = clean_text(html)
        if len(text) < 200:  # skip very short pages
            print("⏭️ Skipped (too short)")
            continue

        file_name = save_as_markdown(text, url)
        markdown_to_url[file_name] = url
        pages_scraped += 1

        soup = BeautifulSoup(html, "html.parser")
        for link_tag in soup.find_all("a", href=True):
            link = urljoin(url, link_tag["href"])
            if is_valid_doc_url(link) and link not in visited:
                to_visit.append(link)

        visited.add(url)

    except Exception as e:
        print(f"⚠️ Failed to scrape {url}: {e}")


with open("markdown_to_url.json", "w") as f:
    json.dump(markdown_to_url, f, indent=2)
