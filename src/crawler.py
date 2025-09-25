# -*- coding: utf-8 -*-
"""
AWS Documentation Crawler
Crawls AWS documentation from sitemaps and converts to markdown files.
"""
from __future__ import annotations
import asyncio, aiohttp, async_timeout, gzip, io, re, hashlib, time, logging
from pathlib import Path
from typing import AsyncIterator, List, Tuple, Dict, Optional
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import urllib.robotparser as robotparser

# pip install aiohttp markdownify
from markdownify import markdownify as html2md

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("awsdocs_crawler")

# Configuration constants
SITEMAP_INDEX = "https://docs.aws.amazon.com/sitemap_index.xml"
DEFAULT_OUT_DIR = Path("aws_docs")
UA = "Sam-Lewis/1.0 (samlewis1999.sl@gmail.com)"

# Service alias mapping for CLI slugs to AWS URL tokens
SERVICE_ALIASES = {
    "s3": ["AmazonS3"],
    "iam": ["IAM"],
    "ec2": ["EC2"],
    "lambda": ["lambda"],
    "rds": ["AmazonRDS"],
    "dynamodb": ["amazondynamodb"],
    "cloudformation": ["AWSCloudFormation"],
    "eks": ["eks"],
    "ecs": ["AmazonECS"],
    "vpc": ["vpc"],
    # add more as needed
}

ALLOWED_PREFIXES = [
    # S3
    "https://docs.aws.amazon.com/AmazonS3/latest/userguide/",
    "https://docs.aws.amazon.com/AmazonS3/latest/API/",
    # IAM
    "https://docs.aws.amazon.com/IAM/latest/UserGuide/",
    "https://docs.aws.amazon.com/IAM/latest/APIReference/",
    # Add more services as needed
]

SKIP_SUBSTRINGS = [
    "/javadoc/", "/java/api/", "/sdk-for-", "/dotnet/", "/cpp/", "/golang/",
    "/ruby/", "/php/", "/powershell/", "/python/latest/reference/",
]


class AWSDocsCrawler:
    """Main crawler class for AWS documentation."""
    
    def __init__(self, 
                 out_dir: Path = DEFAULT_OUT_DIR,
                 user_agent: str = UA,
                 sitemap_index: str = SITEMAP_INDEX):
        self.out_dir = out_dir
        self.user_agent = user_agent
        self.sitemap_index = sitemap_index
    
    # -------- Utility functions --------
    
    @staticmethod
    def sha1(b: bytes) -> str:
        """Generate SHA1 hash of bytes."""
        return hashlib.sha1(b).hexdigest()

    @staticmethod
    def safe_slug(url: str) -> str:
        """Create a safe filename slug from URL."""
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", url.split("/")[-1].split(".")[0]).strip("-")
        return slug or "index"

    @staticmethod
    def service_match(url: str, limit_services: list[str] | None) -> bool:
        """Check if URL matches any of the requested services using aliases."""
        if not limit_services:
            return True
        url_l = url.lower()
        for slug in limit_services:
            tokens = SERVICE_ALIASES.get(slug.lower(), [slug])
            for token in tokens:
                if f"/{token}/".lower() in url_l:
                    return True
        return False

    async def fetch(self, session: aiohttp.ClientSession, url: str, *, expect_xml=False) -> bytes:
        """Fetch a URL with retry logic and error handling."""
        for attempt in range(4):
            try:
                timeout = 30
                async with async_timeout.timeout(timeout):
                    async with session.get(url, timeout=timeout) as resp:
                        if resp.status == 200:
                            raw = await resp.read()
                            # handle *.xml.gz sitemaps
                            if expect_xml and url.endswith(".gz"):
                                return gzip.decompress(raw)
                            return raw
                        elif resp.status in (429, 500, 502, 503, 504):
                            backoff = 2 ** attempt
                            log.warning("Retry %s %s after %ss (status %s)", attempt+1, url, backoff, resp.status)
                            await asyncio.sleep(backoff)
                        else:
                            raise RuntimeError(f"HTTP {resp.status} for {url}")
            except Exception as e:
                if attempt == 3: 
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to fetch {url}")

    @staticmethod
    def parse_sitemap(xml_bytes: bytes) -> List[str]:
        """Parse sitemap XML and return list of URLs."""
        root = ET.fromstring(xml_bytes)
        locs = []
        # Robust extraction across odd namespaces
        for el in root.iter():
            if el.tag.endswith("loc"):
                locs.append(el.text.strip())
        return locs

    async def load_robots(self, base: str, session: aiohttp.ClientSession) -> robotparser.RobotFileParser:
        """Load and parse robots.txt for the given base URL."""
        parsed = urlparse(base)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        try:
            content = await self.fetch(session, robots_url)
            rp.parse(content.decode("utf-8", errors="ignore").splitlines())
        except Exception:
            log.warning("robots.txt not found or unreadable at %s; be extra conservative.", robots_url)
            rp.disallow_all = True
        return rp

    def allowed(self, rp: robotparser.RobotFileParser, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return False

    @staticmethod
    def html_to_markdown(url: str, html: bytes) -> str:
        """Convert HTML to Markdown with source header."""
        md = html2md(html.decode("utf-8", errors="ignore"), strip=["script", "style"])
        # Keep a header with provenance
        head = f"<!-- Source: {url} -->\n\n"
        return head + md

    @staticmethod
    def is_allowed_doc(url: str) -> bool:
        """Check if document URL is in allowed list."""
        if not any(url.startswith(p) for p in ALLOWED_PREFIXES):
            return False
        if any(s in url for s in SKIP_SUBSTRINGS):
            return False
        return url.endswith(".html")

    # -------- Main pipeline functions --------

    async def iter_all_doc_urls(self, 
                                session: aiohttp.ClientSession,
                                limit_services: list[str] | None = None
                               ) -> AsyncIterator[tuple[str, str]]:
        """Iterate through all document URLs from sitemaps."""
        idx_bytes = await self.fetch(session, self.sitemap_index, expect_xml=True)
        all_sitemaps = self.parse_sitemap(idx_bytes)

        log.info("Sitemap index returned %d entries", len(all_sitemaps))

        # Filter to service sitemaps using the improved service matching
        if limit_services:
            all_sitemaps = [
                sm for sm in all_sitemaps
                if self.service_match(sm, limit_services)
            ]

        log.info("After service filter: %d sitemaps", len(all_sitemaps))

        for sm in all_sitemaps:
            try:
                # Check Content-Type before treating it as XML
                async with session.get(sm) as resp:
                    ctype = resp.headers.get("Content-Type", "")
                    body = await resp.read()
                if "xml" not in ctype:
                    # Not an XML sitemap—skip quietly
                    continue

                sm_bytes = gzip.decompress(body) if sm.endswith(".gz") else body
                pages = self.parse_sitemap(sm_bytes)
                for page in pages:
                    if self.is_allowed_doc(page) and self.service_match(page, limit_services):
                        yield sm, page
            except Exception:
                # Bad or odd sitemap—skip and move on
                continue

    async def crawl(self, 
                   limit_services: Optional[List[str]] = None, 
                   max_pages: Optional[int] = None,
                   progress_interval: int = 25) -> int:
        """
        Main crawling function.
        
        Args:
            limit_services: List of service names to crawl (e.g., ['s3', 'iam'])
            max_pages: Maximum number of pages to crawl
            progress_interval: How often to log progress
            
        Returns:
            Number of pages successfully crawled
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        count = 0

        async with aiohttp.ClientSession(headers={"User-Agent": self.user_agent}) as session:
            rp = await self.load_robots(self.sitemap_index, session)

            # Single loop over sitemap pages
            async for sm, page in self.iter_all_doc_urls(session, limit_services=limit_services):
                if max_pages and count >= max_pages:
                    log.info("Reached max_pages limit of %d", max_pages)
                    break
                    
                if not self.allowed(rp, page):
                    continue
                if not self.is_allowed_doc(page):
                    continue
                if not self.service_match(page, limit_services):
                    continue

                norm = page.split("#")[0]  # dedupe ignoring fragments
                if norm in seen:
                    continue
                seen.add(norm)

                try:
                    html = await self.fetch(session, page)
                except Exception as e:
                    log.warning("Fetch fail %s: %s", page, e)
                    continue

                md = self.html_to_markdown(page, html)

                # Improved service extraction using regex
                service_match_obj = re.search(r"docs\.aws\.amazon\.com/([^/]+)/latest/", page)
                service = service_match_obj.group(1) if service_match_obj else "misc"
                subdir = self.out_dir / service
                subdir.mkdir(parents=True, exist_ok=True)

                slug = self.safe_slug(page)
                path_md = subdir / f"{slug}.md"
                path_jsonl = self.out_dir / "index.jsonl"

                try:
                    path_md.write_text(md, encoding="utf-8")
                    meta = {
                        "url": page,
                        "service": service,
                        "slug": slug,
                        "sha1": self.sha1(html),
                        "ts": int(time.time()),
                    }
                    with path_jsonl.open("a", encoding="utf-8") as f:
                        f.write(str(meta) + "\n")
                    count += 1
                    if count % progress_interval == 0:
                        log.info("Saved %s pages (latest: %s)", count, page)
                except Exception as e:
                    log.warning("Write fail %s: %s", page, e)

            log.info("Done. Saved %s pages.", count)
            return count


# Convenience functions for backward compatibility and easy import
async def crawl_aws_docs(limit_services: Optional[List[str]] = None, 
                        max_pages: Optional[int] = None,
                        out_dir: Path = DEFAULT_OUT_DIR) -> int:
    """
    Convenience function to crawl AWS documentation.
    
    Args:
        limit_services: List of service names to crawl (e.g., ['s3', 'iam'])
        max_pages: Maximum number of pages to crawl
        out_dir: Output directory for markdown files
        
    Returns:
        Number of pages successfully crawled
    """
    crawler = AWSDocsCrawler(out_dir=out_dir)
    return await crawler.crawl(limit_services=limit_services, max_pages=max_pages)


def main():
    """CLI entry point for the crawler."""
    import sys
    
    # Parse command line arguments
    svcs = [a for a in sys.argv[1:] if not a.startswith("--")]
    max_pages = None
    out_dir = DEFAULT_OUT_DIR
    
    for arg in sys.argv[1:]:
        if arg.startswith("--max"):
            try: 
                max_pages = int(arg.split("=")[1])
            except: 
                pass
        elif arg.startswith("--out"):
            try:
                out_dir = Path(arg.split("=")[1])
            except:
                pass
    
    # Run crawler
    result = asyncio.run(crawl_aws_docs(
        limit_services=svcs or None, 
        max_pages=max_pages,
        out_dir=out_dir
    ))
    
    print(f"Crawling complete! Saved {result} pages to {out_dir}")


if __name__ == "__main__":
    main()