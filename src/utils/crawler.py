import os
from firecrawl import FirecrawlApp, ScrapeOptions
from dotenv import load_dotenv

load_dotenv()

class Crawler:
    def __init__(self):
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("Missing FIRECRAWL_API_KEY environment variable")
        self.app = FirecrawlApp(api_key=api_key)

    def crawl(self, query: str, num_results: int = 10):
        try:
            result = self.app.search(
                query=query,
                lang="en",
                limit=num_results,
                scrape_options=ScrapeOptions(
                    formats=["markdown"]
                )
            )
            return result.data
        except Exception as e:
            print(e)
            return []

    def scrape(self, url: str):
        try:
            result = self.app.scrape_url(
                url,
                formats=["markdown"]
            )
            return result
        except Exception as e:
            print(e)
            return None