from collections import deque
from bs4 import BeautifulSoup
import re
import requests
from googlesearch import search
from urllib.parse import quote


class Crawler:
    def __init__(self):
        pass

    def crawl(self, query: str, num_results: int = 10) -> list[str]:
        language = "en"
        return self._crawl_wikipedia_urls(query, language, max_visits=num_results)
        

    def scrape(self, sites: list[dict[str, str]]) -> list[dict[str, str]]:
        return self._scrap_text_from_urls(sites)

    def _get_seeds(self, query: str, limit=5):
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }

        response = requests.get(url, params=params)
        data = response.json()

        titles = [result["title"] for result in data["query"]["search"]]
        titles = titles[:limit]
        urls = [f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}" for title in titles]

        return urls

    def _crawl_wikipedia_urls(self, seed_query: str, language: str = "es", max_visits: int = 30) -> list[dict[str, str]]:
        visited = set()
        queue = deque()
        base_url = f"https://{language}.wikipedia.org"
        results = []

        seeds = self._get_seeds(seed_query, language)
        for url in seeds:
            queue.append({'url': url})

        while queue and len(visited) < max_visits:
            current = queue.popleft()
            current_url = current['url']
            if current_url in visited:
                continue
            visited.add(current_url)
            results.append({'url': current_url})
            try:
                page = requests.get(current_url, timeout=10)
                soup = BeautifulSoup(page.text, 'html.parser')
                links_table = soup.find('div', class_='columns')
                if links_table:
                    for link in links_table.find_all('a', href=True):
                        href = link['href']
                        if href.startswith('/wiki/') and not href.startswith('/wiki/Special:'):
                            full_url = base_url + href
                            if full_url not in visited:
                                link_title = link.get('title') or href.split('/')[-1].replace('_', ' ')
                                queue.append({'title': link_title, 'url': full_url})
            except Exception:
                continue 
        return results

    def _scrap_text_from_urls(self, links: list[dict[str, str]]) -> list[dict[str, str]]:
        result = []
        html_tag_pattern = re.compile(r'<[^>]+>')
        for item in links:
            url = item.get('url')
            try:
                page = requests.get(url, timeout=10)
                soup = BeautifulSoup(page.text, 'html.parser')
                body = soup.find('div', id='bodyContent')
                if body:
                    for tag in body(['script', 'style']):
                        tag.decompose()
                    text = body.get_text(separator=' ', strip=True)
                    text = re.sub(r'\s+', ' ', text)
                    text = html_tag_pattern.sub('', text)
                else:
                    text = "No se encontró el div bodyContent."
            except Exception:
                continue
            result.append({"url": url, "text": text})
        return result