from collections import deque
from time import time
from bs4 import BeautifulSoup
import re
import requests
from googlesearch import search
from urllib.parse import quote


headers = {'User-Agent': 'Mozilla/5.0'}

def search_web(query: str, num_results: int = 10) -> list[dict[str, str]]:
    num_wiki = num_results // 2
    num_gfg = num_results - num_wiki
    
    # crawl
    wikipedia_urls = crawl_wikipedia_urls(query, max_visits=num_wiki)
    gfg_urls = get_gfg_urls(query, limit=num_gfg)
    
    # scrape
    scraped_wikipedia = scrap_text_from_urls_wikipedia(wikipedia_urls)
    
    result = []
    result.extend(scraped_wikipedia)
    return result
    

def crawl( query: str, num_results: int = 10) -> list[dict[str, str]]:
    result = crawl_wikipedia_urls(query, max_visits=num_results)
    result.extend(get_gfg_urls(query, limit=num_results))
    return result
    

def scrape(sites: list[dict[str, str]]) -> list[dict[str, str]]:
    return scrap_text_from_urls_wikipedia(sites)

def get_wikipedia_urls(query: str, language: str = "en", limit=5) -> list[dict[str, str]]:
    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    titles = [result["title"] for result in data["query"]["search"][:limit]]
    urls = []
    for title in titles:
        url = f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        urls.append({"title": title, "url": url})
    return urls

def crawl_wikipedia_urls( seed_query: str, language: str = "en", max_visits: int = 30) -> list[dict[str, str]]:
    visited = set()
    queue = deque()
    base_url = f"https://{language}.wikipedia.org"
    results = []

    seeds = get_wikipedia_urls(seed_query, language)
    for seed in seeds:
        queue.append(seed)

    while queue and len(visited) < max_visits:
        current = queue.popleft()
        current_url = current['url']
        if current_url in visited:
            continue
        visited.add(current_url)
        results.append({'title': current.get('title', ''), 'url': current_url})
        try:
            page = requests.get(current_url, headers=headers, timeout=10)
            soup = BeautifulSoup(page.text, 'html.parser')
            links_table = soup.find('div', id='bodyContent')
            if links_table:
                for link in links_table.find_all('a', href=True):
                    href = link['href']
                    if href.startswith('/wiki/') and not any(href.startswith(p) for p in [
                        '/wiki/Special:', '/wiki/Talk:', '/wiki/User:', '/wiki/Help:',
                        '/wiki/File:', '/wiki/Portal:', '/wiki/Category:']):
                        full_url = base_url + href
                        if full_url not in visited:
                            link_title = link.get('title') or href.split('/')[-1].replace('_', ' ')
                            queue.append({'title': link_title, 'url': full_url})
            time.sleep(0.5)
        except Exception:
            continue 
    return results

def scrap_text_from_urls_wikipedia(links: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for item in links:
        url = item.get('url')
        try:
            page = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(page.text, 'html.parser')
            body = soup.find('div', id='bodyContent')
            if body:
                for tag in body(['script', 'style']):
                    tag.decompose()

                text = body.get_text(separator=' ', strip=True)
                text = re.sub(r'\s+', ' ', text)
            else:
                continue
        except Exception:
            continue

        result.append({
            "url": url,
            "text": text[:5000],
            "length": len(text)
        })

    return result


def get_gfg_urls(query: str, limit: int = 5) -> list[dict[str, str]]:
    
    url = f"https://www.geeksforgeeks.org/search/?gq={query.replace(' ', '+')}"

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    articles = soup.find_all('article')
    links = []

    for article in articles:
        a_tag = article.find('a', href=True)
        if a_tag:
            links.append({"url": a_tag['href']})
        if len(links) >= limit:
            break

    return links