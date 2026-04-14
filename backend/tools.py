import logging
import requests
import asyncio
from typing import Tuple, List, Dict, Optional
from config import SEARXNG_URL, SEARXNG_SECRET_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def search_web(query: str, max_results: int = 30, time_range: Optional[str] = None, format: str = "html") -> str:
    """
    Unified web search using SearXNG with support for both HTML and JSON parsing.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        time_range: Time filter - 'day', 'week', 'month', 'year', or None for all time
        format: Output format - 'html' for detailed parsing or 'json' for structured data
    """
    try:
        loop = asyncio.get_running_loop()

        def fetch_results() -> Tuple[List[Dict], str]:
            search_endpoint = f"{SEARXNG_URL.rstrip('/')}/search"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            time_info = f" (past {time_range})" if time_range else " (all time)"
            logger.info(f"🔍 Searching: {query}{time_info}")

            params = {
                "q": query,
                "categories": "general",
                "safesearch": 1,
            }

            # Add time range filter if specified
            if time_range and time_range in ['day', 'week', 'month', 'year']:
                params['time_range'] = time_range
                logger.info(f"⏰ Time filter: {time_range}")

            # Try JSON format first
            if format == "json":
                params['format'] = 'json'
                try:
                    response = requests.get(
                        search_endpoint,
                        params=params,
                        headers=headers,
                        timeout=15
                    )
                    response.raise_for_status()
                    data = response.json()
                    results = data.get("results", [])

                    if results:
                        formatted_results = []
                        for result in results[:max_results]:
                            formatted_results.append({
                                'title': result.get("title", "No title"),
                                'content': result.get("content", "No description available"),
                                'url': result.get("url", ""),
                                'published': result.get("publishedDate")
                            })
                        logger.info(f"✓ Parsed {len(formatted_results)} results (JSON)")
                        return formatted_results, "json"
                except Exception as e:
                    logger.warning(f"JSON format failed, falling back to HTML: {e}")

            # HTML parsing (fallback or default)
            response = requests.get(
                search_endpoint,
                params=params,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            logger.info(f"✓ Response received")

            # Parse HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            results = []
            for article in soup.select('article.result'):
                try:
                    title_elem = article.select_one('h3 a')
                    content_elem = article.select_one('.content, p.content')

                    # Try to extract published date if available
                    published_elem = article.select_one('.published_date, time, .date')
                    published_date = published_elem.get_text(strip=True) if published_elem else None

                    if title_elem:
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'content': content_elem.get_text(strip=True) if content_elem else '',
                            'url': title_elem.get('href', ''),
                            'published': published_date
                        })
                except:
                    continue

            logger.info(f"✓ Parsed {len(results)} results (HTML)")
            return results, "html"

        results, method = await loop.run_in_executor(None, fetch_results)

        if not results:
            return f"No results found for '{query}'{' in the past ' + time_range if time_range else ''}."

        # Format output
        summaries = []
        for idx, result in enumerate(results[:max_results], 1):
            title = result.get('title', 'No title')
            content = result.get('content', '')[:200]
            url = result.get('url', '')
            published = result.get('published')

            result_text = f"{idx}. {title}\n"
            if published:
                result_text += f"   📅 {published}\n"
            result_text += f"   {content}{'...' if len(result.get('content', '')) > 200 else ''}\n"
            result_text += f"   URL: {url}\n"

            summaries.append(result_text)

        logger.info(f"✓ Returning {len(summaries)} results")
        return "\n".join(summaries)

    except requests.Timeout:
        error_msg = f"⏱️ Search timeout for: {query}"
        logger.error(error_msg)
        return error_msg
    except requests.RequestException as e:
        error_msg = f"❌ Search error for '{query}': {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Unexpected error during search: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg
