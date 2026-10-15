"""Extract job description text from a URL.

Uses requests + BeautifulSoup to scrape job posting pages. Attempts to
extract the main content while filtering out navigation, ads, and other
noise.
"""
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def is_valid_url(url):
    """Basic URL validation."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def extract_job_description_from_url(url, timeout=10):
    """Fetch and extract job description text from a URL.
    
    Returns:
        tuple: (success: bool, text_or_error: str)
    """
    if not is_valid_url(url):
        return False, "Invalid URL format. Please enter a valid URL starting with http:// or https://"
    
    try:
        # Set headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the page
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                            'aside', 'iframe', 'noscript', 'form']):
            element.decompose()
        
        # Try to find main content containers (common class/id names)
        main_content = None
        content_selectors = [
            # Common job posting containers
            {'class_': re.compile(r'job.*description', re.I)},
            {'id': re.compile(r'job.*description', re.I)},
            {'class_': re.compile(r'description', re.I)},
            {'id': re.compile(r'description', re.I)},
            {'class_': re.compile(r'job.*detail', re.I)},
            {'id': re.compile(r'job.*detail', re.I)},
            {'class_': re.compile(r'content', re.I)},
            {'id': re.compile(r'content', re.I)},
            {'role': 'main'},
            {'class_': re.compile(r'posting', re.I)},
        ]
        
        for selector in content_selectors:
            main_content = soup.find('div', selector) or soup.find('section', selector) or soup.find('article', selector)
            if main_content:
                break
        
        # If no main content found, try <main> tag
        if not main_content:
            main_content = soup.find('main')
        
        # If still nothing, use body
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return False, "Could not extract content from the page."
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        if len(text.strip()) < 100:
            return False, "Extracted text is too short. The page may not contain a job description or requires JavaScript rendering."
        
        return True, text
        
    except requests.exceptions.Timeout:
        return False, f"Request timed out after {timeout} seconds. Please try again."
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the URL. Please check your internet connection."
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP error occurred: {e.response.status_code}"
    except Exception as e:
        return False, f"Error scraping URL: {str(e)}"


def extract_company_and_title(soup):
    """Try to extract company name and job title (optional enhancement)."""
    # This can be enhanced based on common job board patterns
    company = None
    title = None
    
    # Try common patterns
    company_elem = soup.find('span', {'class': re.compile(r'company', re.I)})
    if company_elem:
        company = company_elem.get_text(strip=True)
    
    title_elem = soup.find('h1') or soup.find('title')
    if title_elem:
        title = title_elem.get_text(strip=True)
    
    return company, title
