import requests
from bs4 import BeautifulSoup
from time import sleep

def scrape_terms_of_service(platform):
    """Scrapes the terms of service from a social media platform.

    Args:
        platform (str): The name of the platform.

    Returns:
        str: The extracted terms of service text, or None if scraping failed.
    """  
# Define platform-specific URLs and selectors
    if platform == 'twitter - x':
        url = "https://x.com/en/privacy"  # Using the privacy policy since no dedicated TOS
        selector = 'div[data-testid="privacy-policy-section-container"] > div > div' 

    elif platform == 'meta - facebook':
        url = "https://www.facebook.com/privacy/policy"
        selector = '#terms_of_service_content'

# Fetch the webpage
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the terms of service content using the selector
        terms_section = soup.select_one(selector)
        if terms_section:
            terms_of_service = terms_section.get_text(separator=' ').strip()
            return terms_of_service
        else:
            print(f"Error: Couldn't find terms of service section for {platform}")
            return None
    else:
        print(f"Error: Unable to fetch page for {platform}")
        return None

import requests
from bs4 import BeautifulSoup
from time import sleep

def scrape_terms_of_service(platform):
    """Scrapes the terms of service from a social media platform.

    Args:
        platform (str): The name of the platform.

    Returns:
        str: The extracted terms of service text, or None if scraping failed.
    """  
# Define platform-specific URLs and selectors
    if platform == 'twitter - x':
        url = "https://x.com/en/privacy"  # Using the privacy policy since no dedicated TOS
        selector = 'div[data-testid="privacy-policy-section-container"] > div > div' 

    elif platform == 'meta - facebook':
        url = "https://www.facebook.com/privacy/policy"
        selector = '#terms_of_service_content'

# Fetch the webpage
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the terms of service content using the selector
        terms_section = soup.select_one(selector)
        if terms_section:
            terms_of_service = terms_section.get_text(separator=' ').strip()
            return terms_of_service
        else:
            print(f"Error: Couldn't find terms of service section for {platform}")
            return None
    else:
        print(f"Error: Unable to fetch page for {platform}")
        return None
        