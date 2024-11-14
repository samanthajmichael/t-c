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
        url = "https://x.com/en/privacy" #using the privacy policy
        selector = '#tos-content > div > div > article > div > div'

    elif platform == 'meta - facebook':
        url = "https://www.facebook.com/privacy/policy"
        selector = '#terms_of_service_content'

    elif platform == 'instagram':
        url = "https://help.instagram.com/581069655588408"
        selector = '#content > div:nth-child(2) > div > article > div:nth-child(2)'
    elif platform == 'snapchat':
        url = "https://www.snapchat.com/legal/terms-of-service"
        selector = '#terms-of-service-content'
    elif platform == 'youtube':
        url = "https://www.youtube.com/static?template=terms"
        selector = '#content > div > div > article > div > div'
    elif platform == 'telegram':
        url = "https://core.telegram.org/terms"
        selector = '#terms'
    elif platform == 'whatsapp':
        url = "https://www.whatsapp.com/legal/terms-of-service"
        selector = '#terms-of-service'
    elif platform == 'wechat':
        url = "https://www.wechat.com/en/agreement.html"
        selector = '#container > div.agreement > div > div > div > div.agreement-box > div.agreement-content'
    elif platform == 'kuaishou':
        url = "https://kuaishou.com/user/agreement"
        selector = 'body > div.container > div > div.article-content'
    elif platform == 'sinaweibo':
        url = "https://weibo.com/legal/agreement"
        selector = 'body > div.wrap > div > div > div > div.article > div.content > div'

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

# Example usage:
platforms = ['twitter', 'facebook', 'instagram', 'snapchat', 'youtube', 'telegram',
            'whatsapp', 'wechat', 'kuaishou', 'sinaweibo'] 

for platform in platforms:
    terms = scrape_terms_of_service(platform)
    if terms:
        print(f"Platform: {platform}")
        with open(f"{platform}_terms.txt", 'w', encoding='utf-8') as file:
            file.write(f"Terms of Service:\n{terms}\n")
    else:
        print(f"Error scraping terms for {platform}")

    sleep(5)  # Add a delay to avoid overwhelming servers