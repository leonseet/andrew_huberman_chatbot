import requests
from bs4 import BeautifulSoup

INPUT_FILE_PATH = "https://www.hubermanlab.com/sitemap.xml"
FILE_OUTPUT_PATH = "data/episode_urls.txt"

import requests


def loadRSS(url):
    """
    Downloads the RSS feed from the specified URL and saves it to a file.

    Args:
        url (str): The URL of the RSS feed to download.

    Returns:
        None
    """
    resp = requests.get(url)
    with open("data/sitemap.xml", "wb") as f:
        f.write(resp.content)


def parseXML(xmlfile):
    """
    Parses an XML file and extracts all URLs contained within <loc> tags.

    Args:
    xmlfile (str): The path to the XML file to be parsed.

    Returns:
    list: A list of URLs extracted from the XML file.
    """
    with open(xmlfile, "r") as f:
        file = f.read()

    soup = BeautifulSoup(file, "xml")
    locs = soup.find_all("loc")

    urls = []

    for url in locs:
        urls.append(url.text.strip())

    return urls


def filter_episodes_urls(urls):
    """
    Filters a list of URLs to only include those that contain '/episode/' in the URL path.

    Args:
        urls (list): A list of URLs to filter.

    Returns:
        list: A list of URLs that contain '/episode/' in the URL path.
    """
    episodes = []
    for url in urls:
        if "/episode/" in url:
            episodes.append(url)
    return episodes


if __name__ == "__main__":
    loadRSS(INPUT_FILE_PATH)
    urls = parseXML("data/sitemap.xml")
    urls = filter_episodes_urls(urls)
    urls_text = "\n".join(urls)

    with open(FILE_OUTPUT_PATH, "w") as file:
        file.write(urls_text)
