import os
import time
import urllib

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def download_resource(url, output_dir, pbar):
    """Downloads a resource and updates the progress bar."""
    response = requests.get(url)
    filename = os.path.basename(url)
    output_path = os.path.join(output_dir, filename)

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as file:
        file.write(response.content)

    # Update the progress bar only if pbar is not None
    if pbar is not None:
        pbar.update(1)


def scrape_website(url, output_dir):
    """Scrapes a website recursively, downloading all resources and displaying a progress bar."""
    start_time = time.time()

    # Fetch the page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Download the current HTML file
    download_resource(url, output_dir, None)  # Pass None for the first download

    # Find all links
    links = soup.find_all("a", href=True)

    # Total number of resources to download
    total_resources = len(links)

    # Initialize the progress bar
    with tqdm(total=total_resources, unit="resources", unit_scale=True) as pbar:
        for link in links:
            href = link["href"]

            # Skip empty links
            if not href:
                continue

            # Make a full URL if necessary
            if not href.startswith("http"):
                href = urllib.parse.urljoin(url, href)

            # Download the resource
            download_resource(href, output_dir, pbar)

            # Recursively scrape linked resources (except for external ones)
            if href.startswith(url):
                scrape_website(href, output_dir)

    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución: {execution_time} segundos")


# URL to scrape
url = "https://api.python.langchain.com/en/latest/langchain_api_reference.html"

# Directory to store downloaded content
output_dir = "./langchain_docs/"

scrape_website(url, output_dir)
