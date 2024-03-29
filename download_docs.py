import os
import time
import urllib

import requests
from bs4 import BeautifulSoup

# Inicio del temporizador
start_time = time.time()

# Ejecución de la tarea
# The URL to scrape
url = "https://api.python.langchain.com/en/latest/api_reference.html"

# The directory to store files in
output_dir = "./langchain-docs/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Fetch the page
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find all links to .html files
links = soup.find_all("a", href=True)

for link in links:
    href = link["href"]

    # If it's a .html file
    if href.endswith(".html"):
        # Make a full URL if necessary
        if not href.startswith("http"):
            href = urllib.parse.urljoin(url, href)

        # Fetch the .html file
        file_response = requests.get(href)

        # Write it to a file
        file_name = os.path.join(output_dir, os.path.basename(href))
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(file_response.text)

# Fin del temporizador
end_time = time.time()

# Cálculo del tiempo de ejecución
tiempo_ejecucion = end_time - start_time

# Impresión del tiempo de ejecución
print(f"Tiempo de ejecución: {tiempo_ejecucion} segundos")
