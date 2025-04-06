import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

# Load CSV
df = pd.read_csv("combined_catalog.csv")

# Configure Selenium WebDriver (Headless Chrome)
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920x1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Storage for new data
descriptions, job_levels, languages, assessment_lengths = [], [], [], []

for index, row in df.iterrows():
    url = row['URL']
    print(f"Scraping: {url}")

    try:
        driver.get(url)
        time.sleep(2)  # Let the page load

        # Extract Description
        try:
            desc_element = driver.find_element(By.XPATH, "//h4[text()='Description']/following-sibling::p")
            description = desc_element.text.strip()
        except:
            description = "Not Found"

        # Extract Job Levels
        try:
            job_element = driver.find_element(By.XPATH, "//h4[text()='Job levels']/following-sibling::p")
            job_level = job_element.text.strip()
        except:
            job_level = "Not Found"

        # Extract Languages
        try:
            lang_element = driver.find_element(By.XPATH, "//h4[text()='Languages']/following-sibling::p")
            languages_text = lang_element.text.strip()
        except:
            languages_text = "Not Found"

        # Extract Assessment Length
        try:
            length_element = driver.find_element(By.XPATH, "//h4[text()='Assessment length']/following-sibling::p")
            assessment_length = length_element.text.strip()
        except:
            assessment_length = "Not Found"

    except Exception as e:
        print(f"Error at {url}: {e}")
        description, job_level, languages_text, assessment_length = ["Not Found"] * 4

    # Append data
    descriptions.append(description)
    job_levels.append(job_level)
    languages.append(languages_text)
    assessment_lengths.append(assessment_length)

# Add new columns to DataFrame
df['Description'] = descriptions
df['Job Levels'] = job_levels
df['Languages'] = languages
df['Assessment Length'] = assessment_lengths

# Save updated CSV
df.to_csv("shl_product_catalog_updated.csv", index=False)

print("✅ Data extraction complete. File saved as 'shl_product_catalog_updated.csv'.")

# Close browser
driver.quit()
