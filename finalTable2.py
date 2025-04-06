from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# Set up headless Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Launch WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# List to store extracted data
assessments = []

# Function to extract data from the current page
def extract_data():
    rows = driver.find_elements(By.XPATH, "//tr[@data-entity-id]")
    
    for row in rows:
        try:
            # Extract Name and Link
            title_element = row.find_element(By.CLASS_NAME, "custom__table-heading__title")
            name = title_element.text.strip()
            link = title_element.find_element(By.TAG_NAME, "a").get_attribute("href")

            # Extract Remote Support & Adaptive Support
            general_elements = row.find_elements(By.CLASS_NAME, "custom__table-heading__general")
            remote_support = "Yes" if "catalogue__circle -yes" in general_elements[0].get_attribute("innerHTML") else "No"
            adaptive_support = "Yes" if "catalogue__circle -yes" in general_elements[1].get_attribute("innerHTML") else "No"

            # Extract Assessment Types
            types_element = row.find_element(By.CLASS_NAME, "product-catalogue__keys")
            assessment_types = types_element.text.strip()

            assessments.append([name, link, remote_support, adaptive_support, assessment_types])

        except Exception as e:
            print(f"Error processing row: {e}")

# Iterate through pages with increments of 12 (from 0 to 372)
for start in range(0, 373, 12):
    url = f"https://www.shl.com/solutions/products/product-catalog/?start={start}&type=1&type=1"
    print(f"📄 Processing URL with start={start} ...")
    driver.get(url)
    time.sleep(3)  # Wait for content to load
    extract_data()

# Explicitly process the last page again (372)
last_page_url = "https://www.shl.com/solutions/products/product-catalog/?start=372&type=1&type=1"
driver.get(last_page_url)
time.sleep(5)  # Ensure full load
extract_data()

print(f"Total assessments extracted: {len(assessments)}")

# Close WebDriver
driver.quit()

# Save data to CSV
df = pd.DataFrame(assessments, columns=["Assessment Name", "URL", "Remote Support", "Adaptive Support", "Types"])
df.to_csv("table2.csv", index=False)

print(f"✅ Data extraction complete! {len(assessments)} items saved in 'shl_product_catalog_table2.csv'.")
