from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# Set up Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode (no UI)
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

assessments = []

# Function to extract data from the current page
def extract_data():
    rows = driver.find_elements(By.XPATH, "//tr[@data-course-id]")
    for row in rows:
        try:
            title_element = row.find_element(By.CLASS_NAME, "custom__table-heading__title")
            name = title_element.text.strip()
            link = title_element.find_element(By.TAG_NAME, "a").get_attribute("href")

            remote_element = row.find_elements(By.CLASS_NAME, "custom__table-heading__general")[0]
            remote_support = "Yes" if "catalogue__circle -yes" in remote_element.get_attribute("innerHTML") else "No"

            adaptive_element = row.find_elements(By.CLASS_NAME, "custom__table-heading__general")[1]
            adaptive_support = "Yes" if "catalogue__circle -yes" in adaptive_element.get_attribute("innerHTML") else "No"

            types_elements = row.find_elements(By.CLASS_NAME, "product-catalogue__key")
            assessment_types = [elem.text.strip() for elem in types_elements]

            assessments.append([name, link, remote_support, adaptive_support, ", ".join(assessment_types)])

        except Exception as e:
            print(f"Error processing row: {e}")

# Loop through pages using 'start' parameter: 0 to 132 inclusive
for start in range(0, 133, 12):
    url = f"https://www.shl.com/solutions/products/product-catalog/?start={start}&type=2&type=2"
    print(f"📄 Processing URL with start={start} ...")
    driver.get(url)
    time.sleep(3)  # Wait for content to load
    extract_data()

last_page_url = "https://www.shl.com/solutions/products/product-catalog/?start=132&type=2&type=2"
driver.get(last_page_url)
time.sleep(5)  # Ensure full load
extract_data()
print(f"Total assessments extracted: {len(assessments)}")

# Close Selenium WebDriver
driver.quit()

# Save data to CSV
df = pd.DataFrame(assessments, columns=["Assessment Name", "URL", "Remote Support", "Adaptive Support", "Types"])
df.to_csv("11111shl_product_catalog_final.csv", index=False)

print(f"✅ Data extraction complete! {len(assessments)} items saved in 'shl_product_catalog_final.csv'.")

