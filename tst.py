from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def fetch_references(paper_title):
    # Start a browser session
    browser = webdriver.Chrome()

    # Navigate to Google Scholar
    browser.get('https://scholar.google.com/')

    # Enter the query and search
    search_box = browser.find_element(By.NAME, 'q')
    search_box.send_keys(paper_title)
    search_box.send_keys(Keys.RETURN)

    # Wait for the results page to load and extract the results.
    references = []

    try:
        while True:
            # Extract references
            for element in browser.find_elements_by_css_selector(".gs_rt a"):
                references.append(element.text)

            # Try to click the "Next" button
            try:
                next_button = browser.find_element_by_text_link("Next")
                next_button.click()

                # Wait to make sure the next page is loaded properly
                #time.sleep(3)
            except NoSuchElementException:
                break  # No more "Next" button, so we've reached the end of the results

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Close the browser session
        browser.quit()

    return references

# Example usage
paper_title_input = "Fast Segment Anything"
all_references = fetch_references(paper_title_input)
print(all_references)
