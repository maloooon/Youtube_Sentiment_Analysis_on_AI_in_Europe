import os 
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd 
from datetime import date 
import re 



# initialize a web driver instance to control a Chrome window
# in headless mode
options = Options()
#options.add_argument('--headless=new')
options.add_argument('--disable-blink-features=AutomationControlled')
#options.add_argument('load_extension=adblock.crx')

driver = webdriver.Chrome(
    service=ChromeService(ChromeDriverManager().install()),
    options=options
)



def get_comment_count(driver):

    comment_count_element = driver.find_element(By.XPATH, "//ytd-comments[@id='comments']/ytd-item-section-renderer[@id='sections']/div[1]/ytd-comments-header-renderer/div[1]/div[1]/h2/yt-formatted-string/span[1]")
    comment_count_text = comment_count_element.text
    comment_count = int(comment_count_text.replace(",",""))
    return comment_count


def extract_comments(driver):
    comment_elements = driver.find_elements(By.CSS_SELECTOR, '#content-text')
    comments = [comment.text for comment in comment_elements]
    return comments


url =  'https://www.youtube.com/watch?v=G97ZJU44vTY&ab_channel=ARTEde'


driver.get(url)

try:
    # wait up to 15 seconds for the consent dialog to show up
    consent_overlay = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.ID, 'dialog'))
    )

    # select the consent option buttons
    consent_buttons = consent_overlay.find_elements(By.CSS_SELECTOR, '.eom-buttons button.yt-spec-button-shape-next')
    if len(consent_buttons) > 1:
        # retrieve and click the 'Accept all' button
        accept_all_button = consent_buttons[1]
        accept_all_button.click()
except TimeoutException:
    print('Cookie modal missing')



#driver.execute_script("window.scrollTo(0,200)")

time.sleep(5) # Give time for site to reload ...



names = []
time_of_comment = []
text = []
likes = []
number_of_answers = []

timeout = time.time() + 60*2   # 2 minutes from now


# Get an initial count on comments
initial_comment_count = get_comment_count(driver)
print("INITIAL : {}".format(initial_comment_count))
comments = []

# We load comments for two minutes...
while True:


    if len(comments) == initial_comment_count:
        # Break if we have no comments left anymore
        break

    else: 
  
        comments = extract_comments(driver)
       

        time.sleep(5) # Give time for site to reload ..
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(5) # Give time for site to reload ...

        print("CURRENTLY WE HAVE {} COMMENTS LOADED".format(len(comments)))
    

    
# We add it into a dataframe

all_comments = {'comments' : comments}

all_comments_df = pd.DataFrame(all_comments)


all_comments_df.to_csv('/Users/marlon/VS-Code-Projects/Youtube/comments_0.csv')


"""
range_for_names = list(range(0,len(all_info_per_comment),6))
range_for_time_of_comment = list(range(1,len(all_info_per_comment),6))
range_for_text = list(range(2,len(all_info_per_comment),6))
range_for_likes = list(range(3,len(all_info_per_comment),6))
range_for_number_of_answers = list(range(5,len(all_info_per_comment),6))


idx = 0

# Then we add all the information given 
for _,info in enumerate(all_info_per_comment):
    if idx in range_for_names:
        names.append(info)
    if idx in range_for_time_of_comment:
        time_of_comment.append(info)
    if idx in range_for_text:
        text.append(info)
    if idx in range_for_likes: # FIX : sometime double comments because of new lines --> leads to adding comment text here and fucking up the structure
        try:
            info = re.sub("\.", "", info)
            likes.append(int(info))
        except ValueError:
            text.append(info)
            idx -= 1
    if idx in range_for_number_of_answers:
        number_of_answers.append(info)
    
    idx += 1
"""
    

driver.close()