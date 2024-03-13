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




url =  'https://www.youtube.com/watch?v=dtoUfUgrGB4&ab_channel=BleacherReport'


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


# We load comments for two minutes...
while True:


    if time.time() > timeout:
        break

    else:
        comments = driver.find_elements(By.XPATH, "//ytd-comments[@id='comments']/ytd-item-section-renderer[@id='sections']/div[@id='contents']")


        all_info_per_comment = comments[0].text.splitlines()

        time.sleep(5) # Give time for site to reload ..
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(5) # Give time for site to reload ...


# We clear the list of unneccessary informations
        
idx_to_delete = []

for i, ele in enumerate(all_info_per_comment):
    # We remove strings 'Antworten'

    if len(ele) == 0:
        idx_to_delete.append(i)
    else:

        if 'Antworten' in ele or 'Antwort' in ele:
            idx_to_delete.append(i)
        if 'Reply' in ele:
            idx_to_delete.append(i)
        
        # Try converting to integer, if possible, we can delete
        
        # Delete names
        if ele[0] == '@':
            idx_to_delete.append(i)
        
        # Delete Dates
        if ele[0:3] == 'vor':
            idx_to_delete.append(i)

        # Delete integers (likes)
        if (re.sub("\.", "", ele)).isdigit():
            idx_to_delete.append(i)


# We clean

idx_to_delete.sort(reverse=True)

for t in idx_to_delete:
    del all_info_per_comment[t]


    
# Now we only have the comments, i.e. the data we want!
    
# We add it into a dataframe

all_comments = {'comments' : all_info_per_comment}

all_comments_df = pd.DataFrame(all_comments)


all_comments_df.to_csv('/Users/marlon/VS-Code-Projects/Youtube/comments_example.csv')


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