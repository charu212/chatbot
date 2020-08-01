from selenium import webdriver
from selenium.webdriver.common.keys import Keys


driver = webdriver.Chrome('C:\\Users\\Z\\Downloads\\chromedriver.exe')
driver.get('http://stackoverflow.com/')
driver.get('http://google.com')