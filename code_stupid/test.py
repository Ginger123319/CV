# import time
# from selenium import webdriver

# option = webdriver.ChromeOptions()
# #隐藏窗口
# option.add_argument('headless')
# #防止打印一些无用的日志
# option.add_experimental_option("excludeSwitches",['enable-automation','enable-logging'])
# driver = webdriver.Chrome(options=option)

# driver.maximize_window()
# driver.implicitly_wait(8)

# driver.get("http://www.baidu.com")
# print (driver.title)

# driver.find_element_by_xpath("//*[@id='kw']").send_keys("selenium")
# driver.find_element_by_xpath("//*[@id='su']").click()

# res_string=driver.find_element_by_xpath("//*[@id='1']/h3/a/em").text
# print(res_string)
# if(res_string == "selenium"):
    # print ("ok!")

# driver.quit()

# print("ex13---------------------------------------------------------------")
# from sys import argv
# script, first, second, third=argv
# print("The script is called:", script)
# print("Your first variable is:", first)
# print("Your second variable is:", second)
# print("Your third variable is:", third)



print("ex15---------------------------------------------------------------")
from sys import argv

script, filename=argv
txt=open(filename,encoding='utf-16')

print(txt)
 #open()是返回一个文件对象，可以通过read()函数读取到该文件的内容
 #<_io.TextIOWrapper name='ex15_sample.txt' mode='r' encoding='cp936'>
print(f"Here's your file {filename}:")
print(txt.read())
# print(txt.close())
 #None

# print("Type the filename again:")
# file_agian=input('>')

# txt_again=open(file_agian)
# print(txt_again.read())
# print(txt_again.close())
 #文件操作之打开、读取与关闭
