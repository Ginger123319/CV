#笨办法学python练习题合集

print("ex4---------------------------------------------------------------")
cars=100
space_in_a_car=4.0
drivers=30
passengers=90
cars_not_driven=cars - drivers
cars_driver=drivers
carpool_capacity=cars_driver * space_in_a_car
average_passengers_per_car=passengers / cars_driver

print("There are", cars,"cars available.")
print("There are only", drivers,"drivers available.")
print("There will be", cars_not_driven,"empty cars today.")
print("We can transport", carpool_capacity,"people today.")
print("We have", passengers,"to carpool today.")
print("We need to put about", average_passengers_per_car,"in each car.")
#浮点数参与运算，结果也是浮点数；print括号中变量可以直接通过变量名打印出来

print("ex5---------------------------------------------------------------")
my_name='Zed A. Shaw'
my_age=35
my_height= 74
my_eyes='Black'
my_teeth='White'
my_hair='Black'
my_weight=180

#print(f"Let's talk about {my_name}.")
print("Let's talk about",my_name)
total=my_age+(my_height * 2.54)+(my_weight * 0.454)
print(f"if I add {my_age}, {my_height * 2.54 }, and {my_weight * 0.454} I get {total}.")
#print(f"var is {var}")--按照格式打印变量

print("ex6---------------------------------------------------------------")
types_of_people=10
x=f"There are {types_of_people} types of people."
binary="binary"
do_not="don't"
y=f"Those who knows {binary} and those who {do_not}."

print(x)
print(y)

print(f"I said: {x}")
print(f"I also said '{y}'")

hilarious=False
joke_evalution="Isn't that joke so funny?!{}"

print(joke_evalution.format(hilarious))

w="This is the left side of ..."
e="a string with a right side."

print(w + e)
#把字符串放到另一个字符串中指定的位置上{}；
#使用.format()可以传入一个字符串，{}花括号为参数插入位置的标识

print("ex7---------------------------------------------------------------")
print("Mary had a little lamb.")
print("Its fleece was white as {}.".format(' snow'))
print("And everwhere that Mary went")
print("." * 10)
end1="C"
end2="h"
end3="e"
end4="e"
end5="s"
end6="e"
end7="B"
end8="u"
end9="g"

print(end1 + end2 + end3 + end4 + end5 + end6, end=' ')
print(end7+end8+end9)
#万物皆对象，字符串对象可以直接调用format函数，常见的参数就是字符串
#print(, end=' ')参数end，可以指定打印末尾的字符，默认为换行符
#不过一般单引号会被用来创建简短的字符串，如′a′、′snow′等

print("ex8---------------------------------------------------------------")
formatter="{} {} {} {}"
print(formatter.format(1,2,3,4))
print(formatter.format("one", "two", "three", "four"))
print(formatter.format(True, False, False, True))
print(formatter.format(formatter, formatter, formatter, formatter))
print(formatter.format("Try your",
						"Own text here",
						"Maybe a poem",
						"Or a song about fear"))
#还是format函数的使用

print("ex9---------------------------------------------------------------")
days="Mon Tue Wed Thu Fri Sat Sun"
months="\nJan\nFeb\nMar\nApr\nMay\nJun\nJul\nAug"

print("Here are the days: ", days)
print("Here are the months: ", months)

print("""
There's something going on here.
With the three double-quotes.
We'll be able to type as much as we like.
Even 4 lines if we want, or 5, or 6.
""")
#三引号之间可以放任意数目行数的字符串
#\反斜杠也叫转义字符，放在他后面的字符都会转变一个意思或者作用

print("ex10---------------------------------------------------------------")
tabby_cat="\tI'm tabbed in."
persian_cat="I'm split \non a line."
#此处就是要打印\a与\两个特殊字符串，不转义无法打印出来
backslash_cat="I'm  \\a \\ cat."

fat_cat="""
I'll do a list:
\t * Cat food
\t * Fishies
\t * Catnip\n\t * Grass
"""						
print(tabby_cat)
print(persian_cat)
print(backslash_cat)
print(fat_cat)						
#\t制表符（tab）

print("ex11---------------------------------------------------------------")
# print("How old are you?", end=' ')
# age=input()
# print("How tall are you?", end=' ')
# height=input()
# print("How much do you weigh?", end=' ')
# weight=input()
# print(f"So, you're {age} old, {height} tall and {weight} heavy.")
 ##input()函数，获取一个键盘输入
 ##在python3中对input和raw_input函数进行了整合，仅保留了input函数（认为raw_input函数是冗余的）。
 ##同时改变了input的用法——将所有的输入按照字符串进行处理，并返回一个字符串。



print("ex12---------------------------------------------------------------")
# age=input("How old are you?")
# height=input("How tall are you?")
# weight=input("How much do you weigh?")

# print(f"So, you're {age} old, {height} tall and {weight} heavy.")
 # #input()函数括号中可以添加字符串提示符，打印在输入界面上

# print("ex13---------------------------------------------------------------")
# from sys import argv
# script, first, second, third=argv
# print("The script is called:", script)
# print("Your first variable is:", first)
# print("Your second variable is:", second)
# print("Your third variable is:", third)
 # #其第一个元素是程序本身，随后才依次是外部给予的参数
 # #sys.argv[ ]其实就是一个列表，里边的项为用户输入的参数
 # #argv即所谓的参数变量（argument variable）
 # #解包（unpack）:把argv中的东西取出，解包，将所有的参数依次赋值给左边的这些变量
 # #import 之后的名字叫模块，此处即导入了sys模块
 # #使用自动化脚本时传参应该使用argv的方式而不是用input()函数

print("ex14---------------------------------------------------------------")
# from sys import argv

# script, user_name=argv
# prompt='>'

# print(f"Hi {user_name}, I'm the {script} script.")
# print("I'd like to ask you a few questions.")
# print(f"Do you like me {user_name}?")
# likes=input(prompt)

# print(f"Where do you live {user_name}?")
# lives=input(prompt)

# print("What kind of computer do you have?")
# computer=input(prompt)

# print(f'''
# Alright, so you said {likes} about liking me.
# You live in {lives}. Not sure where that is.
# And you have a {computer} computer. Nice.
# ''')
 # #在涉及变量的时候要考虑其使用的频率，频率高的应当设置为全局变量，也方便修改
 # #重复使用的次数多了，就要考虑是否需要抽出来做一个变量或者单独写个函数
 
print("ex15---------------------------------------------------------------")
# from sys import argv

# script, filename=argv
# txt=open(filename)

# print(txt)
 # #open()是返回一个文件对象，可以通过read()函数读取到该文件的内容
 # #<_io.TextIOWrapper name='ex15_sample.txt' mode='r' encoding='cp936'>
# print(f"Here's your file {filename}:")
# print(txt.read())
# print(txt.close())
 # #None

# print("Type the filename again:")
# file_agian=input('>')

# txt_again=open(file_agian)
# print(txt_again.read())
# print(txt_again.close())
 # #文件操作之打开、读取与关闭

print("ex16---------------------------------------------------------------")
 # #readline读一行；truncate清空文件；write('stuff')将stuff写入文件
 # #seek(0)将读写位置移动到文件开头
# from sys import argv

# script, filename=argv

# print(f"We're going to erase {filename}.")
# print("If you don't want that, hit CTRL-C(^C).")
# print("If you do want that, hit RETRN.")

# input("?")

# print("Opening the file...")
# target=open(filename, 'w')
# print(target)
 # #<_io.TextIOWrapper name='.\\ex15_sample.txt' mode='w' encoding='cp936'>

# print("Truncating the file. Goodbye!")
 # #target.truncate()
 # #默认是以r模式打开文件，如果需要w权限需要在open()函数中指定值
 # #以w权限打开的文件在写文件之前就不需要再调用清空的函数了
 
 # #因为这样写文件会把源文件的内容替换掉，相当于先清空再写入，效果一致

# print("Now I'm going to ask you for three lines.")

# line1=input("line 1:")
# line2=input("line 2:")
# line3=input("line 3:")

# print("I'm going to write these to the file.")

# target.write(line1)
# target.write("\n")
# target.write(line2)
# target.write("\n")
# target.write(line3)
# target.write("\n")

# print("And finally, we close it.")
# target.close()
 # #文件打开open(filename, '?')函数的?处可指定r，w，a（追加写）模式；可以+连接
print("ex17---------------------------------------------------------------")
# from sys import argv
# from os.path import exists

# script, from_file, to_file=argv

# print(f"Copying from {from_file} to {to_file}")

# in_file=open(from_file)
# indata=in_file.read()
 # #<class 'str'>
# print(type(indata))
 
# print(f"The input file is {len(indata)} bytes long")

# print(f"Does the output file exist?{exists(to_file)}")
# print("Ready, hit RETURN to continue, CTRL-C to abort.")
# input()

# out_file=open(to_file, 'w')
# out_file.write(indata)

# print("Alright, all done.")

# out_file.close()
# in_file.close()
 # #命令exists。这个命令将文件名字符串作为参数，
 # #如果文件存在的话，它将返回True；否则将返回False
 # #len(str)

print("ex18---------------------------------------------------------------")
 #this one is like your script with argv
def print_two(*args):
	arg1, arg2=args
	print(f"arg1:{arg1}, arg2:{arg2}")
 #ok, that *args is actually pointless, we can just do this
def print_two_again(arg1,arg2):
	print(f"arg1:{arg1}, arg2:{arg2}")

print_two("Zed","Shaw")
print_two_again("Z","S")
 #注意def前面不能有任何空格或者tab，否则缩进错误
 #def->define
 
print("ex19---------------------------------------------------------------")
def cheese_and_crackers(cheese_count,boxes_of_crackers):
	print(f"You have {cheese_count} cheeses!")
	print(f"You have {boxes_of_crackers} boxes of crackers!")
	print("Man that's enough for a party!")
	print("Get a blanket.\n")
	
print("We can just give the function numbers directly:")
cheese_and_crackers(20,30)

print("Or, we can use variables from our script:")

amount_of_cheese=10
amount_of_crackers=50

cheese_and_crackers(amount_of_cheese,amount_of_crackers)

print("We can even do math inside too:")
cheese_and_crackers(10+20,5+6)

print("And we can combine the two, variables and math:")
cheese_and_crackers(amount_of_cheese+100,amount_of_crackers+100)

print("ex20---------------------------------------------------------------")
from sys import argv

# script, input_file=argv

def print_all(f):
	print(f.read())
	
def rewind(f):
	f.seek(0)

def print_a_line(line_count, f):
	print(line_count, f.readline())
	#,并不会打印出来，此处仅作分隔符

# current_file=open(input_file)

# print("First let's print the whole file:\n")

# print_all(current_file)

# print("Now let's rewind, kind of like a tape.")

# rewind(current_file)

# print("Let's print three lines:")

# current_line=1
# print_a_line(current_line, current_file)

# current_line=current_line+1
# print_a_line(current_line, current_file)

# current_line=current_line+1
# print_a_line(current_line, current_file)
 # #位置问题：
 # #readline()是怎么知道每一行在哪里的？
 # #该函数会扫描文件中的每一个字节，直到找到\n为止；
 # #然后就停止读取，文件f会记录每次调用函数后的读取位置
 # #下一次调用该函数时就能读取接下来的一行（从记录位置开始读取）
 # #空行问题：
 # #readline()返回值中本就包含\n，所以加上print默认以\n结尾
 # #那么就出现了空行，处理方法就是print()，添加参数end=""
 
print("ex21---------------------------------------------------------------")
def add(a,b):
	print(f"ADDING {a} + {b}")
	return a+b

def subtract(a,b):
	print(f"SUBSTRACTING {a} - {b}")
	return a-b
	
def multiply(a,b):
	print(f"MULTIPLYING {a} * {b}")
	return a*b
	
def divide(a,b):
	print(f"DIVIDING {a} / {b}")
	return a / b
	
print("Let's do some math with just function")

age=add(30,5)
height=subtract(78,4)
weight=multiply(99,1.5)
iq=divide(100,0.5)

print(f"Age:{age}, Height:{height}, Weight:{weight}, IQ:{iq}")

#A puzzle for the extra credit, type it in a new way.
print("Here is a puzzle.")

what=add(age, subtract(height, multiply(weight, divide(iq, 2))))

print("That becomes: ", what, "Can you do it by hand?")
print(add(24, subtract(divide(34,100),1023)))
 #运算符的优先级
 
print("ex23---------------------------------------------------------------")
# import sys
# script, encoding, error=sys.argv

# def main(language_file, encoding, errors):
	# line=language_file.readline()
	# if line:
		# print_line(line, encoding, errors)
		 # #递归调用，目的：逐行读出文件中的每一行内容
		# return main(language_file, encoding, errors)
		
# def print_line(line, encoding, errors):
	 # #删掉每一行行末的\n
	# next_lang=line.strip()
	
	 # #DBES：decode bytes,encode strings;
	 # #字节串解码：字节串解码为字符串；字符串编码：字符串编码后转为字节串
	# raw_bytes=next_lang.encode(encoding, errors=errors)
	# cooked_string=raw_bytes.decode(encoding, errors=errors)
	
	# print(raw_bytes, "<===>", cooked_string)
	
# languages=open("languages.txt",encoding="utf-8")

# main(languages, encoding, error)
 # #计算机根本上是一个开关阵列（矩阵？）；用电流来触发这些开关开启或关闭
 # #用0（关）和1（开）：1表示有电、开启、接通；而0刚好相反
 # #计算机中将0和1称为“位”（bit）
 # #编码是大家认同的位序列转换为数字的转换标准；比如00000000表示0；11111111表示255
 # #1字节（byte）=8（bit）
 # #常见的编码有ASCII：这个标准将数字和字母互相对应，如90表示Z，用位表示即01011010
 # #计算机会根据存储的ASCII表做对应的转换工作
 # #为了统一编码：有了Unicode编码；工作原理与ASCII编码类似，不过他的表格更大
 # #可以用32位编码一个Unicode字符；而ASCII码只能用8位编码一个；
 # #这样可以存储任意语言的字符，由于都用32位去编码一个字符太过浪费
 # #有了一个约定俗称，默认用8位即一个字节去编码一个字符，需要的时候再用16或32位
 # #该约定即UTF-8（Unicode Transformation Format 8 bits）
 
 # #额外挑战：使用b′′字节串取代UTF-8字符串重写代码，结果就是把程序反写一遍
 # #难点在于不能直接encode，结果不符合预期
 # #特殊处理就是分割('\\x')，分割后再转10进制存入数组，再把数组转为字节串
print("ex_extra23---------------------------------------------------------------")
# import sys
# script, encoding, error=sys.argv

# def main(language_file, encoding, errors):
	# line=language_file.readline()
	# strTobytes=[]
	# #处理包含字节串的字符串，以'\x'作为分隔符，此处'\\x'的第一个\表示转义
	# for i in line.split('\\x'):
		# if i !=  '':
			# #将分割后的字符串转换为integer对象，以十进制格式展示
			# #16进制转为10进制
			# num=int(i,16)
			# print(num)
			# #转换后的数放到数组中
			# strTobytes.append(num)
	# #integer数组转换为字节串的形式展示
	# #字节串：b'\xe6\x96\x87\xe8\xa8\x80'
	# a=bytes(strTobytes)
	# b=a.decode()
	# line=b
	# if line:
		# print_line(line, encoding, errors)
		# #递归调用，目的：逐行读出文件中的每一行内容
		# return main(language_file, encoding, errors)
		
# def print_line(line, encoding, errors):
	# #删掉每一行行末读到的\n，注意print语句默认自带一个\n
	# next_lang=line.strip()
	
	# #DBES：decode bytes,encode strings;
	# #字节串解码：字节串解码为字符串；字符串编码：字符串编码后转为字节串
	# raw_bytes=next_lang.encode(encoding, errors=errors)
	# cooked_string=raw_bytes.decode(encoding, errors=errors)
	
	# print(raw_bytes, "<===>", cooked_string)
	
# languages=open("bytes.txt",encoding="utf-8")

# main(languages, encoding, error)

print("ex24---------------------------------------------------------------")
print('You\' d need to kown \' bout escapes with \\ that do:')
#转义字符\的使用
#参数列表在格式化时前面需要加上*
#print("{}{}{}".format(*argv))

print("ex25---------------------------------------------------------------")
def break_words(stuff):
	words=stuff.split(' ')
	return words
	
def sort_words(words):
	return sorted(words)
	
def print_first_word(words):
	word=words.pop(0)
	print(word)
	
def print_last_word(words):
	word=words.pop(-1)
	print(word)

def sort_sentence(sentence):
	words=break_words(sentence)
	return sort_words(words)
	
def print_first_and_last(sentence):
	words=break_words(sentence)
	print_first_word(words)
	print_last_word(words)
	
def print_first_and_last_sorted(sentence):
	words=sort_sentence(sentence)
	print_first_word(words)
	print_last_word(words)
	
#配置电脑环境变量，所以可以找到放在任何位置的python文件
#并且可以在python命令行中导入该py文件，在其中调用对应的函数
#import xxx:在调用函数之前要加上xxx作为函数前缀即xxx.function1(argv)
#而from xxx import *就不用带前缀，相当于直接把该文件粘贴过去，费空间
#前者节约空间，调用方式复杂；后者占用空间大，调用方式简单
#列表的pop操作，弹出列表，会改变列表

print("ex26---------------------------------------------------------------")
#改错
# print("How old are you?", end=' ')
# age = input()
# print("How tall are you?", end=' ')
# print("How much do you weigh?", end=' ')
# weight = input()

# print(f"So, you're {age} old, {height} tall and {weight} heavy.")

# script, filename = argv

# txt = open(filename)

# print("Here's your file {filename}:")
# print(txt.read())

# print("Type the filename again:")
# file_again = input("> ")

# txt_again = open(file_again)

# print(txt_again.read())


# print('Let\'s practice everything.')
# print('You\'d need to know \'bout escapes with \\ that do \n newlines and \t tabs.')

# poem = """
# \tThe lovely world
# with logic so firmly planted
# cannot discern \n the needs of love
# nor comprehend passion from intuition
# and requires an explanation
# \n\t\twhere there is none.
# """

# print("--------------")
# print(poem)
# print("--------------")


# five = 10 - 2 + 3 - 6
# print(f"This should be five: {five}")

# def secret_formula(started):
    # jelly_beans = started * 500
    # jars = jelly_beans / 1000
    # crates = jars + 100
    # return jelly_beans, jars, crates


# start_point = 10000
# beans, jars, crates = secret_formula(start_point)

# # remember that this is another way to format a string
# print("With a starting point of: {}".format(start_point))
# # it's just like with an f"" string
# print(f"We'd have {beans} beans, {jars} jars, and {crates} crates.")

# start_point = start_point / 10

# print("We can also do that this way:")
# formula = secret_formula(start_point)
# # this is an easy way to apply a list to a format string
# print("We'd have {} beans, {} jars, and {} crates.".format(*formula))



# people = 20
# cats = 30
# dogs = 15


# if people < cats:
    # print ("Too many cats! The world is doomed!")

# if people < cats:
    # print("Not many cats! The world is saved!")

# if people < dogs:
    # print("The world is drooled on!")

# if people > dogs:
    # print("The world is dry!")


# dogs += 5

# if people >= dogs:
    # print("People are greater than or equal to dogs.")

# if people <= dogs:
    # print("People are less than or equal to dogs.")


# if people == dogs:
    # print("People are dogs.")

	
print("ex28---------------------------------------------------------------")
print(True and False)
print(True or "test")
print("test" or True)
print(False or "test" or "b")
print("test" and False or "a")
print("a" or "b")
print("b" and "a" or "c")
#简单来说就是把任何字符都当成True来加以判断，包括1,2,3以及"a","b","c"等等
#都是给布尔表达式返回两个被操作对象中的一个，不一定是True或False

#不再往后推算的这种现象就叫短路逻辑
#何以False开头的and语句都会直接处理成False，不会继续检查后面的语句
#False or 继续往下推算，如果还没有形成短路逻辑就打印末尾的值
#任何包含True的or语句，只要处理到True，就不会继续向下推算，直接返回True
#True and 继续往下推算，如果还没有形成短路逻辑就打印末尾的值

print("ex29---------------------------------------------------------------")
#为什么if语句的下一行需要4个空格的缩进？
#行尾的冒号的作用是告诉Python接下来你要创建一个新的代码块
#缩进告诉Python这些代码处于该代码块中
#这跟你前面创建函数时的冒号是一个道理

#如果多个elif块都是True, Python会如何处理？
#Python只会运行它遇到的是True的第一个块，所以只有第一个为True的块会运行

print("ex32---------------------------------------------------------------")
the_count=[1,2,3,4,5]
fruits=['apples','oranges','pears','apricots']
change=[1,'pennies',2,'dimes',3,'quarters']

for number in the_count:
	print(f"this is count {number}")
	
#notice we have to use {} since we don't know what's in it
for i in change:
	print(f"I got {i}")
	
print("ex35---------------------------------------------------------------")
from sys import exit

# def gold_room():
	# print("This room is full of gold. How much do you take?")
	# #限定只能输入数字--try---except--语句
	# while True:
		# try:
			# x = input('Input an integer: ')
			# how_much = int(x)
			# break
		# except ValueError:
			# print ('Please input an *integer*')
		
	# if how_much < 50:
		# print("Nice, you're not greedy, you win!")
		# exit(0)
	# else:
		# dead("You greedy bastard!")
	
# def bear_room():
	# print("There is a bear here.")
	# print("The bear has a bunch of honey.")
	# print("The fat bear is in front of another door.")
	# print("How are you going to move the bear?")
	# bear_moved=False
	
	# while True:
		# choice=input(">")
		# if choice == "take honey":
			# dead("The bear looks at you then slaps your face off.")
		# elif choice == "taunt bear" and not bear_moved:
			# print("The bear has moved from the door.")
			# print("You can go through it now.")
			# bear_moved=True
		# elif choice == "taunt bear" and bear_moved:
			# dead("The bear gets pissed off and chews your leg off.")
		# elif choice == "open door" and bear_moved:	
			# gold_room()
		# else:
			# print("I go no idea what that means.")
			
# def cthulhu_room():
	# print("Here you see the great evil Cthulhu.")
	# print("He, it, whatever stares at you and you go insane.")
	# print("Do you flee for your life or eat your head?")
	
	# choice=input(">")
	
	# if "flee" in choice:
		# start()
	# elif "head" in choice:
		# dead("Well that was tasty!")
	# else:
		# cthulhu_room()
		
# def dead(why):
	# print(why, "Good job!")
	# exit(0)

# def start():
	# print("You are in a dark room.")
	# print("There is a door to your right and left.")
	# print("Which one do you take?")
	
	# choice=input(">")
	
	# if choice == "left":
		# bear_room()
	# elif choice == "right":
		# cthulhu_room()
	# else:
		# dead("You stumble around the room until you starve.")

# start()

#调试程序的最好的方法是
#使用print在各个想要检查的关键点将变量打印出来
#从而检查那里是否有错

#让程序一部分一部分地运行起来
#不要等写了一大堆代码文件后才去运行它们
#写一点儿，运行一点儿，再修改一点儿

#1．在纸上或者索引卡上列出你要完成的任务。这就是你的待办任务。
#2．从中挑出最简单的任务。
#3．在源代码文件中写下注释，作为你完成任务代码的指南。
#4．在注释下面写一些代码。
#5．快速运行你的代码，看它是否工作。
#6．循环“写代码，运行代码进行测试，修正代码”的过程。
#7．从任务列表中划掉这条任务，挑出下一个最简单的任务，重复上述步骤。
#8．随时更新任务列表，添加新的任务，删除不必要的任务。

#列任务表，取最简单的任务，用中文写实现思路，
#根据思路转换成代码，跑一下，测一下，修改一下
#进入下一个简单任务，循环进行以上步骤

#关键字：
print(True and False == False)

#w,r,wt,rt都是python里面文件操作的模式。
#w是写模式，r是读模式。
#t是windows平台特有的所谓text mode(文本模式）
#区别在于会自动识别windows平台的换行符。

#类Unix平台的换行符是\n，
#而windows平台用的是\r\n两个ASCII字符来表示换行，
#python内部采用的是\n来表示换行符。
#rt模式下，python在读取文本时会自动把\r\n转换成\n

with open("1.txt","wt") as out_file:
	out_file.write("该文本会写入文件中\n看到我了吧！")

with open("1.txt", "rt") as in_file:
	text=in_file.read()

print(text)


try:
	assert False,'Error!'
	# 如果条件不成立，则打印出 'Error!' 并抛出AssertionError异常
	#此处使用try--except--语句避免抛出异常，自然也不会打印'Error!'
except AssertionError:
	print("assert 条件不成立！")


while True:
	print("break")
	break
	
print("'continue' is different from 'break' in the loop statement.")

print(1 is 1 == True)

print("lambda创建短的匿名函数！")
s=lambda y:y**y;
print(s(3))


def empty():
	pass

print("pass 代表空代码块！")
empty()

#raise ValueError("No!")

#在 Python 中，使用了 yield 的函数被称为生成器（generator）
#更简单点理解生成器就是一个迭代器。
#在调用生成器运行的过程中，
#每次遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值。
#并在下一次执行 next() 方法时从当前位置继续运行。

#以下实例使用 yield 实现斐波那契数列：
import sys

# def fibonacci(n):
	# a,b,counter = 0,1,0
	# while True:
		# if(counter > n):
			# return
		# yield a
		
		# a,b = b,a+b
		# counter += 1
# f=fibonacci(10)
# print(type(f))
# # <class 'generator'> 返回的是一个迭代器对象，可以用 next()函数 进行迭代

# while True:
	# try:
		# print(next(f), end = " ")
	# except StopIteration:
		# #print("打印结束")
		# #如果不写退出的话会一直循环打印"打印结束"的字符串
		# sys.exit()
		
print("Data Type:")
print("bytes 字节串可以存储文本图片或者文件")
print(b'hello')
print('hello')

print("字符串转义序列：")
print("\\")
print("\'")
print("\"")
print("\n换行符")
print("\t制表符")
print("\r回车",end = "    ")#回车并没有换行的作用
print("回车")
print("\v垂直制表符")

# print(3//4)
# print(3%4)
# a=5
# a+=4
# print(a)
# a-=3
# print(a)

if(1 is 1):
	print("'if' is without 'else'")

print("ex38---------------------------------------------------------------")
ten_things = "Apples Oranges Crows Telephone Light Sugar"

print("Wait there are not 10 things in that list. Let's fix that.")

stuff = ten_things.split(' ')
more_stuff = ["Day", "Night", "Song", "Frisbee", "Corn", "Banana", "Girl", "Boy"]

while len(stuff) != 10:
	next_one = more_stuff.pop()
	print("Adding:", next_one)
	stuff.append(next_one)
	print(f"There are {len(stuff)} items now.")
	
print("There we go:", stuff)

print("Let's do some things with stuff.")

print(stuff[1])
print(stuff[-1])
print(stuff.pop())
print(' '.join(stuff))
print('#'.join(stuff[3:5]))

#实际在Python内部调用的情况append(mystuff, ′hello′)
#不过你看到的只是mystuff.append (′hello′)。
#特别是在定义类中的成员函数的时候，
#一定要this作为参数，this即该类的一个对象？？？

#数据结构的目的就是组织数据，是存储数据的一种方式，将数据结构化
#每个编程概念都与现实世界中的某个东西有关，能找到类比的对象有助于理解它们

#什么时候使用了列表：
#维持次序，指的是列表按照内容进行排列顺序
#通过索引随机访问内容
#需要线性访问，从头到尾的内容依次读取出来。for循环

print("ex39-----------------字典类型--------------------------------------")
print("字典的关键理念就是映射")
states = {
        'Oregon' : 'OR' ,
        'Florida' : 'FL' ,
        'California' : 'CA' ,
        'New York' : 'NY'  ,
        'Michigan' : 'MI'
}

cities = {
        'CA' : 'San Francisco' ,
        'MI' : 'Detroit' ,
        'FL' : 'Jacksonville'
}

cities['NY'] = 'New York'
cities['OR'] = 'Portland'

print('-' * 10)
print("NY State has: ", cities['NY'])
print("OR State has: ", cities['OR'])

print('-' * 10)
print(" ",states['Michigan'])
print(" ",states['Florida'])

print('--dict--' * 3)
print(" ",cities[states['Michigan']])
print(" ",cities[states['Florida']])

print("--access the dict by using for-loop statement--")
for state, abbrev in list(states.items()):
        print(f"{state} is abbreviated {abbrev}")

for abbrev, city in list(cities.items()):
        print(f"{abbrev} has the city {city}")

for state, abbrev in list(states.items()):
        print(f"{state} is abbreviated {abbrev}")
        print(f"and has the city {cities[abbrev]}")

state = states.get('Texas')
print(state)
#get()方法如果没有获取到该索引的话会返回None而不是抛出异常
if not state:
                print("Sorry, no Texas.")

#get a city with a default value
#如果没有获取到索引的话，可以设置一个默认值，此处为'Does Not Exist'
city = cities.get('TX', 'Does Not Exist')
print(f"The city for the state 'TX' is: {city}")                
        
#字典能用在哪里？
#各种需要通过某个值去查看另一个值的场合。
#事实上，你也可以把字典叫“查找表”。需要表格表示数据结构时使用

#对象OOP
#字典vs模块(import)
#1．拿一个类似“键=值”风格的容器；
#2．通过“键”的名称获取其中的“值”

print("ex40-----------------类的定义--------------------------------------")

class Song(object):

#有了self.cheese= ′Frank′就清楚地知道这指的是实例的属性self.cheese
#不加self参数有歧义，它指的可能是实例的cheese属性，
#也可能是一个叫cheese的局部变量        
        def __init__(self, lyrics):
                self.lyrics = lyrics
                
        def sing_me_a_song(self):
                for line in self.lyrics:
                        print(line)

#注意以下几行代码要与class对齐，避免缩进有误引发的异常                        
happy_bday = Song(["Happy", "Clap", "Alone"])
        
bulls_on_parade = Song(["Brings", "You", "Down"])
        
happy_bday.sing_me_a_song()
        
bulls_on_parade.sing_me_a_song()

#“object oriented programming”（面向对象编程）

print("ex41-----------------面向对象--------------------------------------")
#继承（父子）和组合（车子和车轮）
#is a和has a
#object可以写也可以不写，显示优于隐式
class Animal(object):
	pass
	
class Dog(Animal):
	
	def __init__(self, name):
		self.name = name

class Person(object):
	
	def __init__(self, name):
		self.name = name
		self.pet = None

class Employee(Person):
	
	def __init__(self, name, salary):
		#沿用父类的构造函数
		super(Employee, self).__init__(name)
		self.salary = salary

satan = Dog("Satan")

mary = Person("Mary")
mary.pet = satan

frank = Employee("Frank", 120000)
#继承父类的属性pet
frank.pet = satan

print(mary.pet.name)
print(frank.pet.name)

print("基本的面向对象分析和设计----------------------------------------")
#自顶向下：从抽象概念入手逐渐细化直到变成概念具体可以用代码实现的东西
#提出问题-寻找相关概念-结构关系图-名词和动词-找关系生成类属性函数的名称列表
#相似概念放一块-给他们找个父类-画出一个简单的树状图-把动词名词放在对应位置
#根据这个树状图就可以写骨架代码-只包含类与成员变量以及成员函数-小测一下
#细化代码-逐步优化-使得代码更加贴合需要解决的问题

from sys import exit
from random import randint
from textwrap import dedent

class Scene(object):
	
	def enter(self):
		print("This scene is not yet configured.")
		print("Subclass it and implement enter().")
		exit(1)

class Engine(object):
	
	def __init__(self, scene_map):
		self.scene_map = scene_map
	
	def play(self):
		current_scene = self.scene_map.opening_scene()
		last_scene = self.scene_map.next_scene('finished')
		
		while current_scene != last_scene:
			next_scene_name = current_scene.enter()
			print(f"name:{next_scene_name}")
			current_scene = self.scene_map.next_scene(next_scene_name)
			
			current_scene.enter()


class Death(Scene):
	#类变量，详情见https://www.cnblogs.com/20150705-yilushangyouni-Jacksu/p/6238187.html
	quips = [
		"You died.1",
		"You died.2",
		"You died.3",
		"You died.4",
		"You died.5"
	]
	def enter(self):
		print(Death.quips[randint(0, len(self.quips)-1)])
		exit(1)
		
class CentralCorridor(Scene):
	
	def enter(self):
		print(dedent("""
			game description
		"""))
		action = input(">")
		
		if action == "shoot!":
			print(dedent("""
				it eats you!
			"""))
			return 'death'
		elif action == "dodge!":
			print(dedent("""
				it eat your head!
			"""))
			return 'death'
			
		elif action == "tell a joke":
			print(dedent("""
				it laughs you!
			"""))
			return 'laser_weapon_armory'
		
		else:
			print("DOES NOT COMPUTE!")
			return 'centrral_corridor'
		
class LaserWeaponArmory(Scene):
	
	def enter(self):
		print(dedent("""
			The code is 3 digits.
		"""))
	
		code = f"{randint(1,9)}{randint(1,9)}{randint(1,9)}"
		guess = input("[keypad]>")
		guesses = 0
	
		while guess != code and guesses < 10:
			print("BZZZZEDDD!")
			guesses += 1
			guess = input("[keypad]>")
		
			if guess == code:
				print(dedent("""
					to the bridge
				"""))
				return 'the_bridge'
			else:
				print(dedent("""
					locking!
				"""))
				return 'death'

class TheBridge(Scene):
	
	def enter(self):
		print(dedent("""
			bomb under your arm
		"""))
		
		action = input(">")
		
		if action == "throw the bomb":
			print(dedent("""
				bomb up when it goes off.
			"""))
			return 'death'
			
		elif action == "slowly place the bomb":
			print(dedent("""
				run to the escape pod.
			"""))
			return 'escape_pod'
		
		else:
			print("DOES NOT COMPUTE!")
			return 'the_bridge'
			
				
class EscapePod(Scene):
	
	def enter(self):
		print(dedent("""
			5 pods,which one do you take?
		"""))
		
		good_pod = randint(1,5)
		guess = input("[pod#]>")
		
		if int(guess) != good_pod:
			print(dedent("""
				crushing your body into jam jelly.
			"""))
			return 'death'
		else:
			print(dedent("""
				you won!
			"""))
			return 'finished'
			
class Finished(Scene):
	
	def enter(self):
		print("You won! Good job.")
		return 'finished'
		
class Map(object):
	
	scenes = {
		'centrral_corridor':CentralCorridor(),
		'laser_weapon_armory':LaserWeaponArmory(),
		'the_bridge':TheBridge(),
		'escape_pod':EscapePod(),
		'death':Death(),
		'finished':Finished(),
		
	}
	def __init__(self, start_scene):
		self.start_scene = start_scene
		
	def next_scene(self, scene_name):
		val = Map.scenes.get(scene_name)
		return val
		
	def opening_scene(self):
		return self.next_scene(self.start_scene)
		
a_map = Map('centrral_corridor')
a_game = Engine(a_map)
#a_game.play()

print()
print("ex44----------继承与组合-----------------------------------------")
print()
print('''大部分使用继承的场合都可以用组合取代或简化，
而多重继承则需要不惜一切地避免。''')

class Parent(object):
	
	def __init__(self):
		print("Parent initializing...")
		
	def implicit(self):
		print("Parent implicit()")
		
class Child(Parent):
	
	pass

class Child2(Parent):
	
	def implicit(self):
		print("CHILD override()")
		
class Child3(Parent):
	
	def implicit(self):
		print("CHILD,BEFORE PARENT implicit()")
		
		super(Child3, self).implicit()
		print("CHILD, AFTER PARENT implicit()")
	
	
dad = Parent()
son = Child()
son2 = Child2()
son3 = Child3()


dad.implicit()
#下面仍旧是调用从父类继承来的方法
son.implicit()
#重写后的结果
son2.implicit()
#用Child3和self这两个参数调用super，然后在此返回的基础上调用implicit()
son3.implicit()

#super()
#在父类中查找函数执行
#为实现这一点Python使用了一个叫“方法解析顺序”（method resolutionorder, MRO）的东西，
#还用了一个叫C3的算法。

#super()和__init__搭配使用
class Child4(Parent):
	def __init__(self, stuff):
		self.stuff =stuff
		super(Child4, self).__init__()

son4 = Child4(58)
son4.implicit()
print(son4.stuff)

#继承和组合主要是解决代码复用的问题
#继承通过创建一种让你在基类里隐含父类的功能的机制来解决这个问题，
#而组合则是利用模块和别的类中的函数调用达到了相同的目的。
#如果有一些代码会在不同位置和场合应用到，那就用组合来把它们做成模块。


#提高解决问题能力的唯一方法就是自己去努力解决尽可能多的问题

#在使用类的过程中，你的很大一部分时间用在告诉你的类如何“做事情”。
#给这些函数命名的时候，与其命名成一个名词，
#不如命名为一个动词，作为给类的一个命令。
#就和list的pop（弹出）函数一样，让函数保持简单小巧


#不要使用来自模块的变量或者全局变量，让这些东西自顾自就行了
#如果一段代码你无法朗读出来，那么这段代码的可读性可能就有问题
#要尽量让注释短小精悍、一语中的，如果你对代码做了更改，
#记得检查并更新相关的注释，确认它们还是正确的。


