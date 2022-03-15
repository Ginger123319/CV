# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def print_msg(msg):
    # 2-1
    print(f'msg is: {msg}')
    # 2-2 2-3
    msg = 'day by day'
    print(f'new mag is :{msg}')
    # 2-5
    print('"123456!"have')
    # 2-6
    msg = '{} said: " love is powerful! "'.format(msg)
    print(msg)


'''
Ginger
20220121
'''


def upper_lower_name(name):
    # 2-4
    print(f'the upper name is:{name.upper()}')
    print(f'the lower name is:{name.lower()}')
    print(f'the first name is upper:{name.capitalize()}')
    # 2-7
    print(name.lstrip())
    print(name.rstrip())
    print(name.strip())


def print_name_list(namelist):
    # 3-1
    print(namelist)
    for i in list(namelist):
        print(f'{i} hello')
    # 3-5
    p = namelist.pop()
    print(p)
    print(namelist)
    namelist.append('D')
    print(namelist[2])
    namelist[2] = 'E'
    print(namelist)
    print(len(namelist))
    # 3-6
    namelist.insert(0, 'Head')
    namelist.insert(2, 'Middle')
    namelist.append('Tail')
    print(namelist)
    # 3-7
    while len(namelist) > 2:
        print(namelist.pop())
    print(namelist)
    while namelist:
        del namelist[0]
    print(namelist)

# 3-8


def place_sorted(place_list):
    print(place_list)
    for place in  list(place_list):
        print(place)
        type(place)
    print(sorted(place_list))
    print(place_list)
    print(sorted(place_list, reverse=True))
    print(place_list)

    # list.reverse()
    print(place_list.reverse())
    print(place_list)
    print(place_list.reverse())
    print(place_list)

    # list.sort()
    print(place_list.sort())
    print(place_list)
    print(place_list.sort(reverse=True))
    print(place_list)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print_msg('on the road!')
    upper_lower_name(' Gin\tger ')
    names = ['A', 'B', 'C']
    print_name_list(names)
    places = ['I', 'H', 'G', 'Q']
    place_sorted(places)
