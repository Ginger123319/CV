# 2--14
class Student(object):

    def __init__(self, name, age, grades):
        self.name = name
        self.age = age
        self.grades = grades
        self.course = ['C', 'M']

    def get_name(self):
        return self.name

    def get_age(self):
        return self.age

    def get_course(self):
        # return max(self.grades)
        if self.grades:
            length = len(self.grades)
            max_score = self.grades[0]
            if length == 1:
                return max_score
            else:
                # O(len(self.grades));  O(n)
                for grade in self.grades:
                    if max_score <= grade:
                        max_score = grade
                return max_score
        else:
            return "grades input error!"


zm = Student('zhangming', 20, [2, 1, 5])
print(zm.get_name())
print(type(zm.get_age()))
print(zm.get_course())


# 2--15
class Dict(object):

    def __init__(self, dic):
        self.dic = dic

    def del_dict(self, key):
        return self.dic.pop(key, 'no exist!')

    def get_dict(self, key):
        return self.dic.get(key, 'not found')

    def get_keys(self):
        return list(self.dic.keys())

    def update_dict(self, another):
        self.dic.update(another)


dict1 = Dict({
    'A': '1',
    'B': '2',
    'C': '3'
})
dict2 = {
    'D': '4',
    'E': '5'
}
print(dict1)
print(dict1.del_dict('E'))
print(dict1.get_dict('A'))
print(dict1.get_keys())
print(dict1.update_dict(dict2))
print(dict1.get_keys())


# 2--16
class ListInfo(object):
    def __init__(self, list_info_p):
        self.list_info = list_info_p

    def add_key(self, key_name):
        return self.list_info.append(key_name)

    def get_key(self, num):
        return self.list_info[num]

    def update_list(self, list_merge):
        return self.list_info.extend(list_merge)

    def del_key(self):
        return self.list_info.pop()


list_info = ListInfo([44, 222, 111, 333, 454, 'sss', '333'])
print(list_info.add_key('555'))
print(list_info.get_key(-1))
print(list_info.update_list(['X', 'Y']))
print(list_info.del_key())
print(list_info)

2 - -17


class SetInfo(object):
    def __init__(self, set_p):
        self.set_p = set_p

    def add_set_info(self, key_name):
        self.set_p.add(key_name)
        print(self.set_p)

    def get_intersection(self, union_info):
        print(self.set_p.intersection(union_info))

    def get_union(self, union_info):
        print(self.set_p.union(union_info))

    def get_difference(self, union_info):
        print(self.set_p.difference(union_info))


# disordered
set_info = SetInfo({'1', '2', '3', '4'})
set_info.add_set_info('5')
# clear the repeated
set_info.get_union({'4', '5', '6'})
set_info.get_intersection({'4', '5'})
set_info.get_difference({'1', '2'})


# 2--19
class Undergraduate(Student):
    def __init__(self, name, age, grades, year):
        super(Undergraduate, self).__init__(name, age, grades)
        self.year = year


s1 = Undergraduate('Mary', 18, ['A', 'C', 'D'], 'grade one')
print(s1.course)
print(s1.get_age())
