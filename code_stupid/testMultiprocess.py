from multiprocessing import Process


def print_name(name):
    print(name)
    return name


print(111111111111111)

if __name__ == '__main__':
    p = Process(target=print_name, args=("JYF",))
    print(22222222222222222222222222222)
    p.start()
    p.join()
    print(333333333333333333333)
    p.terminate()
    print(.3)
