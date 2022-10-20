import requests
import time
import os
import re


def main(save_img_dir="bg_pic"):
    word = input("请输入关键词：")

    img_dir = save_img_dir
    os.makedirs(img_dir, exist_ok=True)

    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Upgrade-Insecure-Requests': '1'
    }

    session = requests.Session()
    session.headers = headers

    pn = 0
    num = 1
    print(f"准备下载关于 {word} 的图片，保存在文件夹 {img_dir} 中")

    url_list = []
    while (pn + 60) < 100:
        url = f"https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word={word}&pn={pn}"
        pn = pn + 60
        result = session.get(url, timeout=10, allow_redirects=False)

        for img_url in re.findall('"objURL":"(.*?)",', result.text, re.S):
            print(f"正在下载第{len(url_list) + 1}张图片, URL={img_url}")

            try:
                if img_url not in url_list:
                    pic = requests.get(img_url, timeout=7)
                    img_path = os.path.join(img_dir, str(num) + '.jpg')
                    fp = open(img_path, 'wb')
                    fp.write(pic.content)

                    fp.close()
                    url_list.append(img_url)
                    num += 1

            except BaseException as e:
                # print(e)
                print("当前图片下载出错")
    print('下载完成')


if __name__ == '__main__':
    main(save_img_dir="/home/room/crawler_pictures/paoMaoDLou")
