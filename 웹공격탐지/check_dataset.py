
keyword = ["POST", "GET", "User-Agent", "Pragma", "Cache-control", "Accept", "Accept-Encoding",
        "Accept-Charset", "Accept-Language","Host", "Cookie", "Content-Type", "Connection", "Content-Length"]
get_keyword= ["POST", "User-Agent", "Pragma", "Cache-control", "Accept", "Accept-Encoding",
        "Accept-Charset", "Accept-Language","Host", "Cookie", "Connection"]

def check_data(dataset_name):
    fp = open(dataset_name, "r", encoding="UTF-8")
    print("대상 데이터셋 : " + dataset_name)
    lines = fp.readlines()
    fp.close()

    result_get = []
    result_post = []
    temp_list = []
    content_read = False

    for line in lines:
        if content_read == True:
            if line != "\n":
                temp_list.append(line)
                content_read = False
                continue

        words = line.split(" ")
        temp = words[0].replace(":", "")

        if temp in keyword:
            if temp == "GET":
                result_get.append(temp_list)
                temp_list = []
            elif temp == "POST":
                result_post.append(temp_list)
                temp_list = []

            add = ""
            for i in range(1, len(words)):
                add += words[i]
            add = add[:len(add)-1]
            temp_list.append(add)

            if temp == "Content-Length" and int(words[1]) > 0:
                content_read = True

    del result_get[0]
    del result_post[0]

    for i in range(0, len(result_get[0])):
        temp = []
        for result in result_get:
            if result[i] not in temp:
                temp.append(result[i])
        if len(temp) < 3:
            print("get 로그의 " + get_keyword[i] + " 내용은 모두 같다.")
    for i in range(0, len(result_post[0])):
        temp = []
        for result in result_post:
            if result[i] not in temp:
                temp.append(result[i])
        if len(temp) < 3:
            print("post 로그의 " + keyword[i+1] + " 내용은 모두 같다.")

check_data("norm_train.txt")
print("\n")
check_data("anomal_train.txt")
