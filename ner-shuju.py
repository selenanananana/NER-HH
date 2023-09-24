import codecs
import sys


def delete_char(input_path, output_path):
    linedata = []
    lines = 0

    input_data = codecs.open(input_path, 'r', 'utf-8')
    output_data = codecs.open(output_path, 'w', 'utf-8')
    for line in input_data.readlines():  # 按行读取数据
        line = line.split()
        wordCount = 0
        for word in line:
            word = word.split()  # 用'/'将word给划分开。可以将标记和词语分开。
            linedata.append(word[0])
            wordCount = wordCount + 1
        # if wordCount > 500:
        #     print("##################THE LINE IS:", lines+1)
        #     print("##################THE wordCount IS:",wordCount)
        if wordCount < 500:
            print("##################THE LINE IS:", lines + 1)
            print("##################THE wordCount IS:", wordCount)
            for word in line:
                word = word.split()
                output_data.write(word[0] + " ")
            output_data.write('\n')
        lines = lines + 1
    print("SUCCESS")


def check_word(file_name):
    line_count = 0
    word_count = 0
    character_count = 0
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                continue
            word = line.split()
            line_count += 1
            print("###########第", line_count, "行########")
            word_count = len(word)
            print("句子中词语的个数是（不包括空格）：", word_count)
            temp = 0
            for i in line:
                char = str(i)
                if char != " " and char != "\n":
                    # print(temp,i,"#####")
                    character_count = character_count + 1
                    temp = temp + 1
            print("句子中字符的个数是（不包括空格）：", character_count)
            character_count = 0


def delete_word(input_path, output_path):
    line_count = 0
    character_count = 0
    output_data = codecs.open(output_path, 'w', 'utf-8')
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                continue
            temp = 0
            for i in line:
                char = str(i)
                if char != " " and char != "\n":
                    character_count = character_count + 1
                    temp = temp + 1
            if character_count < 500:
                for word in line.strip():
                    output_data.write(word[0])
                output_data.write('\n')
                print("##################THE LINE IS:", line_count + 1)
                print("句子中字符的个数是（不包括空格）：", character_count)
            character_count = 0
            line_count = line_count + 1


if __name__ == "__main__":
    # delete_char("E://PycharmCode//data_processing//MSRA//train_char.txt","E://PycharmCode//data_processing//output.txt")
    # delete_char("E://PycharmCode//data_processing//MSRA//train_bioattr.txt","E://PycharmCode//data_processing//output2.txt")
    # delete_char("E://PycharmCode//data_processing//test.txt","E://PycharmCode//data_processing//output3.txt")
    # delete_char("E://PycharmCode//data_processing//test.txt","E://PycharmCode//data_processing//output.txt")
    # check_word("test.txt")
    delete_word("test.txt", "delete_word_output.txt")