import ntpath
import re
import shutil
import string
import csv
import os


###################################################### FUNCTION 1 ######################################################
#                              Split a long text into many short texts with chosen length                              #
########################################################################################################################
import numpy

def toShortText(path):
    lines_per_file = 3
    lines = ''
    # dataSetFile = None
    if not os.path.isdir('data/'):
        os.mkdir('data/')
    dataSetFile = open('data/dataSet.txt', "w")
    # for file in os.listdir(path):
    for file in path:
        if file.endswith(".txt"):
            # textFile = os.path.join(path, file)
            textFile = file
            with open(textFile) as text:
                for lineno, line in enumerate(text):
                    # lines = (line.split("\t"))
                    lines = lines + line
                    # print(lines)
                    if lineno % lines_per_file == 0:
                        # if dataSetFile:
                        #     dataSetFile.close()
                        # dataSetFile = open('data/dataSet.txt', "a")
                        name, txt = file.split('.')
                        name = ntpath.basename(name).strip()
                        lines = lines.replace('\n', ' ')
                        lines = lines.translate(str.maketrans('', '', string.punctuation))
                        lines = lines + ',' + name + '\n'

                        dataSetFile.write(lines)
                        lines = ''
                if text:
                    text.close()
    if dataSetFile:
        dataSetFile.close()


###################################################### FUNCTION 2 ######################################################
#                                           Creating CSV file from TXT file                                            #
########################################################################################################################
# toCSV
def toCSV(path):
    # print("toCSV\n")
    with open(path, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        with open('data/dataSet.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            for line in stripped:
                if line:
                    text, author = line.split(",")
                    writer.writerow((text, author))


###################################################### FUNCTION 3 ######################################################
#                         Creating a directory for each author and file for each authors texts                         #
########################################################################################################################
def createDataSet(path):
    printable = set(string.printable)
    smallfile = None
    authorsList = []
    if os.path.isdir('authors/'):
        shutil.rmtree('authors/', ignore_errors=True)
    os.mkdir('authors/')
    with open(path, encoding="latin-1") as bigfile:
        for lineno, line in enumerate(bigfile):
            if line is '\n':
                continue
            lines = (line.split(","))
            lines[0] = lines[0].strip()
            lines[1] = lines[1].strip()
            # print(lines)
            small_filename = '{}'.format(lineno)
            # if os.path.isdir('authors/'):
            #     shutil.rmtree('/authors', ignore_errors=True)
            #     os.rmdir('authors/')
            # os.mkdir('authors/')
            if not os.path.isdir('authors/' + lines[1] + '/'):
                os.mkdir('authors/' + lines[1] + '/')
                authorsList.append(lines[1])

            ###########################################################################
            # Copy 1
            smallfile = open('authors/' + lines[1] + '/' + small_filename, "a")
            smallfile.write(''.join(filter(lambda x: x in printable, lines[0])))
            # Copy 2
            smallfile1 = open('authors/' + lines[1] + '/' + small_filename + '1', "a")
            smallfile1.write(''.join(filter(lambda x: x in printable, lines[0])))
            # Copy 3
            smallfile2 = open('authors/' + lines[1] + '/' + small_filename + '2', "a")
            smallfile2.write(''.join(filter(lambda x: x in printable, lines[0])))
            ###########################################################################
        if smallfile:
            smallfile.close()
        if smallfile1:
            smallfile1.close()
        if smallfile2:
            smallfile2.close()
    return authorsList

# toShortText(['C:/Users/dbrailov/PycharmProjects/Authorship-Attribution-of-Short-Texts/Temp/John_Ronald_Reuel_Tolkien.txt','C:/Users/dbrailov/PycharmProjects/Authorship-Attribution-of-Short-Texts/Temp/Joanne_Jo_Rowling.txt'])
# toCSV('data/dataSet.txt')
# createDataSet('data/dataSet.csv')
