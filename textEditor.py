import string
import csv
import os
################################ FUNCTION 1 ################################
#        Split a long text into many short texts with chosen length        #
############################################################################
def toShortText():
    lines_per_file = 3
    smallfile = None
    with open('data/dataSet/dataset.txt') as bigfile:
        for lineno, line in enumerate(bigfile):
            lines = (line.split("\t"))
            print(lines)
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = 'small_file_{}.txt'.format(lineno + lines_per_file)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

################################ FUNCTION 2 ################################
#                     Creating CSV file from TXT file                      #
############################################################################
#toCSV
def toCSV():
    with open('data/sentiment_analysis/yelp_labelled.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split("\t") for line in stripped if line)
        with open('log.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('text', 'author'))
            writer.writerows(lines)

################################ FUNCTION 3 ################################
#   Creating a directory for each author and file for each authors texts   #
############################################################################
printable = set(string.printable)
smallfile = None
with open('data/Gungor_2018_VictorianAuthorAttribution_data-train.csv', encoding="latin-1") as bigfile:
    for lineno, line in enumerate(bigfile):
        lines = (line.split(","))
        # print(lines)
        lines[0] = lines[0].strip()
        lines[1] = lines[1].strip()
        # print(lines)
        small_filename = '{}'.format(lineno)
        if not os.path.isdir('authors/' + lines[1] + '/'):
            os.mkdir('authors/' + lines[1] + '/')
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