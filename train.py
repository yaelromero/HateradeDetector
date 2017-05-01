import sys
import re

def preprocess(comment):

    # Method that preprocesses the comment, prepares it for extracting features

    comment = comment.lower()
    # Replace urls with the word URL
    comment = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', comment)
    # Remove NEWLINE_TOKEN
    comment = re.sub('newline_token', '', comment)
    # Remove TAB_TOKEN
    comment = re.sub('tab_token', '', comment)
    # Convert `` to "
    comment = re.sub('``', '"', comment)
    # Convert ` to '
    comment = re.sub('`', '\'', comment)
    # Remove \n
    comment = re.sub('\n', '', comment)
    # Remove :
    comment = re.sub(':', '', comment)
    # Remove extra spaces
    comment = re.sub('[\s]+', ' ', comment)
    # Strip leading and trailing spaces
    comment = comment.strip()
    # Remove quotes
    comment = comment.strip('\'"')
    comment = comment.strip()

    return comment

def main():

    # Open the file

    try:
        file = open('data/comments_sample.txt', 'r')
    except(IndexError, IOError):
        print("Invalid path for train txt file")

    lineList = file.readlines()[1:]
    file.close()

    comments = {}

    for row in lineList:
        row = row.split("\t")
        attack = row[0]
        comment = row[1]
        preprocessed_comment = preprocess(comment)
        # feature_vector = get_feature_vector(preprocessed_comment, stop_words)
        comments[preprocessed_comment] = attack

    for comment in comments:
        print(str(comment) + "\t" + str(comments[comment]) + "\n")

main()





