import sys
import re
import svm
from svmutil import *
import re, pickle, csv, os

def format_output(test_raw, attack_ratings, p_labels, p_vals, total, correct, incorrect, accuracy):

    output_file = open("output.txt", 'w')
    output_file.write('Total = %d \n' % \
                                    (total))
    output_file.write('Correct = %d \n' % \
                                    (correct))
    output_file.write('Incorrect = %d \n' % \
                                    (incorrect))
    output_file.write('Accuracy = %.3f \n' % \
                                    (accuracy))
    output_file.write("Test Comment" + "\t" + "Assigned Attack Rating" + "\t" + "Predicted Attack Rating" + 
        "\t" + "Probability Estimate" + "\n")
    for t, a, p, e in zip(test_raw, attack_ratings, p_labels, p_vals):
        output_file.write(t + "\t" + str(a) + "\t" + str(int(p)) + "\t" + str(e) + "\n")

def get_SVM_feature_vector(comments, feature_list):
    sorted_features = sorted(feature_list)
    word_map = {}
    feature_vector = []
    for i in comments:
        label = 0
        word_map = {}
        for feature in sorted_features:
            word_map[feature] = 0
        for word in i:
            if word in word_map:
                word_map[word] = 1
        values = list(word_map.values())
        feature_vector.append(values)
    return feature_vector

def get_SVM_feature_vector_and_labels(comments, feature_list):
    sorted_features = sorted(feature_list)
    word_map = {}
    feature_vector = []
    labels = []
    for i in comments:
        label = 0
        word_map = {}
        for feature in sorted_features:
            word_map[feature] = 0

        comment_words = i[0]
        comment_attack = i[1]

        for word in comment_words:
            # Replace repetitions of character and replace with the character itself
            pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
            pattern.sub(r"\1\1", word)
            # Strip all of the punctuation
            word = word.strip('\'"?,.!')
            # Set word_map[word] to 1 if word exists
            word_map[word] = 1
        values = list(word_map.values())
        feature_vector.append(values)
        if comment_attack == 0:
            label = 0
        elif comment_attack == 1:
            label = 1
        labels.append(label)
    # Return the feature_vector and the labels
    return {'feature_vector': feature_vector, 'labels': labels}

def get_feature_vector(comment, stop_words):

    # Method that returns a list of words, or features, from the comment inputted
    feature_vector = []
    words = comment.split()
    for word in words:
        # Replace repetitions of character and replace with the character itself
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        pattern.sub(r"\1\1", word)
        # Strip all of the punctuation
        word = word.strip('\'"?,.!')
        # Match words that begin with a letter
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
        if(word in stop_words or val is None):
            continue
        else:
            feature_vector.append(word.lower())

    return feature_vector

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
    # Remove =
    comment = re.sub('=', '', comment)
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
        file = open('data/comments_sample.txt')
    except(IndexError, IOError):
        print("Invalid path for train txt file")

    comment_list = file.readlines()[1:]
    file.close()

    # Open the stop words file
    try:
        file = open('data/stop_words.txt')
    except(IndexError, IOError):
        print("Invalid path for stop words txt file")

    stop_list = file.readlines()
    file.close()
    stop_words = []

    for word in stop_list:
        word = word.strip()
        stop_words.append(word)

    comments = []
    feature_list = []

    for row in comment_list:
        row = row.split("\t")
        attack = int(row[0])
        comment = row[1]
        preprocessed_comment = preprocess(comment)
        feature_vector = get_feature_vector(preprocessed_comment, stop_words)
        comments.append((feature_vector, attack))
        for feature in feature_vector:
            feature_list.append(feature)

    # Train the classifier
    training = get_SVM_feature_vector_and_labels(comments, feature_list)
    problem = svm_problem(training['labels'], training['feature_vector'])
    param = svm_parameter('-s 1 -b 1')
    param.kernel_type = LINEAR
    classifier = svm_train(problem, param)
    svm_save_model('classifier.txt', classifier)

    # Test the classifier

    try:
        file = open('data/test_comments.txt')
    except(IndexError, IOError):
        print("Invalid path for train txt file")

    test_list = file.readlines()[1:]
    file.close()
    test_comments = []
    attack_ratings = []
    test_raw = []

    for row in test_list:
        row = row.split("\t")
        attack = int(row[0])
        comment = row[1]
        words = [e.lower() for e in comment.split()]
        test_comments.append(words)
        test_raw.append(comment)
        attack_ratings.append(attack)

    test_feature_vector = get_SVM_feature_vector(test_comments, feature_list)
    p_labels, p_accs, p_vals = svm_predict(attack_ratings, test_feature_vector, classifier, options = "-b 1")
    # (a, b, c) = evaluations(attack_ratings, p_labels)
    # print((a,b,c))

    count = 0
    total, correct, incorrect = 0, 0, 0
    accuracy = 0.0
    for l in attack_ratings:
        label = int(p_labels[count])
        if(label == int(l)):
            correct += 1
        else:
            incorrect += 1
        total += 1
        count += 1
    accuracy = (float(correct)/total)*100

    format_output(test_raw, attack_ratings, p_labels, p_vals, total, correct, incorrect, accuracy)

main()





