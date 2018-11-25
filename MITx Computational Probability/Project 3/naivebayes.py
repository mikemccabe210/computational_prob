import sys
import os.path
import numpy as np

import util
from collections import defaultdict

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    ### TODO: Comment out the following line and write your code here
    count_dict = defaultdict(lambda: 0)
    for file in file_list:
        words = list(set(util.get_words_in_file(file)))
        for word in words:
            count_dict[word] += 1
#    print(sorted(count_dict.items(), key = lambda x: x[1]))
#    print(count_dict['delegating'])
    return count_dict
        

def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    ### TODO: Comment out the following line and write your code here
    raw_counts = get_counts(file_list)
    total_files = len(file_list)
    log_probs = defaultdict(lambda: np.log(1/(2+total_files)))
    for key, val in raw_counts.items():
        log_probs[key] = np.log((val + 1) / (total_files + 2))

#    print(log_probs['delegating'])
    global prob
    prob = log_probs
    return log_probs
        


def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    ### TODO: Comment out the following line and write your code here
    total = len(file_lists_by_category[0]) + len(file_lists_by_category[1]) 
    prior = [np.log(len(file_lists_by_category[0])/total), np.log(len(file_lists_by_category[1])/total)]
    
    
    return ([get_log_probabilities(file_lists_by_category[0]), get_log_probabilities(file_lists_by_category[1])], prior)
    
    

def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names. sempra
    """
    ### TODO: Comment out the following line and write your code here
    
    words_in_file = list(set(util.get_words_in_file(email_filename)))
#    for spam in log_probabilities_by_category[0]:
#        if spam not in words_in_file:
#            print(spam, 1-np.exp(log_probabilities_by_category[0][spam]))
    spam_prod = 0
    ham_prod = 0
    
    for word in words_in_file:
#        if 1-np.exp(log_probabilities_by_category[0][word]) <= 0:
#            print(word, 1-np.exp(log_probabilities_by_category[0][word]))
        spam_prod += log_probabilities_by_category[0][word]
        ham_prod += log_probabilities_by_category[0][word]
    for word in log_probabilities_by_category[0]:
        if word not in words_in_file:
            spam_prod += np.log(1 - np.exp(log_probabilities_by_category[0][word]))
#            if 1-np.exp(log_probabilities_by_category[0][word]) <= 0 and word == 'delegating':
#                print('spam', word, 1-np.exp(log_probabilities_by_category[0][word]))
# 
    for word in log_probabilities_by_category[1]:
        if word not in words_in_file:
            ham_prod += np.log(1 - np.exp(log_probabilities_by_category[1][word]))  
#            if 1-np.exp(log_probabilities_by_category[0][word]) <= 0 and word == 'delegating':
#                print('ham', word, 1-np.exp(log_probabilities_by_category[1][word]))  
#    print('horcrux', word, 1-np.exp(log_probabilities_by_category[0]['horcrux']))
#            
#    print('horcrux', word, 1-np.exp(log_probabilities_by_category[1]['horcrux']))

#    spam_prod = sum([log_probabilities_by_category[0][x] if x in words_in_file else np.log(1-np.exp(log_probabilities_by_category[0][x])) for x in log_probabilities_by_category[0]])
#    ham_prod =  sum([log_probabilities_by_category[0][x] if x in words_in_file else np.log(1-np.exp(log_probabilities_by_category[1][x])) for x in log_probabilities_by_category[1]])
    log_odds_num = log_prior_by_category[0] + spam_prod
    log_odds_denom = log_prior_by_category[1] + ham_prod
    
    log_odds = log_odds_num - log_odds_denom
    
#    print(log_odds_num, log_odds_denom, log_odds)
    
#    print('spam3 ' + str(len(log_probabilities_by_category[0])))
#    print('ham3 ' + str(len(log_probabilities_by_category[1])))
    
    if log_odds >= 0:
        return 'spam'
    else:
        return 'ham'
    
#    return 'spam'

def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels

def main():
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
        
    args = ['data/testing/', 'data/spam/', 'data/ham/']
    testing_folder = 'data/testing/'
    (spam_folder, ham_folder) = args[1:]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)
#    print('spam1 ' + str(len(log_probabilities_by_category[0])))
#    print('ham1 ' + str(len(log_probabilities_by_category[1])))

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

if __name__ == '__main__':
    main()
