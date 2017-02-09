"""
clean strings from raw data
> lower cases
> only word with letters, numbers and words with "-" are kept
"""

__author__ = 'yiren'

import re
import os


def clean_str(string):
    """
    return cleaned lines
    """

    string = string.lower()
    string = re.sub(r"\'ve", " have", string, flags=re.I)
    string = re.sub(r"\'re", " are", string, flags=re.I)
    string = re.sub(r"\'d", " would", string, flags=re.I)
    string = re.sub(r"\'ll", " will", string, flags=re.I)
    string = re.sub(r"can\'t", " cannot", string, flags=re.I)
    string = re.sub(r"n\'t", " not", string, flags=re.I)
    string = re.sub(r"\'m", " am", string, flags=re.I)
    string = re.sub(r"\'s", " is", string, flags=re.I)
    string = re.sub(r"[^a-z\- \n]", "", string=string, flags=re.I)
    # TODO: deal with problem with 's
    string = re.sub(r"\s{2,}", " ", string)

    return string

def clean_str_sen(string):
    """
    return cleaned lines
    generate <EOS> make for sentence
    """

    string = string.lower()
    string = re.sub(r"\'ve", " have", string, flags=re.I)
    string = re.sub(r"\'re", " are", string, flags=re.I)
    string = re.sub(r"\'d", " would", string, flags=re.I)
    string = re.sub(r"\'ll", " will", string, flags=re.I)
    string = re.sub(r"can\'t", " cannot", string, flags=re.I)
    string = re.sub(r"n\'t", " not", string, flags=re.I)
    string = re.sub(r"\'m", " am", string, flags=re.I)
    string = re.sub(r"\'s", " is", string, flags=re.I)
    string = re.sub(r"\.", " <EOS>", string, flags=re.I)
    string = re.sub(r"\?", " <EOS>", string, flags=re.I)
    string = re.sub(r"\!", " <EOS>", string, flags=re.I)
    string = re.sub(r"[^a-z\- \n]", "", string=string, flags=re.I)
    # string = re.sub(r"EOS", "<EOS>", string, flags=re.I)
    # TODO: deal with problem with 's
    string = re.sub(r"\s{2,}", " ", string)

    return string

def clean_files(path_in, path_out):
    print "Preprocessing files in", path_in, "...",
    files = os.listdir(path_in)
    try:
        os.stat(path_out)
    except:
        os.mkdir(path_out)
    for file in files:
        with open(path_in + file, "r") as f:
            lines = f.readlines()
        with open(path_out + file, "w") as f:
            for line in lines:
                line = clean_str(line)
                f.write(line)
    print "Done!\n", len(files), "preprocessed!"





if __name__ == "__main__":
    polarity = "neg"
    path_in = "../../Data/Raw/"+polarity+"/"
    path_out = "../../Data/Raw/cleaned_"+polarity+"/"
    # clean_files(path_in=path_in, path_out=path_out)
    print clean_str_sen("this is a be?\n are you a. it's not my!")