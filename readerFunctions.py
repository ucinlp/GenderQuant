import os, sys
import numpy as np
import spacy
import pickle

def refineContext(sentence, opts):
    nonConsideredPeople = [opts.maleTokClf, opts.femTokClf]
    consideredPeople    = [opts.maleTokPronoun, opts.femTokPronoun, opts.maleTokDB, opts.femTokDB]

    sentence = ['PERSON' if (x in nonConsideredPeople or x in consideredPeople) else "HIMSELF_HERSELF" if x=="himself_herself" else 'HIM/HER/HIS/HERS' if x=="mixed_pronouns" else 'PERSON' if x=='unknown_gender' else 'TITLE' if (x in opts.titles) else x for x in sentence]
    return sentence

def giveMatchingIndices(sent, token):
    locs = np.where(np.array(sent) == token)[0]
    return locs

def readFinalDataset(fileLoc):
    with open(fileLoc) as f:
        lines = f.readlines()
    lines = [a.strip() for a in lines]
    return lines

def getClassifierAndNamesList():
    babyNamesDir = 'data/censusData/names/'
    females = readFinalDataset(os.path.join(babyNamesDir, 'ssnFemales.txt'))
    males   = readFinalDataset(os.path.join(babyNamesDir, 'ssnMales.txt'  ))

    return males, females

def processNER(token, label, curr_sent, opts):
    import re

    maleTokClf = "MALE_CLASSIFIER"
    femTokClf  = "FEMALE_CLASSIFIER"

    genderTokens = [maleTokClf, femTokClf, 'MALE_PRONOUN', 'FEMALE_PRONOUN', 'MALE_SUBJECT', 'FEMALE_SUBJECT', 'MALE_DATABASE', 'FEMALE_DATABASE', 'UNKNOWN_GENDER']

    toAppend = ''

    # NER TAG PERSON
    if label=="PERSON" or label=='person':
        regex          = re.compile('[^a-zA-Z]')
        simplifiedName = regex.sub('', token) # removing -'.' etc from name
        if len(curr_sent)>0 and (curr_sent[-1] in genderTokens):
            toAppend = '' # Because this is probably surname. So we already have the name of main person
        elif len(curr_sent)>0 and curr_sent[-1] in opts.mr:
            toAppend = 'MALE_DATABASE'
        elif len(curr_sent)>0 and curr_sent[-1] in opts.mrs:
            toAppend = 'FEMALE_DATABASE'
        elif len(simplifiedName)>1: # Condition on this len, because empty token occured in some files
            gender = GenderFromName(token, opts.femaleNames, opts.maleNames)
            toAppend = gender
        else:
            toAppend = token
    # Check for PRONOUNS
    else:
        toAppend = checkForPronouns(token, opts)

    if toAppend is None: toAppend = token

    return toAppend

def GenderFromName(name, femNameList, maleNameList):
    if name.lower() in femNameList: return "FEMALE_DATABASE"
    elif name.lower() in maleNameList: return "MALE_DATABASE"
    else: return 'UNKNOWN_GENDER'

def checkForPronouns(word, opts):
    toAppend = None
    word = word.lower()
    if word=="he":
        toAppend = opts.prnCheck_maleTok
    elif word=="she":
        toAppend = opts.prnCheck_femTok
    elif word=="himself" or word=="herself":
        toAppend = opts.prnCheck_refl_pronoun
    elif word=="his" or word=="her" or word=="him" or word=="hers":
        toAppend = opts.prnCheck_other_pronouns
    return toAppend
