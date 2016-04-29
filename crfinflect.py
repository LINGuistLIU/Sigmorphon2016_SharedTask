# -*- coding: utf-8 -*-
# Solves the SIGMORPHON shared tasks 2016 for track 1 and 2 and tasks 1,2, and 3

# Usage: python crfinflect.py [d|t] [u|c] [1|2|3] language
#        d=dev t=test u=unconstrained (track 1) c = constrained (track 2)
# Example: python crfinflect.py d u 2 arabic
# runs task 2 on arabic dev set, unconstrained (uses task 1 data, too)

import align, codecs, sys, re
import sklearn
import sklearn_crfsuite
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import consvowOCP

def getsubstrings(word, maxlen = 20, prefix = True, suffix = True):
    word = '#' + word + '#'
    substrings = set()
    if prefix:
        substrings |= set([word[0:j] for j in xrange(2,min((maxlen+1), len(word)+1))])
    if suffix:
        substrings |= set([word[j:len(word)+1] for j in xrange(max(0,len(word)-maxlen),len(word)-1)])
    return {x:1 for x in substrings}
        
def decisionpair(s, t):
    tout = []
    for i in xrange(len(s)):
        if s[i] == t[i]:
            tout.append(str(ord(u'R')))           
        elif t[i] == u'_':
            tout.append(str(ord(u'D')))
        else:
            tout.append(chrencode(t[i]))
    return (s, tout)
            
def singleinput(s, t):
    """Convert aligned sequence so that input can be processed symbol by symbol (no zeroes on input)
    >>> singleinput('   word    ','prewordpost')
    ([u'<', u' ', u' ', u' ', u'w', u'o', u'r', u'd', u' ', u' ', u' ', u' ', u'>'], 
     [u'<', u'p', u'r', u'e', u'w', u'o', u'r', u'd', u'p', u'o', u's', u't', u'>'])
    """
    s = u'<' + s + u'>'
    t = u'<' + t + u'>'
    sout = []
    tout = []
    idx = 0
    while idx < len(s):
        if idx == 0 and s[idx+1] == u'_':
            idxs = idx + 1
            while s[idxs] == u'_':
                idxs += 1
            sout.append(s[idx:idxs].replace(u'_',u''))
            tout.append(t[idx:idxs].replace(u'_',u''))
            idx = idxs
        elif s[idx] != u'_':
            sout.append(s[idx])
            tout.append(t[idx])
            idx += 1
        else:
            idxs = idx + 1
            while s[idxs] == u'_':
                idxs += 1
            sout.append(s[idx:idxs+1].replace(u'_',u''))
            tout.append(t[idx:idxs+1].replace(u'_',u'')) 
            idx = idxs + 1
    return ((sout, tout))
    
def word2features(word):
    return [token2features(word, i) for i in range(len(word))]

def word2labels(word):
    return [action for inword, action in word]

def token2features(word, i):

    allvowels = {v for v in word if v in V}
    lastvowel = 'X'
    j = i
    while j >= 0:
        if word[j] in V:
            lastvowel = word[j]
            break
        else:
            j -= 1
            
    features = {
        'bias': 1.0,
        'insymbol': word[i],
        'frombeg': i,
        'fromend': len(word) - i,
        'isC': word[i] in C,
        'isV': word[i] in V,
        'lastvowel': lastvowel,
        'allvowels': ''.join(sorted(list(allvowels)))
    }

    if i > 0:
        features.update({
            'prevsymbol': word[i-1],
            'geminate': word[i] == word[i-1],
            'prevC': word[i-1] in C,
            'prevV': word[i-1] in V,
            'previoustwo': word[i-1] + word[i],
            'BOW': False
        })
    else:
        features['BOW'] = True
        
    if i > 1:
        features.update({
            'prevsymbol2': word[i-2],
        })
    if i > 2:
        features.update({
            'prevsymbol3': word[i-3],
        })

    if i < len(word)-1:
        features.update({
            'nextsymbol': word[i+1],
            'nextgeminate':  word[i] == word[i+1],
            'nextC': word[i+1] in C,
            'nextV': word[i+1] in V,
            'nexttwo': word[i] + word[i+1],
            'EOW': False
        })

    else:
        features['EOW'] = True

    if i < len(word)-2:
        nextsymbol2 = word[i+2]
        features.update({
            'nextsymbol2': nextsymbol2,
        })

    if i > 0 and i < len(word)-1:
        trigram = word[i-1] + word[i] + word[i+1]
        features.update({
          #  'trigram': trigram,
          #  'trigramcv': ''.join(['C' if x in C else 'V' for x in trigram])
        })

    return features

def chrencode(s):
    return u'-'.join(map(lambda x: str(ord(x)), s))

def chrdecode(s):
    res = []
    chrs = s.split(u'-')
    res = [unichr(int(s)) for s in chrs]
    return ''.join(res)

def unfold(str, prediction):
    outstring = []
    for i in xrange(len(str)):
        if prediction[i] == '82':
            outstring.append(str[i])
        elif prediction[i] == '68':
            continue
        else:
            outstring.append(chrdecode(prediction[i]))
    return ''.join(outstring)
    
mode = sys.argv[1]       # d (dev) or t (test)
uc = sys.argv[2]         # u or c
task = int(sys.argv[3])  # 1, 2 or 3
lang = sys.argv[4]       # arabic, finnish, ...

constrained = False
if uc == 'c':
    constrained = True        # Whether to only use task-specific data
task1totask2expand = False    # Whether to create artificial training data to task 2 from task 1 (only for task 2u)
addcitationforms = False      # Whether to deduce what MSD the citation (lemma) form and add that to task 1 data (only for task 2u)

fromlemma = {}
tolemma = {}


if not constrained or task == 1:
    lines1 = [line.strip() for line in codecs.open(lang + "-task1-train", "r", encoding="utf-8")]
    if mode == 't':
        lines1 += [line.strip() for line in codecs.open(lang + "-task1-dev", "r", encoding="utf-8")]

if task == 2 or (task == 3 and not constrained):
    lines2 = [line.strip() for line in codecs.open(lang + "-task2-train", "r", encoding="utf-8")]
    if mode == 't':
        lines2 += [line.strip() for line in codecs.open(lang + "-task2-dev", "r", encoding="utf-8")]

if task == 3:
    lines3 = [line.strip() for line in codecs.open(lang + "-task3-train", "r", encoding="utf-8")]
    if mode == 't':
        lines3 += [line.strip() for line in codecs.open(lang + "-task3-dev", "r", encoding="utf-8")]

# Train SVM to learn to map word => MSD

if task == 3:
    sys.stderr.write("TRAINING SVM for task 3\n")
    formfeatures = []
    formclasses = []
    if not constrained:
        for l in lines1:
            lemma, msd, form = l.split(u'\t')
            formf = getsubstrings(form)
            formfeatures.append(formf)
            formclasses.append(msd)
        for l in lines2:
            msd1, form1, msd2, form2 = l.split(u'\t')
            formf = getsubstrings(form1)
            formfeatures.append(formf)
            formclasses.append(msd1)
            formf = getsubstrings(form2)
            formfeatures.append(formf)
            formclasses.append(msd2)
    for l in lines3:
        _, msd, form = l.split(u'\t')
        formf = getsubstrings(form)
        formfeatures.append(formf)
        formclasses.append(msd)
    # Split formfeatures and formclasses per POS
    poses = set()
    for p in formclasses:
        thispos = re.search(r'pos=([^,]*)', p).group(1)
        poses.add(thispos)
            
    formfs = {p:[] for p in poses}
    formcs = {p:[] for p in poses}

    for i in xrange(len(formfeatures)):
        thispos = re.search(r'pos=([^,]*)', formclasses[i]).group(1)
        formfs[thispos].append(formfeatures[i])
        formcs[thispos].append(formclasses[i])
            
    clfSVM = {}
    vectorizer = {}
    for pos in poses:
        vectorizer[pos] = DictVectorizer(sparse = True)
        X = vectorizer[pos].fit_transform(formfs[pos])
        clfSVM[pos] = svm.LinearSVC()
        clfSVM[pos].fit(X, formcs[pos]) 

    sys.stderr.write("TRAINED SVM for task 3\n")

words = []
traindata1 = []
traindata2 = []
citationforms = {}

if task == 1 or not constrained:
    for l in lines1:
        lemma, msd, form = l.split(u'\t')
        traindata1.append((lemma, form, msd))
        if lemma == form:
            citationforms[msd] = citationforms.get(msd, 0) + 1
        words.append(lemma)
        words.append(form)

if task == 2:
    for l in lines2:
        msd1, form1, msd2, form2 = l.split(u'\t')
        traindata2.append((msd1, form1, msd2, form2))
        traindata2.append((msd2, form2, msd1, form1))
        words.append(form1) # Append these to C,V 
        words.append(form2)

if task == 3:
    for l in lines3:
        form1, msd2, form2 = l.split(u'\t')
        words.append(form1) # Append these to C,V 
        words.append(form2)        
                
if task1totask2expand and task != 1:
    newlinecount = 0
    ptrn = {}
    for l in lines1:
        lemma, msd, form = l.split(u'\t')
        if lemma not in ptrn:
            ptrn[lemma] = []
        ptrn[lemma].append((form,msd))
    for lemma in ptrn:
        for x in ptrn[lemma]:
            for y in ptrn[lemma]:
                if x[1] != y[1]:
                    newlinecount += 1
                    lines2.append(x[1] + '\t' + x[0] + '\t' + y[1] + '\t' + y[0])
            
# Add known citation forms to task 1 data if we're solving task 2

if task != 1 and addcitationforms == True:
    sys.stderr.write("ADDING 2 citation data\n")
    negcitationforms = {c:0 for c in citationforms}

    for l in lines1:
        lemma, msd, form = l.split(u'\t')
        if lemma != form and msd in citationforms:
            negcitationforms[msd] = negcitationforms.get(msd, 0) + 1

    citationforms = {c for c in citationforms if citationforms[c] > 4 and citationforms[c] / float(citationforms[c] + negcitationforms[c]) >= 0.95}        

    for l in lines2:
        msd1, form1, msd2, form2 = l.split(u'\t')
        if msd1 in citationforms and msd1 != msd2:
            traindata1.append((form1, form2, msd2))
        if msd2 in citationforms and msd1 != msd2:
            traindata1.append((form2, form1, msd1))

if task == 1 or not constrained:
    wordpairs = [(x[0],x[1]) for x in traindata1]
    a = align.Aligner(wordpairs, align_symbol = u'_', iterations = 30)
    traindata1 = [(traindata1[i][0], traindata1[i][1], traindata1[i][2], a.alignedpairs[i][0], a.alignedpairs[i][1]) for i in range(len(traindata1))]

C, V = consvowOCP.candv(words)

# Lemma > form
if task == 1 or not constrained:
    for lemma, form, msd, lemmaaligned, formaligned in traindata1:
        if msd not in fromlemma:
            fromlemma[msd] = []
        if msd not in tolemma:
            tolemma[msd] = []
     
        alignedpair1 = (lemmaaligned, formaligned)
        choppedpair1 = singleinput(alignedpair1[0], alignedpair1[1])
        decision1 = decisionpair(choppedpair1[0], choppedpair1[1])
        fromlemma[msd].append(zip(decision1[0], decision1[1]))

        if task > 1:
            alignedpair2 = (formaligned, lemmaaligned)
            choppedpair2 = singleinput(alignedpair2[0], alignedpair2[1])
            decision2 = decisionpair(choppedpair2[0], choppedpair2[1])
            tolemma[msd].append(zip(decision2[0], decision2[1]))
    

if (task == 2 or task == 3) and constrained:  # Collect form > form data
    formtoform = {}
    if task == 3:
        wordpairs = []
        traindata3 = []
        for l in lines3:
            form1, msd2, form2 = l.split('\t')
            thispos = re.search(r'pos=([^,]*)', msd2).group(1) # Since we don't know the msd of the first column, we use SVM
            msd1 = clfSVM[thispos].predict(vectorizer[thispos].transform([getsubstrings(form1)]))[0]
            traindata3.append((msd1,form1,msd2,form2))
            wordpairs.append((form1,form2))
            traindata3.append((msd2,form2,msd1,form1))
            wordpairs.append((form2,form1))
        a = align.Aligner(wordpairs, align_symbol = u'_', iterations=30)
        traindata3 = [(traindata3[i][0], traindata3[i][1], traindata3[i][2], traindata3[3], a.alignedpairs[i][0], a.alignedpairs[i][1]) for i in range(len(traindata3))]
        td = traindata3
        
    if task == 2:
        wordpairs = [(x[1],x[3]) for x in traindata2]
        a = align.Aligner(wordpairs, align_symbol = u'_', iterations=30)
        traindata2 = [(traindata2[i][0], traindata2[i][1], traindata2[i][2], traindata2[3], a.alignedpairs[i][0], a.alignedpairs[i][1]) for i in range(len(traindata2))]
        td = traindata2
    
    for msd1, form1, msd2, form2, form1aligned, form2aligned in td:
        if msd1 + '-' + msd2 not in formtoform:
            formtoform[msd1 + '-' + msd2] = []
        alignedpair1 = (form1aligned, form2aligned)
        choppedpair1 = singleinput(alignedpair1[0], alignedpair1[1])
        decision1 = decisionpair(choppedpair1[0], choppedpair1[1])
        formtoform[msd1 + '-' + msd2].append(zip(decision1[0], decision1[1]))

crffromlemma = {}
crftolemma = {}
crfformtoform = {}
# From lemma

if task == 1 or not constrained:
    sys.stderr.write("TRAINING lemma > form\n")
    for it in fromlemma:
        train_words = fromlemma[it]
        X_train = [word2features(zip(*s)[0]) for s in train_words]
        y_train = [word2labels(s) for s in train_words]        
        crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True, all_possible_states=True)
        crf.fit(X_train, y_train)
        crffromlemma[it] = crf

if task > 1 and not constrained:
    sys.stderr.write("TRAINING form > lemma\n")
    for it in tolemma:
        train_words = tolemma[it]
        X_train = [word2features(zip(*s)[0]) for s in train_words]
        y_train = [word2labels(s) for s in train_words]
        crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True, all_possible_states=True)
        crf.fit(X_train, y_train)
        crftolemma[it] = crf

if task >= 2 and constrained:
    sys.stderr.write("TRAINING form > form direct\n")
    for it in formtoform:
        train_words = formtoform[it]
        X_train = [word2features(zip(*s)[0]) for s in train_words]
        y_train = [word2labels(s) for s in train_words]
        crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True, all_possible_states=True)
        crf.fit(X_train, y_train)
        crfformtoform[it] = crf
    
if task == 1:
    if mode == 'd':
        devlines1 = [line.strip() for line in codecs.open(lang + "-task1-dev", "r", encoding="utf-8")]
    else:
        devlines1 = [line.strip()for line in codecs.open(lang + "-task1-test-covered", "r", encoding="utf-8")]
if task == 2:
    if mode == 'd':
        devlines2 = [line.strip() for line in codecs.open(lang + "-task2-dev", "r", encoding="utf-8")]
    else:
        devlines2 = [line.strip() for line in codecs.open(lang + "-task2-test-covered", "r", encoding="utf-8")]
if task == 3:
    if mode == 'd':
        devlines3 = [line.strip() for line in codecs.open(lang + "-task3-dev", "r", encoding="utf-8")]
    else:
        devlines3 = [line.strip() for line in codecs.open(lang + "-task3-test-covered", "r", encoding="utf-8")]

if task == 1:
    numcorrect = 0
    numguesses = 0
    for l in devlines1: # Solve task 1
        lemma, msd, correct = (l.split(u'\t') + [None])[:3]
        teststring = ['<'] + list(lemma) + ['>']
        if msd in crffromlemma:
            crf = crffromlemma[msd]
            stringfeats = word2features(teststring)
            prediction = crf.predict_single(stringfeats)
            guess = unfold(teststring, prediction).replace('<','').replace('>','')
        else:
            guess = lemma
        
        if guess == correct:
            numcorrect += 1
        numguesses +=1
        print (lemma + "\t" + msd + "\t" + guess).encode("utf-8")
    sys.stderr.write(lang[0:3] + " task1: " + str(numcorrect/float(numguesses))[0:6] + "\n")   

if task == 2:
    numcorrect = 0
    numguesses = 0    
    for l in devlines2: # Solve task2
        msd1, form1, msd2, form2 = (l.split(u'\t') + [None])[:4]
        if msd1 == msd2:
            guess = form1
        elif constrained and msd1 + '-' + msd2 in crfformtoform:
            crf = crfformtoform[msd1 + '-' + msd2]
            teststring = ['<'] + list(form1) + ['>']
            stringfeats = word2features(teststring)
            prediction = crf.predict_single(stringfeats)
            guess = unfold(teststring, prediction).replace('<','').replace('>','')

        elif not constrained and msd1 in crftolemma and msd2 in crffromlemma:
            crf = crftolemma[msd1]
            teststring = ['<'] + list(form1) + ['>']
            stringfeats = word2features(teststring)
            prediction = crf.predict_single(stringfeats)
            guess = unfold(teststring, prediction).replace('<','').replace('>','')

            crf = crffromlemma[msd2]
            teststring = ['<'] + list(guess) + ['>']
            stringfeats = word2features(teststring)
            prediction = crf.predict_single(stringfeats)
            guess = unfold(teststring, prediction).replace('<','').replace('>','')
        else:
            guess = form1
        
        if guess == form2:
            numcorrect += 1
        numguesses +=1
        print (msd1 + "\t" + form1 + "\t" + msd2 + "\t" + guess).encode("utf-8")
    if constrained:
        sys.stderr.write(lang[0:3] + " task2c: " + str(numcorrect/float(numguesses))[0:6] + "\n")
    else:
        sys.stderr.write(lang[0:3] + " task2u: " + str(numcorrect/float(numguesses))[0:6] + "\n")
        
if task == 3 and constrained:
    numcorrect = 0
    numguesses = 0    
    for l in devlines3: # Solve task3
        form1, msd2, form2 = (l.split(u'\t') + [None])[:3]
        thispos = re.search(r'pos=([^,]*)', msd2).group(1)
        msd1 = clfSVM[thispos].predict(vectorizer[thispos].transform([getsubstrings(form1)]))[0]  # Guess the MSD
        if msd1 == msd2:
            guess = form1
        elif msd1 + '-' + msd2 in crfformtoform:
            crf = crfformtoform[msd1 + '-' + msd2]
            teststring = ['<'] + list(form1) + ['>']
            stringfeats = word2features(teststring)
            prediction = crf.predict_single(stringfeats)
            guess = unfold(teststring, prediction).replace('<','').replace('>','')
        else:
            guess = form1
        
        if guess == form2:
            numcorrect += 1
        numguesses +=1
        print (form1 + "\t" + msd2 + "\t" + guess).encode("utf-8")
    sys.stderr.write(lang[0:3] + " task3c: " + str(numcorrect/float(numguesses))[0:6] + "\n")
    
    
if task == 3 and not constrained:
    numcorrect = 0
    numguesses = 0
    for l in devlines3:  # Solve task 3
        form1, msd2, form2 = (l.split(u'\t') + [None])[:3]
        thispos = re.search(r'pos=([^,]*)', msd2).group(1)
        msd1 = clfSVM[thispos].predict(vectorizer[thispos].transform([getsubstrings(form1)]))[0]  # Guess the MSD
        if msd1 == msd2:
            guess = form1
        elif msd1 in crftolemma and msd2 in crffromlemma:
            crf = crftolemma[msd1]
            teststring = ['<'] + list(form1) + ['>']
            stringfeats = word2features(teststring)
            prediction = crf.predict_single(stringfeats)
            guess = unfold(teststring, prediction).replace('<','').replace('>','')
   
            crf = crffromlemma[msd2]
            teststring = ['<'] + list(guess) + ['>']
            stringfeats = word2features(teststring)
            prediction = crf.predict_single(stringfeats)
            guess = unfold(teststring, prediction).replace('<','').replace('>','')
        else:
            guess = form1
        
        if guess == form2:
            numcorrect += 1
        numguesses +=1
        print (form1 + "\t" + msd2 + "\t" + guess).encode("utf-8")
    sys.stderr.write(lang[0:3] + " task3u: " + str(numcorrect/float(numguesses))[0:6] + '\n')

