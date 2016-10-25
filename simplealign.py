import codecs, sys, align

for lang in sys.argv[1:]:
    lines = [line.strip() for line in codecs.open(lang + "-task1-train", "r", encoding="utf-8")]
    traindata = []
    for l in lines:
        lemma, msd, form = l.split(u'\t')
        traindata.append((lemma, form, msd))
    
    wordpairs = [(x[0],x[1]) for x in traindata]
    a = align.Aligner(wordpairs, align_symbol = u' ', iterations=30)
    traindata = [(traindata[i][0], traindata[i][1], traindata[i][2], a.alignedpairs[i][0], a.alignedpairs[i][1]) for i in range(len(traindata))]

    # we now have a list like:
    #[(lemma,form,msd,lemmaaligned,formaligned), ...]
    
    for lemma, form, msd, lemmaaligned, formaligned in traindata:
        print lemmaaligned
        print formaligned
        print

