import codecs


def load_pmi_dict(filename):
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        pmi_dict = {}
        for line in f:
            line_parts = line.split("\t")
            pmi_dict[line_parts[0]] = float(line_parts[3])
    return pmi_dict

def create_pmi_feature(lemmas, pmi_dict):
        features = []
        for lemma in lemmas:
            feature = [0] * 3
            lemma = lemma.split(" ")
            count = 0
            sum = 0
            max = 0
            min = 0
            for l in lemma:
                l = unicode(l, 'utf-8')
                if l in pmi_dict:
                    pmi = pmi_dict[l]
                    count += 1
                    sum += pmi
                    if max == 0:
                        max = pmi
                    elif max < pmi:
                        max = pmi
                    if min == 0:
                        min = pmi
                    elif min > pmi:
                        min = pmi
            if count == 0:
                count = 1
            feature[0] = sum
            feature[1] = max
            feature[2] = min
            features.append(feature)
        return features