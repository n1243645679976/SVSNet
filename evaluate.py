import argparse
import numpy as np
import scipy.stats as stats

parser = argparse.ArgumentParser()
parser.add_argument('tar', help='target output file')
parser.add_argument('dataset',  help='dataset, vcc2018 or vcc2020')
args = parser.parse_args()

def app(dic, key, tup):
    if key not in dic:
        dic[key] = [tup]
    else:
        dic[key].append(tup)
def app_dic(dic, key, subkey, tup):
    if key not in dic:
        dic[key] = {}
    if subkey not in dic[key]:
        dic[key][subkey] = [tup]
    else:
        dic[key][subkey].append(tup)
r = {}
r1 = {}
r2 = {}
r3 = {}
r4 = {}
r5 = {}
a3 = []
a4 = []
n=0
with open('{}'.format(args.tar)) as f:
    for a in f.readlines():
        sc = float(a.split(',')[-1])
        la = float(a.split(',')[-2])
        sr, ta = a.split()[0:2]
        if args.dataset == 'vcc2018':
            sys = sr[:3]
        else:
            sys = sr.split('-')[0]

#        if ta[0] == 'S':
#            temp = sr
#            sr = ta
#            ta = sr
#        if ta[0] == 'S':
#            continue
#        if sr[0] == 'S' and ta[0] != 'T':
#            continue
        n += 1
        app(r1, sys, (sr, ta, sc))
        app(r, sys, (sr, ta, la))
        if sr > ta:
            app_dic(r2, sys, sr+ta, (sr,ta,sc))
            app_dic(r3, sys, sr+ta, (sr,ta,la))
        else:
            app_dic(r2, sys, ta+sr, (sr,ta,sc))
            app_dic(r3, sys, ta+sr, (sr,ta,la))


        a3.append(float(sc))
        a4.append(float(la))
#        print(sr, ta)
a1 = []
a2 = []
a5 = []
a6 = []
a7 = []
a8 = []
for key in r.keys():
    assert len(r[key]) == len(r1[key])
    for i in range(len(r[key])):
        assert r[key][i][0] +r[key][i][1] == r1[key][i][0] + r1[key][i][1]
    sr1 = np.mean([float(_y) for _, _, _y in r[key]])
    sr2 = np.mean([float(_y) for _, _, _y in r1[key]])
    a1.append(sr1)
    a2.append(sr2)
    for subkey in r2[key].keys():
        a5.append(np.mean([v[2] for v in r2[key][subkey]]))
        a6.append(np.mean([v[2] for v in r3[key][subkey]]))

        app(r4, key, np.mean([v[2] for v in r2[key][subkey]]))
        app(r5, key, np.mean([v[2] for v in r3[key][subkey]]))
    a8.append(np.mean(r5[key]))
    a7.append(np.mean(r4[key]))

#print(a1, a2)
#print('{:.3f} {:.3f} {:.3f} {:.3f}'.format(np.mean(np.clip(np.round(a3),0,3) == np.clip(np.round(a4),0,3)), np.corrcoef(a3, a4)[0][1], stats.spearmanr(a3, a4)[0], np.mean([(_a1-_a2)**2 for _a1, _a2 in zip(a3, a4)])))
#print('{:.3f} {:.3f} {:.3f}'.format(np.corrcoef(a1, a2)[0][1], stats.spearmanr(a1, a2)[0], np.mean([(_a1-_a2)**2 for _a1, _a2 in zip(a1, a2)])))
print('Utterance level: LCC={:.3f} SRCC={:.3f} MSE={:.3f}'.format(np.corrcoef(a5, a6)[0][1], stats.spearmanr(a5, a6)[0], np.mean([(_a1-_a2)**2 for _a1, _a2 in zip(a5, a6)])))
print('System level: LCC={:.3f} SRCC={:.3f} MSE={:.3f}'.format(np.corrcoef(a7, a8)[0][1], stats.spearmanr(a7, a8)[0], np.mean([(_a7-_a8)**2 for _a7, _a8 in zip(a7, a8)])))
