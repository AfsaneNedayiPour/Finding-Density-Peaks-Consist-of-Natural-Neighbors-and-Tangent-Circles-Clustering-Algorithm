from sklearn.neighbors import KDTree
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as patches
from matplotlib.patches import Circle
from sympy import *
from heapq import nsmallest
from collections import namedtuple
from collections import Counter
from sympy.plotting import plot
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.multiclass import OneVsRestClassifier


def find_second_smallest(a: list) -> int:
    f1, f2 = float('inf'), float('inf')
    for i in range(len(a)):
        if a[i] <= f1:
            f1, f2 = a[i], f1
        elif a[i] < f2 and a[i] != f1:
            f2 = a[i]
    return f2


def r_tangent_circle(a, b, c):
    A = math.sqrt((b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2)
    B = math.sqrt((a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2)
    C = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    Ra = (B + C - A) / 2
    Rb = (C + A - B) / 2
    Rc = (A + B - C) / 2
    return [Ra, Rb, Rc]




def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


""" STEP ONE """
# load data
with open('spiral.txt') as f:
    points = [tuple(map(float, i.split('\t')[0:2])) for i in f]
idpoints=[]
for point in points:
    idpoints.append(points.index(point))


#NaN_Searching
tree = KDTree(points)
num_list = [1000, 1001]
r = 1
landa = len(points)

while r < landa:

    num = 0
    all_nn_indices = tree.query_radius(points, r)
    all_nns = [[idx for idx in nn_indices if idx != i] for i, nn_indices in enumerate(all_nn_indices)]

    for i, nns in enumerate(all_nns):

        nns_set = set(nns)
        rnns = [all_nns.index(nns) for nns in all_nns if i in nns]

        for Nb in range(len(rnns) == 0):
            num = num + 1
    num_list.append(num)

    if num_list[r] == num_list[r - 1]:

            break
    else:
        r = r + 1
        continue

num_list.remove(1000)
num_list.remove(1001)
landa = r-1
print("landa =", landa)
print(num_list)


def nn(input):
    r = input
    all_nn_indices = tree.query_radius(points, r)
    all_nns = [[idx for idx in nn_indices if idx != i] for i, nn_indices in enumerate(all_nn_indices)]
    rnns = [[all_nns.index(nns) for nns in all_nns if i in nns] for i, nns in enumerate(all_nns)]

    return rnns

n = 0
TT_NaN_list = []
for i, m in enumerate(nn(landa)):
    print(n)
    n = n + 1
    rnns_set = set(m)
    total_nan = rnns_set
    print("T_NaN = ", list(total_nan))
    print("Nb = ", len(m))
    TT_NaN_list.insert(i, m)


# finding two nearest neighbor of each point

nearest_NaN_sets=[]   #list of sets

prdes=[]  # contains points and source of each points radiuses

for j, tt in enumerate(TT_NaN_list):
    tt1 = list(tt)
    d = []
    prdes.append([j])
    for t in tt1:
        d.append(distance(points[j], points[t]))

    if len(tt) >=2:
        m1 = min(d)
        m2 = find_second_smallest(d)
        if m1 == m2:
            repeated = m2
            indx = [i for i in range(len(d)) if d[i] == repeated]
            set1 = {j, tt1[indx[0]], tt1[indx[1]]}
            nearest_NaN_sets.append(set1)
        else:
            set1 = {j, tt1[d.index(m1)], tt1[d.index(m2)]}
            nearest_NaN_sets.append(set1)

nearest_NaN_lists=[]    # lists of each point's two nearest neighbors
for s1, sets in enumerate(nearest_NaN_sets):

    nearest_NaN_lists.append(list(sets))


# computing radiuses
circles = []
for h, d4 in enumerate(nearest_NaN_lists):
 if len(d4)==3:
    r3 = r_tangent_circle(points[d4[0]], points[d4[1]], points[d4[2]])
    r4 = (r3[0] + r3[1] + r3[2]) / 3

    if r3[0] and r3[1] and r3[2] != 0:
        r01 = r3[0] / r3[1]
        r02 = r3[0] / r3[2]
        r10 = r3[1] / r3[0]
        r12 = r3[1] / r3[2]
        r20 = r3[2] / r3[0]
        r21 = r3[2] / r3[1]

        if r01 < 0.46 or r02 < 0.46:
            prdes[d4[0]].append([r4] + [nearest_NaN_lists.index(d4)])
        else:
            prdes[d4[0]].append([r3[0]] + [nearest_NaN_lists.index(d4)])

        if r10 < 0.46 or r12 < 0.46:
            prdes[d4[1]].append([r4] + [nearest_NaN_lists.index(d4)])
        else:
            prdes[d4[1]].append([r3[1]] + [nearest_NaN_lists.index(d4)])

        if r20 < 0.46 or r21 < 0.46:
            prdes[d4[2]].append([r4] + [nearest_NaN_lists.index(d4)])
        else:
            prdes[d4[2]].append([r3[2]] + [nearest_NaN_lists.index(d4)])

qq = 0
rs = [] # contains minimum radius of points

not_TT_radius=[]

for de, dus in enumerate(prdes):
    if len(dus)==1:
        not_TT_radius.append(dus[0])

    ttt = TT_NaN_list[dus[0]]
    qq = qq + len(dus)
    les = []

    le = len(dus)
    if le >= 1:
        for ee in range(1, le):
            les.append(dus[ee][0])

    if les == []:
        ra = 0
    else:
        ra = min(les)
        rs.append((de, ra))

    dus.remove(dus[0])
    circle03 = plt.Circle(points[de], ra, color='orange', fill=False)
    plt.gcf().gca().add_artist(circle03)
    circles.append(circle03)


""" STEP Two """
srs = sorted(rs, key=lambda tup: tup[1])  # sort points by radius
#print("sort by radius", srs)


srs_id=[]     # only sorted points indexes
for to, top in enumerate(srs):
    srs_id.append(top[0])

print("srs_id",srs_id)
not_delet=[]
deleted=[]
mylist=[]
mylist_count1=[]
mylist_eq = [0]

smeq = 0
for to, top in enumerate(srs_id):
    if mylist_eq[smeq] - mylist_eq[smeq - 1] < (1/10)*len(points):
      ml = 0
      set1 = set(TT_NaN_list[top])

      for tos in range(0, to):

                  if srs_id[tos] not in deleted:
                      if top not in deleted:
                          sb = TT_NaN_list[srs_id[tos]]
                          disb1 = set1.intersection(set(sb))

                          if disb1 != set():

                              deleted.append(top)
                              ml = ml + 1
                              print(to, ml,smeq)


                              print(mylist_eq)
                          if srs_id[tos] not in not_delet:

                           not_delet.append(srs_id[tos])

      mylist.append(ml)
      mylist_count1.append(mylist.count(1))


      if to >=1:
          if mylist_count1[to] == mylist_count1[to-1]:
              if mylist_count1[to] and mylist_count1[to-1]!=0:
                  mylist_eq.append(mylist_count1[to])
                  smeq=smeq+1


remained = not_delet.copy()
sort_centers=sorted(remained)
print("sort_centers",sort_centers)

""" STEP Three """
#clustering
cens = []
xcens = []
ycens = []

allocate=[] #assigned points to clusters
xcentrs=[]
ycentrs=[]
for ci, cc in enumerate(sort_centers):
    allocate.append(cc)
    cens.append([cc])
    xcens.append([points[cc][0]])
    ycens.append([points[cc][1]])
    xcentrs.append([points[cc][0]])
    ycentrs.append([points[cc][1]])
halo = [] #not allocated points
xhalo = []
yhalo = []


for point in points:
    if points.index(point) not in allocate:
        halo.append(points.index(point))
        xhalo.append(point[0])
        yhalo.append(point[1])

    for cs in range(0, len(sort_centers)):

        if points.index(point) in TT_NaN_list[cens[cs][0]]:
            allocate.append(points.index(point))
            cens[cs].append(points.index(point))
            xcens[cs].append(point[0])
            ycens[cs].append(point[1])




if len(allocate)!=len(points):
 while len(halo)!=0:
  for hs in halo:
     mdh = []
     for nh in allocate:
       mdh.append(distance(points[hs], points[nh]))
     for ps in cens:
        if set(TT_NaN_list[hs]).intersection(set(ps)) != set():
          if allocate[mdh.index(min(mdh))] in ps:

              if hs not in allocate:
                allocate.append(hs)
              halo.remove(hs)
              ps.append(hs)
              xcens[cens.index(ps)].append(points[hs][0])
              ycens[cens.index(ps)].append(points[hs][1])

""" STEP FOUR """
nij=[]
for mo in cens:
    nij.append(1 / 10 * len(mo))
mnij=min(nij).__round__()



minm = [] # min distances for remained points
distanceci=[]
max1=[]
NN=[]
SNN=[]
ws_list=[]

for mo in cens:
    mm = []
    mmd = []
    set0= set(mo)
    mo_less=(0.1*len(mo)).__round__()
    for ci1 in mo:
         for ci2 in mo:
             if ci1!=ci2:
              mm.append(distance(points[ci1],points[ci2]))

    ws=sum(mm)/(len(mo)-1)
    ws_list.append(ws)
    for mo2 in cens:
        if mo!=mo2:


            NN.append((len(mo)+len(mo2),) +(cens.index(mo),cens.index(mo2)))

            set2= set(mo2)
            inter1= set0.intersection(set2)
            if inter1!=set():
                SNN.append((len(list(inter1)),) +(cens.index(mo),cens.index(mo2)))

ws_list2=[]
for scs in cens:
    dist1=[]
    for so1 in sort_centers:
        if so1 in scs:
            for sc3 in scs:
                if so1!=sc3:
                    dist1.append(distance(points[so1],points[sc3]))
    ws_list2.append(sum(dist1)/(len(scs)-1))


sim_list=[]
for sm in range(len(NN)):
    dij=distance(points[sort_centers[NN[sm][1]]],points[sort_centers[NN[sm][2]]])
    sim=NN[sm][0]/dij
    sim_list.append((sim,)+(NN[sm][1:3]))
sort_sim=sorted(sim_list,key=lambda tup: tup[0],reverse=True)


wsij=[]
wsij2=[]

for cen1 in cens:
  mind1 = []
  mind2=[]
  for so2 in sort_centers:

       if so2 not in cen1:
           mmsc = []
           for sc1 in cen1:

                   mmsc.append(distance(points[sc1],points[so2]))

           mind1.append((sum(mmsc)/len(cen1),)+ (cens.index(cen1),sort_centers.index(so2)))
           mind2.append((cens.index(cen1),sort_centers.index(so2)))
  wsij.append(min(mind1))
  for mn in mind1:
      wsij2.append(mn)





allsets=[]
for sm3 in sort_sim:
     if ws_list2[sm3[1]] + ws_list2[sm3[2]]< wsij2[sim_list.index(sm3)][0]:
        newc=cens[sm3[1]]+cens[sm3[2]]
        newsc=[sort_centers[sm3[1]]]+[sort_centers[sm3[2]]]

        allsets.append(set(newsc))

def remove_duplicates(lst):
    res = []
    for x in lst:
        if x not in res:
            res.append(x)
    return res

merg1=[]
for ww in wsij:
    if ws_list2[ww[1]] + ws_list2[ww[2]]< wsij[ww[1]][0]:
        newc=cens[ww[1]]+cens[ww[2]]
        newsc=[sort_centers[ww[1]]]+[sort_centers[ww[2]]]
        merg1.append(set(newsc))

setmerg1=remove_duplicates(merg1)



merg2=[]
dism2=[]
len_nn=[]
new_cen=[]
list_new_cen=[]
for sm4 in sim_list:

    newsc = [sort_centers[sm4[1]]] + [sort_centers[sm4[2]]]
    list_new_cen.append(newsc)
    for sm5 in sim_list:
     if wsij2[sim_list.index(sm4)][1] == wsij2[sim_list.index(sm5)][2]:
         if wsij2[sim_list.index(sm5)][1] == wsij2[sim_list.index(sm4)][2]:
            if set(newsc) in setmerg1:
             sum4= (wsij2[sim_list.index(sm4)][0]+ wsij2[sim_list.index(sm5)][0])/2
             merg2.append(set(newsc))
             dism2.append(math.floor(sum4))
             len_nn.append(len(TT_NaN_list[newsc[0]]))
             len_nn.append(len(TT_NaN_list[newsc[1]]))
             if sort_centers[sm4[1]] not in new_cen:
              new_cen.append(sort_centers[sm4[1]])
             if sort_centers[sm4[2]] not in new_cen:
                 new_cen.append(sort_centers[sm4[2]])

setmerg2=remove_duplicates(merg2)
dism3=remove_duplicates(dism2)
merg3=merg2.copy()



if dism3 !=[]:
 mdm3 = min(dism3)
 for sm6 in merg2:
     if dism2[merg2.index(sm6)]> mdm3+1:
         merg3.remove(sm6)
setmerg3=remove_duplicates(merg3)

print("new_cen",new_cen)

setmerg4 = []


merged1=[]
merged2=[]
for nc1 in new_cen:

    for sm7 in setmerg3:
        for sm8 in setmerg3:
          if sm7!=sm8:
           if nc1 in sm7:

            if nc1 in sm8:

               lst=list(sm7.union(sm8)-{nc1})
               if nc1 not in merged1:
                  merged1.append(nc1)
                  merged2.append(nc1)




               if len(TT_NaN_list[lst[0]]) == max(len_nn):
                   lst.remove(lst[1])
                   lst.append(nc1)
                   if set(lst) not in setmerg4:
                       setmerg4.append(set(lst))
                       merged2.append(lst[0])
                       merged2.append(lst[1])

               if len(TT_NaN_list[lst[1]]) == max(len_nn):
                   lst.remove(lst[0])
                   lst.append(nc1)
                   if set(lst) not in setmerg4:
                       setmerg4.append(set(lst))
                       merged2.append(lst[0])
                       merged2.append(lst[1])
               if abs(len(TT_NaN_list[nc1]) - len(TT_NaN_list[lst[0]]))> abs(len(TT_NaN_list[nc1]) - len(TT_NaN_list[lst[1]])):

                    if abs(len(TT_NaN_list[nc1]) - len(TT_NaN_list[lst[1]]))!=0:
                        lst.remove(lst[0])
                        lst.append(nc1)
                        if set(lst) not in setmerg4:
                          setmerg4.append(set(lst))
                          merged2.append(lst[0])
                          merged2.append(lst[1])



               if abs(len(TT_NaN_list[nc1]) - len(TT_NaN_list[lst[0]])) < abs(len(TT_NaN_list[nc1]) - len(TT_NaN_list[lst[1]])):
                   if abs(len(TT_NaN_list[nc1]) - len(TT_NaN_list[lst[0]]))!=0:
                       lst.remove(lst[1])
                       lst.append(nc1)
                       if set(lst) not in setmerg4:
                          setmerg4.append(set(lst))
                          merged2.append(lst[0])
                          merged2.append(lst[1])




print("mer2",merged2)
setmerg5 = []
setmerg51=[]

for set4 in setmerg4:
  for set40 in setmerg4:
    if set4!=set40:
        if set4.intersection(set40)!=set():
           if set4.union(set40) not in setmerg51:
              setmerg51.append(set4.union(set40))




              for ss5 in range(len(setmerg51)):

                       if setmerg51[ss5].intersection(setmerg51[ss5-1])!=set():
                          setmerg5.clear()
                          setmerg5.append(setmerg51[ss5].union(setmerg51[ss5-1]))




        else:
            if setmerg51==[]:
               setmerg5.append(set40)


    if len(setmerg4)==1:
        setmerg5.append(set4)





print("setmerg5",setmerg5)
tak1=[]
tanha=[]
for sm9 in setmerg3:
    for sm10 in setmerg4:
      if sm9 != sm10:


        if sm9 not in setmerg4:

            if list(sm9)[0]  not in merged1:
             tak1.append(sm9)
             tanha.append(list(sm9)[0])
             merged2.append(list(sm9)[0])
            if list(sm9)[1] not in merged1:
                tak1.append(sm9)
                tanha.append(list(sm9)[1])
                merged2.append(list(sm9)[1])



tak2 = remove_duplicates(tak1)
tak21=tak2.copy()
setmerg6=[]
setmerg61=[]
setmerg5_copy=remove_duplicates(setmerg5).copy()
for set5 in remove_duplicates(setmerg5):
    for tk in tak2:
        if tk.intersection(set5) != set():

            setmerg61.append(set5.union(tk))
            if len(TT_NaN_list[list(tk.intersection(set5))[0]]) < len(TT_NaN_list[list(tk.copy() - tk.intersection(set5))[0]]):

                merged2.remove(list(tk.copy()-tk.intersection(set5))[0])
                sd=set5.union(tk)-(tk.copy()-tk.intersection(set5))
                setmerg61.remove(set5.union(tk))
                setmerg61.append(sd)

            tak21.remove(tk)
            if setmerg5_copy != []:
               setmerg5_copy.remove(set5)

            for ss6 in range(len(setmerg61)):

                 if setmerg61[ss6].intersection(setmerg61[ss6 - 1]) != set():
                     setmerg6.clear()
                     setmerg6.append(setmerg61[ss6].union(setmerg61[ss6 - 1]))





last_grop= setmerg6+tak21+setmerg5_copy
print("last merged",last_grop)


new_cen2=[]
for n1 in sort_centers:
    if n1 not in remove_duplicates(merged2):
     new_cen2.append({n1})

ls=new_cen2+last_grop

#clustering new
last_cens=[]
alocated2=[]
x1=[]
y1=[]
x1centrs=[]
y1centrs=[]
for ls1 in ls:
   last_cens.append(list(ls1))

   x1.append([])
   y1.append([])
   x1centrs.append([])
   y1centrs.append([])
   for ls2 in list(ls1):
     alocated2.append(ls2)

print(last_cens)

alocated3=[]
last_cens1=last_cens.copy()

for point in points:
            if points.index(point) in alocated2:
               for lcc in range(0, len(last_cens)):
                   if points.index(point) in last_cens[lcc]:
                       x1[lcc].append(point[0])
                       y1[lcc].append(point[1])
                       x1centrs[lcc].append(point[0])
                       y1centrs[lcc].append(point[1])
            if points.index(point) not in alocated2:
                for lc1 in alocated2:
                    if points.index(point) in TT_NaN_list[lc1]:
                        for lcc in range(0,len(last_cens)):
                            if lc1 in last_cens[lcc]:
                                last_cens1[lcc].append(points.index(point))
                                alocated3.append(points.index(point))
                                x1[lcc].append(point[0])
                                y1[lcc].append(point[1])


t_alocated=alocated2+alocated3
naloc=list(set(idpoints)-set(t_alocated))
print(len(t_alocated),len(naloc),len(points))
if len(t_alocated)!=len(points):
 while len(naloc)!=0:
  for na in naloc:
     mdna = []
     for ta in t_alocated:
       mdna.append(distance(points[na], points[ta]))
     for ls3 in last_cens1:
        if set(TT_NaN_list[na]).intersection(set(ls3)) != set():
          if t_alocated[mdna.index(min(mdna))] in ls3:
              if na not in t_alocated:
                t_alocated.append(na)
              naloc.remove(na)
              ls3.append(na)
              x1[last_cens1.index(ls3)].append(points[na][0])
              y1[last_cens1.index(ls3)].append(points[na][1])


print("lastcens",last_cens)
print(len(t_alocated),len(naloc),len(points))


"""ploting"""
plt.scatter(*zip(*points), s=0.1)
for i, txt in enumerate(points):
    plt.annotate(i, points[i])
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('agg15.tif')
plt.show()



tplecenter=[]
stcens=[]
for kh in range(0,len(sort_centers)):
    tplecenter.append(points[sort_centers[kh]])
    stcens.append((sort_centers[kh],))
print("stcens",stcens)
sizes=[40]
plt.figure(dpi=150)
styles = ['lightgreen', 'salmon', 'lightblue', 'purple', 'b', 'm', 'y', '#9400D3', '#C0FF3E', 'violet', 'black','green']
plt.scatter(xhalo, yhalo, color='black', s=10)
for ls in range(0, len(sort_centers)):
    plt.scatter(xcens[ls], ycens[ls], color=styles[ls], s=10, label='cluster'+str(ls+1))
    plt.scatter(xcentrs[ls], ycentrs[ls], color='black', s=20)
    labels = stcens
    for xc1, yc1, label, size in zip(xcentrs[ls], ycentrs[ls], labels[ls], sizes):
        plt.annotate(label, (xc1, yc1), fontsize=20)


plt.scatter(xcentrs[0], ycentrs[0], color='black', s=10, label='cluster heads')

plt.legend(loc='upper right', fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('spiral031.tif')
plt.show()

sizes=[40]
plt.figure(dpi=300)
styles = ['lightgreen', 'salmon', 'lightblue', 'purple', 'b', 'm', 'y', '#9400D3', '#C0FF3E', 'violet', 'black','green']
plt.scatter(xhalo, yhalo, color='black', s=10)
for ls0 in range(0, len(last_cens1)):
    plt.scatter(x1[ls0], y1[ls0], color=styles[ls0], s=10, label='cluster'+str(ls0+1))
    plt.scatter(x1centrs[ls0], y1centrs[ls0], color='black', s=20)



plt.scatter(x1centrs[0], y1centrs[0], color='black', s=10, label='initial cluster heads')

plt.legend(loc='upper right', fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('spiral032.tif')
plt.show()

