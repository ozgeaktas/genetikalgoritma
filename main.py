import pandas as pd
import numpy as np
import random as rd
from matplotlib import pyplot as plt
from math import sqrt
import tsplib95

with open('berlin52.txt') as f:
    problem = tsplib95.read(f)

#print('Problem adı: ' + problem.name)
#print('Şehir sayısı: ' + str(problem.dimension))

nodes = list(problem.node_coords.values())

# print(nodes) #koordinatları gösterdik.

# Bu koordinatların X ve Y lerine ihtiyacımız var.

node_x = []
node_y = []

for i in nodes:
    node_x.append(i[0])
    node_y.append(i[1])

# print(node_x)
# print(node_y)

# Bunları grafik üzerinde göstermek için:

plt.scatter(node_x, node_y, marker='s', s=5)
#plt.show()

# Bu noktalar arasındaki mesafeyi hesaplamalıyız.

a = [0, 0]
b = [3, 4]

distance = sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# print(distance)

def euclidean_dist(a, b):
    ans = sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return ans


#print(euclidean_dist(a, b))

# Şehirler arası koordinatlar arasındaki uzaklığın hesaplanması.
# Bunu yapmak için ilk olarak referans noktası belirlemeliyiz.

temp = []  # boş liste oluşturduk.

for i in nodes:
    temp_row = []
    for j in nodes:
        temp_row.append(euclidean_dist(i, j))  # Burda 1. noktadan 52. noktaya kadar olan satır oluşturuyoruz.
    temp.append(temp_row)

# print(temp)


# Dataframe kullanarak bu işlemi yapmak için :

# Dist=pd.DataFrame(temp)
# print(Dist)

# Dataframe 0 dan 51 e kadar gidiyor. Bunu istemiyoruz.

node_name = []

for i in range(1, 53):
    node_name.append(i)

# print(node_name)

Dist = pd.DataFrame(temp, columns=node_name, index=node_name)  # Tüm ikili çiftlerin uzaklıkları hazırlanmış oldu.
#print(Dist)

###### GENETİK ALGORİTMA PARAMETRELERİNİN BELİRLENMESİ

# popülasyon sayısı -> 2*kromozom uzunluğu
P_size = 100

# Nesil sayısı -> yaklaşık 500 nesil ilerleyeceğiz.
N_gen = 500

# Çaprazlama olasılığı
p_crossover = 0.9

# Mutasyon olasılığı
p_mutation = 0.1

# Turnuva büyüklüğü -> ebevyn oluşturmak için
K = 5

# Elitzm ->Her jenarasyondan en iyi vkaç çözümü bir sonraki nesile aktaralım.
E = 10

###AMAÇ FONKSİYONUNUN OLUŞTURULMASI ->Noktalar gezinirken ki mesafeyi minimuma indirmek.. mesafeleri topluyoruz.

Known_Best = 7544.3659

rnd_sol = node_name


def objective(rnd_sol):
    obj = 0

    for i in range(len(rnd_sol)):
        Start_node = rnd_sol[i]

        if i + 1 == len(rnd_sol):
            End_node = rnd_sol[0]
        else:
            End_node = rnd_sol[i + 1]

        obj += Dist[Start_node][End_node]

    return obj


# print(objective(rnd_sol))
def initialize():
    Loc_Set = list(Dist.columns)
    Pop_list = []
    for i in range(P_size):
        rnd_sol = rd.sample(Loc_Set, len(Loc_Set))
        Pop_list.append((rnd_sol, objective(rnd_sol)))
    return Pop_list

#print(initialize())

Pop_list=initialize()

#Elitizm tanımlama ---- en iyi çözümü bir sonraki nesile aktarma işlemidir. Her jenerasyondaki en iyi çözümü bir sonraki jenerasyona aktarmak istiyoruz.

def elitism(Pop_list):
    Pop_list_ordered = sorted(Pop_list, key=lambda x: x[1])  # Pop list teki amaç fonksiyonlarını sıralıyoruz önce.

    #print(Pop_list_ordered)

    Elit_list = []
    i = 0
    while len(Elit_list) < E:
        solution = Pop_list_ordered[i][0]  # i numaralı tuple ın ilk üyesini solution a atar.
        Elit_add = (solution, objective(solution))
        if Elit_add not in Elit_list:
            Elit_list.append(Elit_add)
        i += 1
    return Elit_list


# EBEVEYN SEÇİNİ -TURNUVA SEÇİM YÖNTEMİ İLE
#  Turnuva listesi oluşturulur bu listenin kazananı ebeveyn olarak seçilir.
#Popülasyondan rastgele K çözüm seçip bu çözümler arasından en iyi olanları ebeveyn olarak seçmektedir.

def selection_op():
    parents = []

    while len(parents) < 2:
        tournament_selection_pool = []

        while len(tournament_selection_pool) < K:
            index = np.random.randint(0, len(Pop_list))

            if Pop_list[index] not in tournament_selection_pool:
                tournament_selection_pool.append(Pop_list[index])

        tournament_selection_pool_ordered = sorted(tournament_selection_pool, key=lambda x: x[1])

        if tournament_selection_pool_ordered[0] not in parents:
            parents.append(tournament_selection_pool_ordered[0])

    return parents



# CROSSOVER --- sonucunda çocuklar üretilecek.


def crossover_op(parents):
    Childs = []

    parents = selection_op()
    P1 = parents[0][0]
    P2 = parents[1][0]

    param = len(P1) * 0.20

    min_c = param
    max_c = len(P1) - (param - 1)

    min_c = param
    max_c = len(P1) - (param - 1)

    co_point_1 = np.random.randint(min_c, max_c)
    co_point_2 = np.random.randint(min_c, max_c)

    P1_seg_1 = P1[0:co_point_1]
    P1_seg_2 = P1[co_point_1:len(P1)]

    P2_seg_1 = P1[0:co_point_2]
    P2_seg_2 = P1[co_point_2:len(P2)]

    temp_1_seg = list(P2)
    temp_2_seg = list(P1)

    ### BİRİNCİ ÇOCUK
    op_rand = np.random.rand()

    if op_rand < 0.5:

        for i in range(len(P1_seg_1)):
            temp_1_seg.remove(P1_seg_1[i])

        Child_1 = P1_seg_1 + temp_1_seg
    else:
        for i in range(len(P1_seg_2)):
            temp_1_seg.remove(P1_seg_2[i])

        Child_1 = temp_1_seg + P1_seg_2

    Childs.append((Child_1, objective(Child_1)))

    ### İKİNCİ ÇOCUK

    op_rand = np.random.rand()

    if op_rand < 0.5:

        for i in range(len(P2_seg_1)):
            temp_2_seg.remove(P2_seg_1[i])

        Child_2 = P2_seg_1 + temp_2_seg
    else:
        for i in range(len(P2_seg_2)):
            temp_2_seg.remove(P2_seg_2[i])

        Child_2 = temp_2_seg + P2_seg_2

    Childs.append((Child_2, objective(Child_2)))
    return Childs


### MUTASYON

## INSERTION MUTATION

def mutation_op_1(mutation_cand):
  #  mutation_cand = Pop_list[0][0] #Sadece rotadan oluşuyor.

    ran_1 = np.random.randint(0, len(mutation_cand))
    ran_2 = np.random.randint(0, len(mutation_cand))

    while ran_1 == ran_2:
        ran_2 = np.random.randint(0, len(mutation_cand))

    x = mutation_cand[ran_1]
    mutated = list(mutation_cand)

    mutated.remove(x)

    mutated.insert(ran_2, x)

    return mutated

#print(mutation_cand)
#print(mutation_cand[ran_1])
#print(mutated)

### SWAP MUTATION

def mutation_op_2(mutation_cand):
    mutation_cand = Pop_list[0][0]
    ran_1 = np.random.randint(0, len(mutation_cand))
    ran_2 = np.random.randint(0, len(mutation_cand))

    while ran_1 == ran_2:
        ran_2 = np.random.randint(0, len(mutation_cand))

    x = mutation_cand[ran_1]
    y = mutation_cand[ran_2]

    mutated = list(mutation_cand)

    mutated[ran_1] = y
    mutated[ran_2] = x

    return mutated


### 2-OPT MUTATION
def mutation_op_3(mutation_cand):
    mutation_cand = Pop_list[0][0]

    ran_1 = np.random.randint(0, len(mutation_cand))
    ran_2 = np.random.randint(0, len(mutation_cand))

    while abs(ran_1 - ran_2) < 3:
        ran_2 = np.random.randint(0, len(mutation_cand))

    mutated = list(mutation_cand)

    if ran_1 < ran_2:
        mutated[ran_1:ran_2] = reversed(mutated[ran_1:ran_2])
    else:
        mutated[ran_2:ran_1] = reversed(mutated[ran_2:ran_1])

    return mutated


### SHUFFLE MUTATION
def mutation_op_4(mutation_cand):
    ran_1 = np.random.randint(0, len(mutation_cand))
    ran_2 = np.random.randint(0, len(mutation_cand))

    while abs(ran_1 - ran_2) < 3:
        ran_2 = np.random.randint(0, len(mutation_cand))

    mutated = list(mutation_cand)

    if ran_1<ran_2:
        seg = mutated[ran_1:ran_2]
        seg_mod = rd.sample(seg, len(seg))
        while seg == seg_mod:
            seg_mod = rd.sample(seg, len(seg))

        mutated[ran_1:ran_2] = seg_mod
    else:
        seg = mutated[ran_2:ran_1]
        seg_mod = rd.sample(seg, len(seg))
        while seg == seg_mod:
            seg_mod = rd.sample(seg, len(seg))

        mutated[ran_2:ran_1] = seg_mod

    return mutated


### gerekli ön tanımlamalar
### ilk popülasyonun yaratılması
### nesiller için loop
### ebeveyn seçimi
### çaprazlama ve çocuk üretimi
### mutasyon
### elitzim
### sonuçların görselleştirilmesi ve raporlanması

Best_Solutions = []
Best_Objectives = []
Best_Ever_Solution = []
Avg_Objectives=[]


Pop_list = initialize()


### Jenerasyon-0 için

Pop_list_ordered = sorted(Pop_list, key=lambda x: x[1])

Best_Solutions.append(Pop_list_ordered[0][0])
Best_Objectives.append(Pop_list_ordered[0][1])

Best_Ever_Solution = ((Pop_list_ordered[0][0] , Pop_list_ordered[0][1] , 0))

mean=sum(map(lambda x: x[1],Pop_list)) / len(Pop_list)

Avg_Objectives.append(mean)

for i in range(1, N_gen + 1):


    New_gen_Pop_list = []

    for c in range(int((P_size - E) / 2)):

        Childs = []
        parents = selection_op()

        rnd = np.random.rand()

        if rnd < p_crossover:
            Childs = crossover_op(parents)
        else:
            Childs = parents

        New_gen_Pop_list = New_gen_Pop_list + Childs


    for p in range(len(New_gen_Pop_list)):

        mutation_cand = New_gen_Pop_list[p][0]

        rnd = np.random.rand()

        if rnd < p_mutation:

            rnd_m = np.random.rand()

            if rnd_m < 0.3:
                mutated = mutation_op_1(mutation_cand)
                New_gen_Pop_list[p] = ((mutated, objective(mutated)))
            elif rnd_m < 0.7:
                mutated = mutation_op_2(mutation_cand)
                New_gen_Pop_list[p] = ((mutated, objective(mutated)))
            elif rnd_m < 0.95:
                mutated = mutation_op_3(mutation_cand)
                New_gen_Pop_list[p] = ((mutated, objective(mutated)))
            else:
                mutated = mutation_op_4(mutation_cand)
                New_gen_Pop_list[p] = ((mutated, objective(mutated)))

        else:
            pass

    Elit_list = elitism(Pop_list)

    New_gen_Pop_list = New_gen_Pop_list + Elit_list

    Pop_list = list(New_gen_Pop_list)

    Pop_list_ordered = sorted(Pop_list, key=lambda x: x[1])

    Best_Solutions.append(Pop_list_ordered[0][0])
    Best_Objectives.append(Pop_list_ordered[0][1])

    if Pop_list_ordered[0][1] >= Best_Ever_Solution[1]:
        pass
    else:
        Best_Ever_Solution = (Pop_list_ordered[0][0], Pop_list_ordered[0][1], i)
    mean = sum(map(lambda x: x[1], Pop_list)) / len(Pop_list)
    Avg_Objectives.append(mean)




print("#####SOLUTION OUTPUT#####")

print('BEST SOLUTION         : ', Best_Ever_Solution[0])
print('COST                  :  ', Best_Ever_Solution[1])
print('Found at generation   : ', Best_Ever_Solution[2])
print("Known Best Solution   : ", Known_Best)
print("Gap                   : %.2f%%" % ((Best_Ever_Solution[1]-Known_Best)*100/Known_Best))

print("#### PARAMETERS ####")
print("Number of generation    : %s" % N_gen)
print("Population size         : %s" % P_size)
print("Probability of crossover: %.0f%%" % (p_crossover*100))
print("Probability of mutation : %.0f%%" % (p_mutation*100))
print("Tournament selection    : %s" % K)
print("Elitism selection       : %s" % E)

plt.plot(Best_Objectives)
#plt.plot(Avg_Objectives)
plt.title('Objectives', fontsize=20, fontweight='bold')
plt.xlabel('Generations',fontsize=20, fontweight='bold')
plt.ylabel('Cost',fontsize=20, fontweight='bold')
plt.show()

#### Best Ever Solution ####

route_node_x=[]
route_node_y=[]

for i in Best_Ever_Solution[0]:
    route_node_x.append(node_x[i-1])
    route_node_y.append(node_y[i-1])


route_node_x.append(node_x[Best_Ever_Solution[0][0]-1])
route_node_y.append(node_y[Best_Ever_Solution[0][0]-1])


plt.plot(route_node_x,route_node_y)
#plt.show()


#### Optimal Tour

with open('berlin52.opt.txt') as o:
    solution = tsplib95.read(o)

#print(solution.tours[0])
optimal_tour=solution.tours[0]

opt_node_x=[]
opt_node_y=[]

for i in optimal_tour:
    opt_node_x.append(node_x[i-1])
    opt_node_y.append(node_y[i - 1])

opt_node_x.append(node_x[optimal_tour[0]-1])
opt_node_y.append(node_y[optimal_tour[0]-1])


#plt.plot(opt_node_x,opt_node_y)
##plt.show()

fig , (best,opt)=plt.subplots(1 , 2, figsize=(10,5))

best.plot(route_node_x,route_node_y, c='r')
best.set_title("Best Solution Found="+str(Best_Ever_Solution[1]))
opt.plot(opt_node_x,opt_node_y)
opt.set_title('Optimal= '+str(objective(optimal_tour)))

plt.show()














