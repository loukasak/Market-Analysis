import util
import simulate
from util import Colours as c
from multiprocessing.pool import Pool
from numpy import mean, array, sqrt
from numpy.random import choice, random, shuffle, randint
from itertools import repeat, combinations
from os import listdir, cpu_count
from shutil import copytree
from matplotlib import pyplot as plt

glob_ranges = {}

btc_glob_ranges = {}

integer_globs = []

glob_range_types = {}

default_values = {}

run_all_functions = {}

precisions = {}

score_len = 8

Test_Files = None

div = c.X+' | '

free_cpus = 0    # set to however many cpu processors you want to be idle during evolution

def count_cpus():
    global cpus
    cpus = cpu_count() - free_cpus
    return

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================

def mutate(configuration, num_mutations):
    mutant = configuration.copy()                # copy to leave input configuration unchanged
    mutations = set()
    for _ in range(num_mutations):
        glob = choice(list(glob_ranges))
        mutation = get_mutation(glob)
        mutant[glob] = mutation
        mutations.add(glob)
    mutations = list(mutations)
    mutations = mutations[0] if len(mutations)==1 else mutations
    return mutant, mutations

def get_mutation(glob):
    high, low = glob_ranges[glob]
    mutation = (high - low)*random() + low
    mutation = int(mutation) if glob in integer_globs else round(mutation, 6)
    return mutation

def random_configuration():
    configuration = {glob:get_mutation(glob) for glob in glob_ranges}
    return configuration

def breed(parent1, parent2, num_children):
    Genes = list(parent1)
    children = []
    for _ in range(num_children):
        genes = Genes.copy()      # copy to preserve element order in Genes after shuffle
        shuffle(genes)
        L = len(genes)
        p1genes = genes[:L//2]
        child = {gene:(parent1[gene] if gene in p1genes else parent2[gene]) for gene in Genes}
        children.append(child)
    return children

def extract_results(results):
    scores, loser_stds = [], []
    for bals in results:
        scores.append(round(mean(bals), 3))
        loser_stds.append(round(compute_loser_std(bals), 3))
    return array(scores), array(loser_stds)

def compute_loser_std(scores, baseline=1000):
    losers = array([score for score in scores if score < baseline])
    return util.std(losers, m=baseline) if any(losers) else 0

def new_breeder_population(strategy, max_mutations=2, child_multiplier=1, tribal_convergence=10, village_parents=4):
    # create new directory from template
    folder = 'genetics/breeder/{}/'.format(strategy)
    pops = sorted([int(path[3:]) for path in listdir(folder)])
    current_pop = pops[-1] if any(pops) else 0
    new_pop_dir = folder+'pop{}/'.format(current_pop+1)
    copytree('genetics/breeder/temp', new_pop_dir)
    # set variables
    variables = {}
    variables['max_mutations'] = int(max_mutations)
    print(c.C+"Max Mutations set to {}".format(int(max_mutations))+c.X)
    variables['child_multiplier'] = int(child_multiplier)
    print(c.C+"Child Multiplier set to {}".format(int(child_multiplier))+c.X)
    variables['tribal_convergence'] = int(tribal_convergence)
    print(c.C+"Tribal Convergence set to {} generations".format(int(tribal_convergence))+c.X)
    variables['village_parents'] = int(village_parents)
    print(c.C+"Village Parents set to {}".format(int(village_parents))+c.X)
    # write variables to files
    V = util.json_to_dict(new_pop_dir+'variables.json')
    V.update(variables)
    util.dict_to_json(V, new_pop_dir+'variables.json')
    print(c.C+"Created {}".format(new_pop_dir)+c.X)
    return

def new_mutator_population(strategy='group', n_mutators=16, gens_per_cycle=100, n_survivors=10, n_survivor_children=3, file_type='normal'):
    # create new directory from template
    folder = 'genetics/mutator/{}/'.format(strategy)
    pops = sorted([int(path[3:]) for path in listdir(folder)])
    current_pop = pops[-1] if any(pops) else 0
    new_pop_dir = folder+'pop{}/'.format(current_pop+1)
    copytree('genetics/mutator/temp', new_pop_dir)
    variables = {}
    variables['n_mutators'] = int(n_mutators)
    print(c.C+"No. Mutators set to {}".format(int(n_mutators))+c.X)
    variables['gens_per_cycle'] = int(gens_per_cycle)
    print(c.C+"Gens Per Cycle set to {}".format(int(gens_per_cycle))+c.X)
    variables['n_survivors'] = int(n_survivors)
    print(c.C+"No. Survivors set to {}".format(int(n_survivors))+c.X)
    variables['n_survivor_children'] = int(n_survivor_children)
    print(c.C+"No. Survivor Children set to {}".format(int(n_survivor_children))+c.X)
    ft = file_type if file_type in ['normal', 'btc'] else 'normal'
    variables['file_type'] = ft
    print(c.C+"File Type set to {}".format(ft))
    # write variables to files
    V = util.json_to_dict(new_pop_dir+'variables.json')
    V.update(variables)
    util.dict_to_json(V, new_pop_dir+'variables.json')
    print(c.C+"Created {}".format(new_pop_dir)+c.X)
    return

def print_history(category, strategy, population):
    d = 'genetics/{}/{}/pop{}/history.json'.format(category, strategy, population)
    history = util.json_to_dict(d)
    history_type = get_history_type(history)
    perf_strings = get_performance_strings(history, history_type)
    gen_strings = get_gen_strings(history, history_type)
    best_string = c.M+get_best_string(history_type)
    best = 0
    for i, gen in enumerate(gen_strings):
        score = get_best_score(perf_strings[i])
        gen_str = c.C+gen
        col = util.red_or_green(score, best)
        score_str = col+util.fill_string(score, score_len, filler=0)
        string = gen_str+div+best_string+score_str
        print(string)
        best = util.binary_max(best, score)
    return

def plot_population(category, strategy, population, figscale=1):
    d = 'genetics/{}/{}/pop{}/history.json'.format(category, strategy, population)
    history = util.json_to_dict(d)
    history_type = get_history_type(history)
    perf_strings = get_performance_strings(history, history_type)
    best_perfs = [get_best_score(ps) for ps in perf_strings]
    gens = range(1, len(history)+1)
    p_gens, p_best_perfs = peak_perfs(gens, best_perfs)
    plt.rcParams["figure.figsize"] = (6*figscale,4*figscale)
    plt.plot(gens, best_perfs, label='True')
    plt.plot(p_gens, p_best_perfs, label='Peak')
    plt.grid(axis='y')
    plt.xlabel('Generations')
    plt.ylabel('Performance')
    plt.title('{}{} Population {}'.format(strategy[0].upper(), strategy[1:], population))
    plt.legend()
    plt.tight_layout()
    plt.show()
    return gens, best_perfs

def get_history_type(history):
    if "01_001" in history:
        return 'mutator'
    return 'normal' if 'perfs ' in history['001'] else 'tribal'

def get_performance_strings(history, history_type):
    if history_type in ['normal', 'mutator']:
        perf_strings = [info['perfs '] for info in history.values()]
    else:
        perf_strings = [''.join([info['perfs ']+' ' for info in tribe_info.values()]) for tribe_info in history.values()]
    return perf_strings

def get_gen_strings(history, history_type):
    if history_type == 'mutator':
        return ["CYCLE {} GENERATION {}".format(cygen.split('_')[0], cygen.split('_')[1]) for cygen in history]
    return ["GENERATION {}".format(gen) for gen in history]

def get_best_string(history_type):
    return "BEST MUTATOR: " if history_type == 'mutator' else "BEST DESCENDANT: "

def get_best_score(scores):
    return max([float(s) for s in scores.split()])

def peak_perfs(gens, perfs):
    gs, ps = [], []
    best = 0
    for i in range(len(gens)):
        if perfs[i] >= best:
            gs.append(gens[i])
            ps.append(perfs[i])
            best = perfs[i]
    if gens[-1] not in gs:
        gs.append(gens[-1])
        ps.append(ps[-1])
    return gs, ps

def compare_populations(pops=None, strats=None, peaked=False, figsize=1):
    pops = pops if pops else list(range(1,int(listdir('genetics/breeder/')[-2][-1])+1))
    pops = util.make_iterable(pops)
    strats = strats if strats else listdir('genetics/breeder/temp/')
    strats = list(util.make_iterable(strats))
    plt.rcParams["figure.figsize"] = (6*figsize,4*figsize)
    for pop in pops:
        d = 'genetics/breeder/pop'+str(pop)+'/'
        for strategy in listdir(d):
            history = util.json_to_dict(d+strategy+'/history.json')
            if not any(history) or strategy not in strats:
                continue
            history = regularise_history(history)
            if peaked:
                gens, scores = survivors(history)
                if gens[-1] < int(list(history)[-1]):
                    gens.append(int(list(history)[-1]))
                    scores.append(scores[-1])
            else:
                gens = [int(gen) for gen in list(history)]
                scores = [get_best_score(scores) for scores in history.values()]
            gens.insert(0, 0)
            scores.insert(0, 1000)
            plt.plot(gens, scores, label='Pop {} {}'.format(pop, strategy), linewidth=1)
    plt.xlabel('Generations')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
    return

def regularise_history(history):
    val = list(history.items())[0][1]
    if isinstance(val, str):
        return history
    return {gen:''.join(inner_hist.values()) for gen, inner_hist in history.items()}

def survivors(history):
    best = 0
    gens, Scores = [], []
    for gen, scores in history.items():
        local_best = get_best_score(scores)
        if local_best >= best:
            best = local_best
            gens.append(int(gen))
            Scores.append(local_best)
    return gens, Scores

def copy_best(from_strat, from_pop, to_strat, to_pop):
    # folders
    f1 = 'genetics/breeder/{}/pop{}/'.format(from_strat, from_pop)
    f2 = 'genetics/breeder/{}/pop{}/'.format(to_strat, to_pop)
    # fittest configuration
    fittest = util.json_to_dict(f1+'fittest.json')
    util.dict_to_json(fittest, f2+'fittest.json')
    # variables
    vars1 = util.json_to_dict(f1+'variables.json')
    vars2 = util.json_to_dict(f2+'variables.json')
    best = vars1['best_score']
    std = vars1['best_std']
    vars2['best_score'] = best
    vars2['best_std'] = std
    util.dict_to_json(vars2, f2+'variables.json')
    print(c.C+"COPIED {} POP {} VARIABLES TO {} POP {}".format(from_strat.upper(), from_pop, to_strat.upper(), to_pop)+c.X)
    return

# requires if __name__ == '__main__':
def run_winners(start_index=0, iterations=30, folder='configs'):
    with open('genetics/winners/finished.txt', 'w') as f:
        f.write('')
    def finished():
        with open('genetics/winners/finished.txt', 'r') as f:
            fin = f.read()
        return any(fin)
    def write_result(file, score):
        results = util.json_to_dict('genetics/winners/results.json')
        results.update({file:score})
        util.dict_to_json(results, 'genetics/winners/results.json')
        return
    d = 'genetics/winners/{}/'.format(folder)
    print(util.now_string_time_only())
    best = 0
    total_time = 0
    for file in simulate.test_files():
        times, _ = zip(*simulate.gen_file(file))
        total_time += times[-1] - times[0]
    days = total_time/86400000
    for winner in listdir(d)[start_index:]:
        if finished():
            print(c.Y+"RUN HALTED"+c.X)
            return
        start = util.now()
        print(winner[:-5]+'... ', end='')
        globs = util.json_to_dict(d+winner)
        try:
            bals = simulate.multiprocess_iterations(globs, iterations=iterations, mute=True)
        except Exception:
            bals = [0]
        p_gains = [util.percentage(bal, 1000) for bal in bals]
        total_p_gain = sum(p_gains)
        p_gain_per_day = total_p_gain/days
        score = round(p_gain_per_day, 3)
        time = round((util.now()-start)/60000,1)
        string = "| {} | {}m | {}".format(util.fill_string(score, 5, 0), time, util.now_string_time_only())
        best = util.binary_max(score, best)
        print(string)
        write_result(winner, score)
    print(c.C+"ALL WINNERS COMPUTED"+c.X)
    return

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================

class Breeder:

    # breed two configurations by randomly selecting half of their 'genes' each to produce offspring

    def __init__(self, strategy, population, max_generations=50, performance=1):
        self.strategy = strategy
        self.population = population
        self.folder = 'genetics/breeder/{}/pop{}/'.format(strategy, population)
        self.max_generations = max_generations
        self.compute_performance = getattr(self, 'performance'+str(performance))
        variables = self.get_json('variables')
        self.max_mutations = variables['max_mutations']
        self.child_multiplier = variables['child_multiplier']
        self.tribal_convergence = variables['tribal_convergence']
        self.village_parents = variables['village_parents']
        self.globs = self.get_text('globs').split()
        self.start()
        count_cpus()
        return

    def strategy_function(self, func_name):
        return getattr(self, func_name+'_'+self.strategy)

    def get_text(self, attribute):
        with open(self.folder+attribute+'.txt', 'r') as f:
            data = f.read()
        return data

    def set_text(self, attribute, value):
        with open(self.folder+attribute+'.txt', 'w') as f:
            f.write(str(value))
        return

    def get_json(self, attribute):
        return util.json_to_dict(self.folder+attribute+'.json')

    def set_json(self, attribute, value):
        return util.dict_to_json(value, self.folder+attribute+'.json')

    def get_variable(self, variable):
        return self.get_json('variables')[variable]

    def set_variable(self, variable, value):
        variables = self.get_json('variables')
        variables[variable] = value
        self.set_json('variables', variables)
        return

    def set_new_best(self, score, std, configuration):
        self.set_variable('best_score', score)
        self.set_variable('best_std', std)
        self.set_json('fittest', configuration)
        self.record_survivor(score, configuration)
        return

    # used for couple, thruple, khan, and village
    def record_history(self, perfs, scores, loser_stds):
        order, perfs = list(zip(*perfs))
        perfs = [util.fill_string(p, score_len, 0) for p in perfs]
        scores = [util.fill_string(scores[i], score_len, 0) for i in order]
        l_stds = [util.fill_string(loser_stds[i], score_len, 0) for i in order]
        gen = util.parse_integer(self.get_variable('generations'), leading=2)
        history = self.get_json('history')
        history.update({gen:{'perfs ':' '.join(perfs),
                             'scores':' '.join(scores),
                             'l_stds':' '.join(l_stds)}})
        self.set_json('history', history)
        return

    # used for tribe and chiefdom
    def record_tribal_history(self, perfs1, perfs2, scores1, scores2, loser_stds1, loser_stds2):
        # tribe 1 history
        order1, perfs1 = list(zip(*perfs1))
        perfs1 = [util.fill_string(p, score_len, 0) for p in perfs1]
        scores1 = [util.fill_string(scores1[i], score_len, 0) for i in order1]
        l_stds1 = [util.fill_string(loser_stds1[i], score_len, 0) for i in order1]
        # tribe 2 history
        order2, perfs2 = list(zip(*perfs2))
        perfs2 = [util.fill_string(p, score_len, 0) for p in perfs2]
        scores2 = [util.fill_string(scores2[i], score_len, 0) for i in order2]
        l_stds2 = [util.fill_string(loser_stds2[i], score_len, 0) for i in order2]
        # writing
        gen = util.parse_integer(self.get_variable('generations'), leading=2)
        history = self.get_json('history')
        history.update({gen:{self.strategy+'1':{'perfs ':' '.join(perfs1),
                                                'scores':' '.join(scores1),
                                                'l_stds':' '.join(l_stds1)},
                             self.strategy+'2':{'perfs ':' '.join(perfs2),
                                                'scores':' '.join(scores2),
                                                'l_stds':' '.join(l_stds2)}}})
        self.set_json('history', history)
        return

    def record_best_configuration(self, score, configuration):
        fname = self.config_filename(score)
        filename = self.folder+'configurations/'+fname
        util.dict_to_json(configuration, filename)
        return

    def record_survivor(self, score, configuration):
        fname = self.config_filename(score)
        filename = self.folder+'survivors/'+fname
        util.dict_to_json(configuration, filename)
        return

    def config_filename(self, score):
        time = util.now_string_date_only()
        strat = self.strategy.upper()
        pop = self.population
        gen = util.parse_integer(self.get_variable('generations'), leading=2)
        score = util.fill_string(score, score_len, 0)
        fname = "{}_{}{}_GEN{}_{}.json".format(time, strat, pop, gen, score)
        return fname

    def print_title(self):
        welcome_str = "BREEDING CONFIGURATIONS..."
        pop_str = "{}{} POPULATION {}".format(c.M, self.strategy.upper(), self.population)
        gen_str = "{}GENERATION {}".format(c.C, self.get_variable('generations'))
        score_str = "{}CURRENT BEST SCORE: {}".format(c.G, self.get_variable('best_score'))
        string = welcome_str+div+pop_str+div+gen_str+div+score_str+div+util.now_string_time_only()
        print(string)
        return

    def print_best_descendant(self, score):
        gen_str = c.C+"GENERATION {}".format(util.parse_integer(self.get_variable('generations'), leading=2))
        descendant_str = c.M+'BEST DESCENDANT: '
        col = util.red_or_green(score, self.get_variable('best_score'))
        score_str = col+util.fill_string(score, score_len, filler=0)
        time_str = "{}m".format(round((util.now() - self.start_time)/60000, 1))
        string = div+gen_str+div+descendant_str+score_str+div+time_str+div+util.now_string_time_only()
        print(string)
        return

    def mutate(self, configuration, num_mutations):
        mutant = configuration.copy()    # copy to leave input configuration unchanged
        for _ in range(num_mutations):
            glob = choice(self.globs)
            mutant[glob] = get_mutation(glob)
        return mutant

    def evolve(self):
        self.print_title()
        self.strategy_function('setup')()
        breed_function = self.strategy_function('breed')
        while True:
            if self.get_variable('generations') >= self.max_generations:
                print(c.C+"MAX GENERATIONS REACHED"+c.X)
                return
            if self.finished():
                print(c.Y+"BREEDING STOPPED"+c.X)
                return
            self.start_time = util.now()
            print("BREEDING...", end='')
            breed_function()
        return

    def evaluate_mutants(self, mutants):
        self.seen = self.get_json('seen_configurations')
        data = zip(mutants, repeat(1))
        p = Pool(cpus)
        results = p.starmap(self.evaluate, data)
        p.close()
        p.join()
        self.record_seen(mutants, results)
        return results

    def evaluate(self, configuration, iterations):
        repeated, results = self.repeat_configuration(configuration)
        if repeated:
            return results
        try:
            results = simulate.run_all4(Test_Files, iterations, configuration)
        except RuntimeWarning:
            results = [0]
        return results

    def repeat_configuration(self, configuration):
        config_string = self.configuration_string(configuration)
        if config_string in self.seen:
            return True, self.results_from_string(self.seen[config_string])
        return False, None   # None is not used even if False is returned

    def record_seen(self, configurations, results):
        new_seen = {}
        for i in range(len(configurations)):
            config_string = self.configuration_string(configurations[i])
            res_string = self.results_string(results[i])
            new_seen[config_string] = res_string
        self.seen.update(new_seen)
        self.set_json('seen_configurations', self.seen)
        return

    def configuration_string(self, configuration):
        return ''.join([str(val) for val in configuration.values()])

    def results_string(self, results):
        return ' '.join([str(r) for r in results])

    def results_from_string(self, results_string):
        return [float(r) for r in results_string.split()]

    def performance1(self, scores, loser_stds):
        perfs = sorted(enumerate(scores), key=lambda i: i[1])[::-1]
        return perfs

    def performance2(self, scores, loser_stds):
        perfs = [round(perf, 3) for perf in scores-loser_stds**1.5]
        perfs = sorted(enumerate(perfs), key=lambda i: i[1])[::-1]
        return perfs

    def performance3(self, scores, loser_stds):
        perfs = [round(perf, 3) for perf in scores-loser_stds]
        perfs = sorted(enumerate(perfs), key=lambda i: i[1])[::-1]
        return perfs

    def performance4(self, scores, loser_stds):
        perfs = [round(perf, 3) for perf in scores-sqrt(loser_stds)]
        perfs = sorted(enumerate(perfs), key=lambda i: i[1])[::-1]
        return perfs

    def unique_survivors(self, mutants, perfs, n_parents):
        survivor_indices = []
        survivor_mutants = []
        for index, _ in perfs:
            if mutants[index] in survivor_mutants:
                continue
            survivor_indices.append(index)
            survivor_mutants.append(mutants[index])
            if len(survivor_indices) == n_parents:
                break
        return survivor_indices

    def tribal_swap(self):
        self.parent2, self.parent4 = self.parent4.copy(), self.parent2.copy()
        return

    def start(self):
        return self.set_text('finished', '')

    def finished(self):
        return bool(self.get_text('finished'))     # return True if finished.txt is not empty

    #=======================================================================================================================
    # COUPLE

    def setup_couple(self):
        self.n_children = int(6*self.child_multiplier)
        if not self.get_json('setup'):
            self.parent1 = self.get_json('fittest')
            self.parent2 = self.mutate(self.parent1, 3)
        else:
            d = self.get_json('setup')
            self.parent1 = d['parent1']
            self.parent2 = d['parent2']
        return

    def breed_couple(self):
        # breeding
        children = breed(self.parent1, self.parent2, self.n_children)
        mutants = [self.mutate(child, randint(0,self.max_mutations+1)) for child in children]
        # evaluation
        results = self.evaluate_mutants(mutants)
        scores, loser_stds = extract_results(results)
        perfs = self.compute_performance(scores, loser_stds)
        first_index = perfs[0][0]
        second_index = perfs[1][0]
        best = perfs[0][1]
        best_std = loser_stds[first_index]
        # conclusion
        self.set_variable('generations', self.get_variable('generations')+1)
        self.print_best_descendant(best)
        self.record_history(perfs, scores, loser_stds)
        self.parent1 = mutants[first_index]
        self.parent2 = mutants[second_index]
        if best >= self.get_variable('best_score'):
            self.set_new_best(best, best_std, self.parent1)
        self.record_best_configuration(best, mutants[first_index])
        self.backup_couple()
        return

    def backup_couple(self):
        d = {'parent1':self.parent1, 'parent2':self.parent2}
        self.set_json('setup', d)
        return

    #=======================================================================================================================
    # THRUPLE

    def setup_thruple(self):
        self.n_children = int(2*self.child_multiplier)
        if not self.get_json('setup'):
            self.parent1 = self.get_json('fittest')
            self.parent2 = self.mutate(self.parent1, 3)
            self.parent3 = self.mutate(self.parent1, 3)
        else:
            d = self.get_json('setup')
            self.parent1 = d['parent1']
            self.parent2 = d['parent2']
            self.parent3 = d['parent3']
        return

    def breed_thruple(self):
        # breeding
        children = breed(self.parent1, self.parent2, self.n_children) + breed(self.parent1, self.parent3, self.n_children) + breed(self.parent2, self.parent3, self.n_children)
        mutants = [self.mutate(child, randint(0,self.max_mutations+1)) for child in children]
        # evaluation
        results = self.evaluate_mutants(mutants)
        scores, loser_stds = extract_results(results)
        perfs = self.compute_performance(scores, loser_stds)
        first_index = perfs[0][0]
        second_index = perfs[1][0]
        third_index = perfs[2][0]
        best = perfs[0][1]
        best_std = loser_stds[first_index]
        # conclusion
        self.set_variable('generations', self.get_variable('generations')+1)
        self.print_best_descendant(best)
        self.record_history(perfs, scores, loser_stds)
        self.parent1 = mutants[first_index]
        self.parent2 = mutants[second_index]
        self.parent3 = mutants[third_index]
        if best >= self.get_variable('best_score'):
            self.set_new_best(best, best_std, self.parent1)
        self.record_best_configuration(best, mutants[first_index])
        self.backup_thruple()
        return

    def backup_thruple(self):
        d = {'parent1':self.parent1, 'parent2':self.parent2, 'parent3':self.parent3}
        self.set_json('setup', d)
        return

    #=======================================================================================================================
    # KHAN

    def setup_khan(self):
        self.n_children = int(3*self.child_multiplier)
        if not self.get_json('setup'):
            self.khan = self.get_json('fittest')
            self.wife1 = self.mutate(self.khan, 3)
            self.wife2 = self.mutate(self.khan, 3)
        else:
            d = self.get_json('setup')
            self.khan = d['khan']
            self.wife1 = d['wife1']
            self.wife2 = d['wife2']
        return

    def breed_khan(self):
        # breeding
        children = breed(self.khan, self.wife1, self.n_children) + breed(self.khan, self.wife2, self.n_children)
        mutants = [self.mutate(child, randint(0,self.max_mutations+1)) for child in children]
        # evaluation
        results = self.evaluate_mutants(mutants)
        scores, loser_stds = extract_results(results)
        perfs = self.compute_performance(scores, loser_stds)
        first_index = perfs[0][0]
        second_index = perfs[1][0]
        third_index = perfs[2][0]
        best = perfs[0][1]
        best_std = loser_stds[first_index]
        # conclusion
        self.set_variable('generations', self.get_variable('generations')+1)
        self.print_best_descendant(best)
        self.record_history(perfs, scores, loser_stds)
        self.wife1 = mutants[second_index]
        if best >= self.get_variable('best_score'):
            self.khan = mutants[first_index]
            self.wife2 = mutants[third_index]
            self.set_new_best(best, best_std, self.khan)
        else:
            self.wife2 = mutants[first_index]
        self.record_best_configuration(best, mutants[first_index])
        self.backup_khan()
        return

    def backup_khan(self):
        d = {'khan':self.khan, 'wife1':self.wife1, 'wife2':self.wife2}
        self.set_json('setup', d)
        return

    #=======================================================================================================================
    # TRIBE

    def setup_tribe(self):
        self.n_children = int(3*self.child_multiplier)
        if not self.get_json('setup'):
            self.parent1 = self.get_json('fittest')
            self.parent2 = self.mutate(self.parent1, 3)
            self.parent3 = self.mutate(self.parent2, 3)
            self.parent4 = self.mutate(self.parent3, 3)
        else:
            d = self.get_json('setup')
            self.parent1 = d['parent1']
            self.parent2 = d['parent2']
            self.parent3 = d['parent3']
            self.parent4 = d['parent4']
        return

    def breed_tribe(self):
        # evaluation
        perfs1, scores1, loser_stds1 = self.breed_tribe1()
        perfs2, scores2, loser_stds2 = self.breed_tribe2()
        best_score, best_std, best_mutant = self.local_best
        # conclusion
        self.set_variable('generations', self.get_variable('generations')+1)
        self.print_best_descendant(best_score)
        self.record_tribal_history(perfs1, perfs2, scores1, scores2, loser_stds1, loser_stds2)
        if not self.get_variable('generations') % self.tribal_convergence:     # True every 'tribal_convergence' generations
            self.tribal_swap()
        if best_score > self.get_variable('best_score'):
            self.set_new_best(best_score, best_std, best_mutant)
        self.record_best_configuration(best_score, best_mutant)
        self.backup_tribe()
        return

    def breed_tribe1(self):
        # breeding
        children = breed(self.parent1, self.parent2, self.n_children)
        mutants = [self.mutate(child, randint(0,self.max_mutations+1)) for child in children]
        # evaluation
        results = self.evaluate_mutants(mutants)
        scores, loser_stds = extract_results(results)
        perfs = self.compute_performance(scores, loser_stds)
        first_index = perfs[0][0]
        second_index = perfs[1][0]
        best = perfs[0][1]
        # conclusion
        self.parent1 = mutants[first_index]
        self.parent2 = mutants[second_index]
        self.local_best = (best, loser_stds[first_index], self.parent1)
        return perfs, scores, loser_stds

    def breed_tribe2(self):
        # breeding
        children = breed(self.parent3, self.parent4, self.n_children)
        mutants = [self.mutate(child, randint(0,self.max_mutations+1)) for child in children]
        # evaluation
        results = self.evaluate_mutants(mutants)
        scores, loser_stds = extract_results(results)
        perfs = self.compute_performance(scores, loser_stds)
        first_index = perfs[0][0]
        second_index = perfs[1][0]
        best = perfs[0][1]
        # conclusion
        self.parent3 = mutants[first_index]
        self.parent4 = mutants[second_index]
        if best >= self.local_best[0]:
            self.local_best = (best, loser_stds[first_index], self.parent3)
        return perfs, scores, loser_stds

    def backup_tribe(self):
        d = {'parent1':self.parent1, 'parent2':self.parent2, 'parent3':self.parent3, 'parent4':self.parent4}
        self.set_json('setup', d)
        return

    #=======================================================================================================================
    # CHIEFDOM

    def setup_chiefdom(self):
        self.n_children = int(3*self.child_multiplier)
        if not self.get_json('setup'):
            self.parent1 = self.get_json('fittest')
            self.parent2 = self.mutate(self.parent1, 3)
            self.parent3 = self.mutate(self.parent2, 3)
            self.parent4 = self.mutate(self.parent3, 3)
            self.set_variable('chiefdom1_best', 0)
            self.set_variable('chiefdom2_best', 0)
        else:
            d = self.get_json('setup')
            self.parent1 = d['parent1']
            self.parent2 = d['parent2']
            self.parent3 = d['parent3']
            self.parent4 = d['parent4']
        return

    def breed_chiefdom(self):
        # evaluation
        perfs1, scores1, loser_stds1 = self.breed_chiefdom1()
        perfs2, scores2, loser_stds2 = self.breed_chiefdom2()
        best_score, best_std, best_mutant = self.local_best
        # conclusion
        self.set_variable('generations', self.get_variable('generations')+1)
        self.print_best_descendant(best_score)
        self.record_tribal_history(perfs1, perfs2, scores1, scores2, loser_stds1, loser_stds2)
        if not self.get_variable('generations') % self.tribal_convergence:     # True every 'tribal_convergence' generations
            self.tribal_swap()
        if best_score > self.get_variable('best_score'):
            self.set_new_best(best_score, best_std, best_mutant)
        self.record_best_configuration(best_score, best_mutant)
        self.backup_chiefdom()
        return

    def breed_chiefdom1(self):
        # breeding
        children = breed(self.parent1, self.parent2, self.n_children)
        mutants = [self.mutate(child, randint(0,self.max_mutations+1)) for child in children]
        # evaluation
        results = self.evaluate_mutants(mutants)
        scores, loser_stds = extract_results(results)
        perfs = self.compute_performance(scores, loser_stds)
        first_index = perfs[0][0]
        second_index = perfs[1][0]
        best = perfs[0][1]
        # conclusion
        if best >= self.get_variable('chiefdom1_best'):
            self.set_variable('chiefdom1_best', best)
            self.parent1 = mutants[first_index]
            self.parent2 = mutants[second_index]
        else:
            self.parent2 = mutants[first_index]
        self.local_best = (best, loser_stds[first_index], self.parent1)
        return perfs, scores, loser_stds

    def breed_chiefdom2(self):
        # breeding
        children = breed(self.parent3, self.parent4, self.n_children)
        mutants = [self.mutate(child, randint(0,self.max_mutations+1)) for child in children]
        # evaluation
        results = self.evaluate_mutants(mutants)
        scores, loser_stds = extract_results(results)
        perfs = self.compute_performance(scores, loser_stds)
        first_index = perfs[0][0]
        second_index = perfs[1][0]
        best = perfs[0][1]
        # conclusion
        if best >= self.get_variable('chiefdom2_best'):
            self.set_variable('chiefdom2_best', best)
            self.parent3 = mutants[first_index]
            self.parent4 = mutants[second_index]
        else:
            self.parent4 = mutants[first_index]
        if best >= self.local_best[0]:
            self.local_best = (best, loser_stds[first_index], self.parent3)
        return perfs, scores, loser_stds

    def backup_chiefdom(self):
        d = {'parent1':self.parent1, 'parent2':self.parent2, 'parent3':self.parent3, 'parent4':self.parent4}
        self.set_json('setup', d)
        return

    #=======================================================================================================================
    # VILLAGE

    def setup_village(self):
        self.n_parents = int(self.village_parents)
        if not self.get_json('setup'):
            self.parents = {'parent1':self.get_json('fittest')}
            for i in range(1,self.n_parents):
                parent = self.mutate(self.parents['parent1'], 3)
                self.parents['parent'+str(i+1)] = parent
        else:
            self.parents = self.get_json('setup')
        return

    def breed_village(self):
        # breeding
        children = []
        for pair in combinations(list(self.parents), 2):
            p1, p2 = pair
            children.extend(breed(self.parents[p1], self.parents[p2], 2))
        mutants = [self.mutate(child, randint(0,self.max_mutations+1)) for child in children]
        # evaluation
        results = self.evaluate_mutants(mutants)
        scores, loser_stds = extract_results(results)
        perfs = self.compute_performance(scores, loser_stds)
        perf_indices = self.unique_survivors(mutants, perfs, self.n_parents)
        best = perfs[0][1]
        best_std = loser_stds[perf_indices[0]]
        # conclusion
        self.set_variable('generations', self.get_variable('generations')+1)
        self.print_best_descendant(best)
        self.record_history(perfs, scores, loser_stds)
        for i in range(self.n_parents):
            self.parents['parent'+str(i+1)] = mutants[perf_indices[i]]
        if best >= self.get_variable('best_score'):
            self.set_new_best(best, best_std, self.parents['parent1'])
        self.record_best_configuration(best, mutants[perf_indices[0]])
        self.backup_village()
        return

    def backup_village(self):
        self.set_json('setup', self.parents)
        return

#====================================================================================================================================================================================================================
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#====================================================================================================================================================================================================================

class Mutator:

    # inflict mutations and evaluate whether they were beneficial - if not, drop the mutation

    def __init__(self, strategy, population, max_cycle=1, score_len=5):
        self.strategy = strategy
        self.population = population
        self.max_cycle = max_cycle
        self.score_len = score_len
        self.folder = 'genetics/mutator/{}/pop{}/'.format(strategy, population)
        variables = self.get_json('variables')
        self.n_mutators = variables['n_mutators']
        self.gens_per_cycle = variables['gens_per_cycle']
        self.n_survivors = variables['n_survivors']
        self.n_survivor_children = variables['n_survivor_children']
        self.file_type = variables['file_type']
        self.run_iterations = variables['run_iterations']
        self.glob_ranges = glob_range_types[self.file_type]
        self.globs = self.get_text('globs').split()
        self.run_all = run_all_functions[self.file_type]
        self.precision = precisions[self.file_type]
        self.compute_file_time()
        self.start()
        count_cpus()
        return

    def strategy_function(self, func_name):
        return getattr(self, func_name+'_'+self.strategy)

    def get_text(self, attribute):
        with open(self.folder+attribute+'.txt', 'r') as f:
            data = f.read()
        return data

    def set_text(self, attribute, value):
        with open(self.folder+attribute+'.txt', 'w') as f:
            f.write(str(value))
        return

    def get_json(self, attribute):
        return util.json_to_dict(self.folder+attribute+'.json')

    def set_json(self, attribute, value):
        return util.dict_to_json(value, self.folder+attribute+'.json')

    def get_variable(self, variable):
        return self.get_json('variables')[variable]

    def set_variable(self, variable, value):
        variables = self.get_json('variables')
        variables[variable] = value
        self.set_json('variables', variables)
        return

    def print_title(self):
        welcome_str = "MUTATING CONFIGURATIONS..."
        pop_str = "{}{} POPULATION {}".format(c.M, self.strategy.upper(), self.population)
        gen_str = "{}CYCLE {} GENERATION {}".format(c.C, self.get_variable('cycle'), self.get_variable('generations'))
        score_str = "{}CURRENT BEST SCORE: {}".format(c.G, self.get_variable('best_score'))
        string = welcome_str+div+pop_str+div+gen_str+div+score_str+div+util.now_string_time_only()
        print(string)
        return

    def print_best_mutator(self, score, mutator=0, lesser_improved=False):
        cycle = util.parse_integer(self.get_variable('cycle'), leading=1)
        gens = util.parse_integer(self.get_variable('generations'), leading=2)
        mut = util.parse_integer(mutator, leading=1)
        mut_col = c.M if mutator==0 or self.mutators['mutator'+str(mutator)]['best'] > score else c.C
        bracket_col = c.C if lesser_improved else c.M
        gen_str = c.C+"CYCLE {} GENERATION {}".format(cycle, gens)
        mutator_str = c.M+'BEST MUTATOR {}({}{}{}){}: '.format(bracket_col, mut_col, mut, bracket_col, c.M)
        col = util.red_or_green(score, self.get_variable('best_score'))
        score_str = col+util.fill_string(score, self.score_len, filler=0)
        time_str = "{}m".format(round((util.now() - self.start_time)/60000, 1))
        string = div+gen_str+div+mutator_str+score_str+div+time_str+div+util.now_string_time_only()
        print(string)
        return

    def record_history(self, perfs, scores, loser_stds):
        order, perfs = list(zip(*perfs))
        perfs = [util.fill_string(p, self.score_len, 0) for p in perfs]
        scores = [util.fill_string(scores[i], self.score_len, 0) for i in order]
        cycle = util.parse_integer(self.get_variable('cycle'), leading=1)
        gen = util.parse_integer(self.get_variable('generations'), leading=2)
        cycle_gen = cycle+'_'+gen
        history = self.get_json('history')
        history.update({cycle_gen:{'perfs ':' '.join(perfs),
                                   'scores':' '.join(scores)}})
        self.set_json('history', history)
        return

    def set_new_best(self, score, std, configuration):
        self.set_variable('best_score', score)
        self.set_variable('best_std', std)
        self.set_json('fittest', configuration)
        self.record_survivor(score, configuration)
        return

    def record_survivor(self, score, configuration):
        fname = self.config_filename(score)
        filename = self.folder+'survivors/'+fname
        util.dict_to_json(configuration, filename)
        return

    def record_best_configuration(self, score, configuration):
        fname = self.config_filename(score)
        filename = self.folder+'configurations/'+fname
        util.dict_to_json(configuration, filename)
        return

    def config_filename(self, score):
        time = util.now_string_date_only()
        strat = self.strategy.upper()
        pop = self.population
        cycle = util.parse_integer(self.get_variable('cycle'), leading=1)
        gen = util.parse_integer(self.get_variable('generations'), leading=2)
        score = util.fill_string(score, self.score_len, 0)
        fname = "{}_{}{}_CYC{}_GEN{}_{}.json".format(time, strat, pop, cycle, gen, score)
        return fname

    def compute_file_time(self):
        files = simulate.btc_files() if self.file_type == 'btc' else simulate.test_files()
        total_time = 0
        for file in files:
            times, _ = zip(*simulate.gen_file(file))
            total_time += times[-1] - times[0]
        self.days = total_time/86400000
        return

    def evolve(self):
        self.print_title()
        self.strategy_function('setup')()
        mutate_function = self.strategy_function('mutate')
        while True:
            cycle = self.get_variable('cycle')
            gens = self.get_variable('generations')
            if self.finished():
                print(c.Y+"MUTATION STOPPED"+c.X)
                return
            if cycle == self.max_cycle and gens == self.gens_per_cycle:
                print(c.C+"MAX CYCLE REACHED"+c.X)
                return
            if gens == self.gens_per_cycle:
                self.new_cycle()
                continue
            mutate_function()
        return

    def start(self):
        return self.set_text('finished', '')

    def finished(self):
        return bool(self.get_text('finished'))     # return True if finished.txt is not empty

    #=======================================================================================================================
    # GROUP

    def setup_group(self):
        current_setup = self.get_json('setup')
        self.mutators = current_setup if any(current_setup) else {'mutator{}'.format(i+1):{'configuration':self.random_configuration(), 'best':-100} for i in range(self.n_mutators)}
        return

    def mutate_group(self):
        # starting
        self.start_time = util.now()
        print("MUTATING...", end='')
        # evaluation
        mutants = self.mutate_mutators()
        results = self.evaluate_mutants(mutants)
        scores, loser_stds = self.extract_results(results)
        perfs = self.compute_performance(scores, loser_stds)
        perf_indices = [perfs[i][0] for i in range(self.n_mutators)]
        best_index = perf_indices[0]
        best_score = perfs[0][1]
        best_std = float(loser_stds[best_index])
        best_mutator = list(self.mutators)[best_index][7:]
        # conclusion
        self.set_variable('generations', self.get_variable('generations')+1)
        self.record_history(perfs, scores, loser_stds)
        self.choose_mutations(mutants, scores)
        lesser_improved = self.get_lesser_improved(perfs[1:])
        self.print_best_mutator(best_score, best_mutator, lesser_improved)
        if best_score >= self.get_variable('best_score'):
            self.set_new_best(best_score, best_std, mutants[best_index])
        self.record_best_configuration(best_score, mutants[best_index])
        self.backup_group()
        return

    def mutate_mutators(self):
        return [self.mutate(info['configuration']) for info in self.mutators.values()]

    def mutate(self, configuration):
        mutant = configuration.copy()     # copy to leave input configuration unchanged
        glob = choice(self.globs)
        mutant[glob] = self.get_mutation(glob)
        return mutant

    def get_mutation(self, glob):
        high, low = self.glob_ranges[glob]
        mutation = (high - low)*random() + low
        mutation = int(mutation) if glob in integer_globs else round(mutation, self.precision)
        return mutation

    def random_configuration(self):
        configuration = {glob:(self.get_mutation(glob) if glob in self.globs else default_values[glob]) for glob in self.glob_ranges}
        return configuration

    def evaluate_mutants(self, configurations):
        data = zip(configurations, repeat(self.run_iterations))
        p = Pool(cpus)
        results = p.starmap(self.evaluate, data)
        p.close()
        p.join()
        return results

    def evaluate(self, configuration, iterations):
        try:
            results = self.run_all(iterations, configuration)
        except Exception:
            results = [0]
        return results

    def extract_results(self, results):
        scores, loser_stds = [], []
        for bals in results:
            p_gains = [util.percentage(bal, 1000) for bal in bals]
            total_p_gain = sum(p_gains)
            p_gain_per_day = total_p_gain/self.days
            scores.append(round(p_gain_per_day, 3))
            loser_stds.append(round(compute_loser_std(bals), 3))
        return array(scores), array(loser_stds)

    def compute_performance(self, scores, loser_stds):
        perfs = sorted(enumerate(scores), key=lambda i: i[1])[::-1]
        return perfs

    def choose_mutations(self, mutants, scores):
        for mutator, mutant, score in zip(self.mutators, mutants, scores):
            if self.mutators[mutator]['best'] >= score:
                continue   # don't choose inferior mutation
            self.mutators[mutator]['configuration'] = mutant
            self.mutators[mutator]['best'] = score
        return

    def get_lesser_improved(self, perfs):
        setup = self.get_json('setup')
        if not any(setup):
            return False
        for index, score in perfs:
            n = str(index+1)
            best = setup['mutator'+n]['best']
            if score > best:
                return True
        return False

    def backup_group(self):
        self.set_json('setup', self.mutators)
        return

    def new_cycle(self):
        # starting
        self.start_time = util.now()
        print("NEW CYCLE..", end='')
        # breeding
        survivors = self.survivor_mutators(self.n_survivors)
        children = []
        for pair in combinations(survivors, 2):
            m1, m2 = pair
            children.extend(breed(self.mutators[m1]['configuration'], self.mutators[m2]['configuration'], self.n_survivor_children))
        # evaluation
        results = self.evaluate_mutants(children)
        scores, loser_stds = self.extract_results(results)
        perfs = self.compute_performance(scores, loser_stds)
        perf_indices = [perfs[i][0] for i in range(self.n_mutators)]
        best = perfs[0][1]
        best_std = loser_stds[perf_indices[0]]
        # conclusion
        self.set_variable('generations', 0)
        self.set_variable('cycle', self.get_variable('cycle')+1)
        self.print_best_mutator(best)
        self.record_history(perfs, scores, loser_stds)
        for i in range(self.n_mutators):
            index = perf_indices[i]
            self.mutators['mutator{}'.format(i+1)] = {'configuration':children[index], 'best':scores[index]}
        if best >= self.get_variable('best_score'):
            self.set_new_best(best, best_std, children[perf_indices[0]])
        self.record_best_configuration(best, children[perf_indices[0]])
        self.backup_group()
        return

    def survivor_mutators(self, n_survivors):
        mutators = [mutator for mutator,_ in reversed(sorted(self.mutators.items(), key = lambda i: i[1]['best']))]
        return mutators[:n_survivors]
