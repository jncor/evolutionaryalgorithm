from EA.ea import EA, Setup, Individual, Selection, Tournament
from EA.callbacks import Callback
import numpy as np
import argparse
import time
import datetime

class VectorIndividualCallback(Callback):
    def __init__(self):
        super(VectorIndividualCallback, self).__init__()

    def on_evaluation_end(self, population, logs=None):
        super(VectorIndividualCallback, self).on_evaluation_end(population, logs)
        print('finished the evaluation of generation ', self.model.current_generation)
        self.export_stats()

    def export_stats(self):

        # simple exporter per generation...
        with open('vectorMaxSUMevolution-{0}-seed-{1}.csv'.format(timestamp, self.setup.random_seed), 'a') as stats_f:
            if self.model.current_generation == 0:
                # diversity_per_gene = []
                # for gene in range(len(self.model.current_population[0].genes)):
                #     diversity_per_gene.append(str(gene))
                # diversity_per_gene_str = '_gene,'.join(diversity_per_gene)
                stats_f.write(
                    'generation,avg,std,min,max,best_individual,best_individual_genes\n')

            fitness_values = [t_indiv.fitness for t_indiv in self.model.current_population]
            popavg = np.mean(fitness_values)
            popstd = np.std(fitness_values)
            popmin = np.min(fitness_values)
            popmax = np.max(fitness_values)

            best_individual = self.model.current_population[np.argmax(fitness_values)]
            print('best so far: ',best_individual)
            stats_f.write(
                '{},{},{},{},{},{},{}'.format(self.model.current_generation,
                                                popavg, popstd, popmin, popmax, best_individual.fitness,best_individual.genes).replace('\n',' ') +'\n')
        #
        # could be important to export the whole population or do something with it...
        #




class VectorIndividual(Individual):


    def __init__(self, creation_index, model, individuals_dict):
#        super(VectorIndividual, self).__init__(creation_index, model, individuals_dict)
        self.id = creation_index
        self.fitness = 0.0
        self.individuals_dict = individuals_dict
        #print(ref)
        #print(id)
        self.model = model # model to which it belongs..
        self.genes = None
        self.genes_length = individuals_dict.get('genes_length', 20)
        self.max_values_per_gen = individuals_dict.get('max_values_per_gen', 100)
        self.min_values_per_gen = individuals_dict.get('min_values_per_gen', 0)
        self.type_values_per_gen = individuals_dict.get('type_values_per_gen', int)

        # to define custom ranges for each gene.. define the range per gene using lists.. or it just uses the same value for all
        if not isinstance(self.max_values_per_gen, list):
            self.max_values_per_gen = [self.max_values_per_gen] * self.genes_length
        if not isinstance(self.min_values_per_gen, list):
            self.min_values_per_gen = [self.min_values_per_gen] * self.genes_length
        if not isinstance(self.type_values_per_gen, list):
            self.type_values_per_gen = [self.type_values_per_gen] * self.genes_length

    def initialize(self):

        self.genes = np.array([self.pickvalue(gene_i) for gene_i in
                               range(self.genes_length)])

    def pickvalue(self, index):

        if self.min_values_per_gen[index] == self.max_values_per_gen[index]:
            return self.min_values_per_gen[index]

        if self.type_values_per_gen[index] == int:
            # print(self.max_values_per_gen[index])
            # print(self.min_values_per_gen[index])
            # print('------')
            return np.random.randint(self.min_values_per_gen[index], self.max_values_per_gen[index])
        if self.type_values_per_gen[index] == float:
            # (b - a) * random_sample() + a
            return ((self.max_values_per_gen[index] - self.min_values_per_gen[index]) * np.random.random_sample()) + self.min_values_per_gen[index]
            # todo implement generate in an interval of values

    def genotype_to_phenotype(self):
        # for test purposes it manifests as a sum of all members
        return np.sum(self.genes)

    def mutate(self, probability=None):
        # mutate only a random gene... generate a new value
        index_to_mutate = int(np.random.randint(0, self.genes_length - 1))
        self.genes[index_to_mutate] = self.pickvalue(index_to_mutate)

    def crossover(self, other, probability):
        # uniform
        self.genes = np.array([self.genes[n] if np.random.random() < probability else other.genes[n] for n in range(self.genes_length)])

    def genotype(self):
        return self.genes

    def __str__(self):
        return str(self.genes) + " fitness: " + str(self.fitness)


def maxeval(individual):
    individual.fitness = individual.genotype_to_phenotype()
    return individual


class VectorEvolution(EA):

    def check_termination_criterion(self):
        return self.current_generation == self.setup.generations


global timestamp

if __name__ == '__main__':
    start = time.time()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S')
    ea_testing = VectorEvolution()
    ea_cb = VectorIndividualCallback()

    # setup GA
    a_setup = Setup()
    a_setup.generations = 10
    a_setup.multiprocesses = 1
    a_setup.random_seed = 0
    a_setup.torn_size = 3

    # callback needs reference to the GA model and the Setup
    ea_cb.set_model(ea_testing)
    ea_cb.set_setup(a_setup)

    ea_testing.initialize(setup=a_setup, individual_ref=VectorIndividual, selection_ref=Tournament,
                          evaluation_ref=maxeval)

    # pass reference to callback!
    ea_testing.evolve(callbacks=[ea_cb])

    print('*********')
    print('total_time ', time.time() - start)