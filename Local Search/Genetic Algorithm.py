import random
import numpy as np
from search import *
import matplotlib.pyplot as plt

# Especificações do problema
width = 33
height = 16
start = (1, 1)
end = (31, 13)

num_runs = 30
parameter_combinations = [
    {"population_size": 100, "generations": 100, "mutation_rate": 1},
    {"population_size": 400, "generations": 100, "mutation_rate": 1},
    {"population_size": 800, "generations": 100, "mutation_rate": 1},
    {"population_size": 1600, "generations": 100, "mutation_rate": 1},
    {"population_size": 400, "generations": 100, "mutation_rate": 1},
    {"population_size": 400, "generations": 250, "mutation_rate": 1},
    {"population_size": 400, "generations": 500, "mutation_rate": 1},
    {"population_size": 400, "generations": 1000, "mutation_rate": 1},
    {"population_size": 200, "generations": 100, "mutation_rate": 1},
    {"population_size": 200, "generations": 250, "mutation_rate": 5},
    {"population_size": 200, "generations": 500, "mutation_rate": 15},
    {"population_size": 200, "generations": 1000, "mutation_rate": 20}
]
import random
import numpy as np
import random
from search import parse_level, greedy_best_first, path_cost, h_euclidian, transition_model
class Individual:
    def __init__(self, width, height, start, end, genotype=None):
        self.width = width
        self.height = height
        self.start = start
        self.end = end

        # Generate a random quadrant map
        if genotype is not None:
            self.genotype = genotype
        else:
            self.gen_genotype()

    def gen_genotype(self):
        # Cria um genoma aleatório onde X representa paredes e 1 representa os espaços
        self.genotype = np.random.choice(['X', '1'], size=(self.height, self.width), p=[0.5, 0.5])

        # Aplica simetria ao genoma
        for i in range(self.height):
            for j in range(self.width // 2):
                # Espelha o valor da célula para o lado oposto
                self.genotype[i][self.width - 1 - j] = self.genotype[i][j]

    def get_phenotype(self):
        sy,sx=self.start
        gy,gx=self.end
        genotype=self.genotype
        genotype[(sx,sy)]='S'
        genotype[(gx,gy)]='G'
        return genotype.tolist()

    def fitness(self):
        # Obter o fenótipo do indivíduo
        phenotype = self.get_phenotype()

        # Calcular o custo do caminho usando greedy best-first search
        level = parse_level(phenotype)
        path, _ = greedy_best_first(self.start, self.end, level, transition_model, h_euclidian)
        max_cost = self.width * self.height

        if isinstance(path, list):  # Verificar se um caminho foi encontrado
            # Se um caminho foi encontrado, retornar o custo do caminho ponderado pela simetria
            return (path_cost(path, level) / max_cost)
        else:
            # Se nenhum caminho foi encontrado, retornar a heurística entre S e N ponderada pela simetria
            return (path / max_cost) 


class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, width, height, start, end):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.width = width
        self.height = height
        self.start = start
        self.end = end
        self.current_generation = 0
        self.initialize_population()
        self.elite=0.1

    def initialize_population(self):        
        self.population = []
        for _ in range(self.population_size):
            ind = Individual(width=self.width, height=self.height, start=self.start, end=self.end)
            self.population.append(ind)

    def selection(self, fitness_scores):
        parents = []
        for _ in range(2):  # Realizar o torneio duas vezes para selecionar dois pais
            # Selecionar dois indivíduos aleatoriamente
            ind1, ind2 = random.sample(range(self.population_size), k=2)
            # Comparar seus valores de fitness e escolher o de maior fitness
            if fitness_scores[ind1] > fitness_scores[ind2]:
                parents.append(self.population[ind1])
            else:
                parents.append(self.population[ind2])
        return parents[0], parents[1]


    def crossover(self, parent1, parent2):
        # Selecionar um ponto de corte aleatório
        cut_point = random.randint(1, self.height -1)  # Evitar cortar nas bordas para manter 'S' e 'G'

        # Dividir os genótipos dos pais em cima e embaixo do ponto de corte
        parent1_genotype_top = parent1.genotype[:cut_point, :]
        parent1_genotype_bottom = parent1.genotype[cut_point:, :]

        parent2_genotype_top = parent2.genotype[:cut_point, :]
        parent2_genotype_bottom = parent2.genotype[cut_point:, :]

        # Trocar as partes superiores e inferiores entre os pais
        offspring_genotype1 = np.vstack((parent1_genotype_top, parent2_genotype_bottom))
        offspring_genotype2 = np.vstack((parent2_genotype_top, parent1_genotype_bottom))

        # Criar os descendentes
        offspring1 = Individual(width=self.width, height=self.height, start=self.start, end=self.end, genotype=offspring_genotype1)
        offspring2 = Individual(width=self.width, height=self.height, start=self.start, end=self.end, genotype=offspring_genotype2)

        return offspring1, offspring2

    def mutate(self, individual):
        # Gerar um valor aleatório entre 0 e 1
        random_value = random.random()

        # Comparar com a taxa de mutação
        if random_value <= self.mutation_rate:
            genotype = individual.genotype

            # Escolher aleatoriamente uma posição no genótipo
            i = random.randint(0, len(genotype) - 1)
            j = random.randint(0, len(genotype[0]) - 1)

            # Verificar se a posição está na metade esquerda ou direita
            if j < len(genotype[0]) // 2:  # Está na metade esquerda
                # Mutar o gene na posição escolhida
                if genotype[i][j] == '1':
                    genotype[i][j] = 'X'
                    # Mutar o gene simétrico
                    genotype[i][-(j + 1)] = 'X'
                elif genotype[i][j] == 'X':
                    genotype[i][j] = '1'
                    # Mutar o gene simétrico
                    genotype[i][-(j + 1)] = '1'
            else:  # Está na metade direita
                # Calcular a posição simétrica na metade esquerda
                symmetric_j = len(genotype[0]) - 1 - j
                # Mutar o gene na posição escolhida
                if genotype[i][j] == '1':
                    genotype[i][j] = 'X'
                    # Mutar o gene simétrico
                    genotype[i][symmetric_j] = 'X'
                elif genotype[i][j] == 'X':
                    genotype[i][j] = '1'
                    # Mutar o gene simétrico
                    genotype[i][symmetric_j] = '1'
    def evolve(self):
        if self.current_generation >= self.generations:
            return (None, None)
        fitness_scores = [ind.fitness() for ind in self.population]
        # Identificar os melhores indivíduos
        num_elites = int(self.elite * len(self.population))  # Manter os top 10% como elite

        elites_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)[:num_elites]
        elites = [self.population[i] for i in elites_indices]

        new_population = elites  # Adicionar os melhores indivíduos à nova população

        # Repetir até que o tamanho de new_population seja igual a self.population_size
        while len(new_population) < self.population_size:
            # Selecionar dois pais para cruzamento
            parent1, parent2 = self.selection(fitness_scores)

            # Cruzar os pais selecionados para gerar dois filhos
            child1, child2 = self.crossover(parent1, parent2)

            # Mutar os filhos
            self.mutate(child1)
            self.mutate(child2)

            # Adicionar os filhos à nova população
            new_population.append(child1)
            new_population.append(child2)

        # Atualizar a população atual com a nova geração
        self.population = new_population

        # Encontrar o melhor indivíduo da população
        best_individual = max(self.population, key=lambda ind: ind.fitness())

        best = best_individual.get_phenotype()
        level = parse_level(best)
        path, _ = greedy_best_first(self.start, self.end, level, transition_model, h_euclidian)
        self.current_generation += 1
        fit=path_cost(path,level)
        # Retornar o fenótipo do melhor indivíduo e o caminho mais curto
        return fit

# Function to run the genetic algorithm and collect fitness values
def run_genetic_algorithm(width, height, start, end, population_size, generations, mutation_rate):
    fitness_values = []
    genetic_algorithm = GeneticAlgorithm(population_size, generations, mutation_rate, width, height, start, end)
    for _ in range(generations):
        fitness = genetic_algorithm.evolve()
        if fitness is not None:
            fitness_values.append(fitness)
    return fitness_values

# Run the genetic algorithm for each parameter combination and collect fitness values
for params in parameter_combinations:
    population_size = params["population_size"]
    generations = params["generations"]
    mutation_rate = params["mutation_rate"]
    
    all_fitness_values = []
    
    for run in range(num_runs):  #w Run each configuration three times
        print(f"Running genetic algorithm - Population Size: {population_size}, Generations: {generations}, Mutation Rate: {mutation_rate}, Run {run+1}/{num_runs}")
        
        # Run genetic algorithm and collect fitness values
        fitness_values = run_genetic_algorithm(width, height, start, end, population_size, generations, mutation_rate)
        all_fitness_values.append(fitness_values)
    
    # Calculate the mean fitness value for each generation
    mean_fitness_values = np.mean(all_fitness_values, axis=0)
    
    # Plot the adaptation curve for the current parameter combination
    plt.plot(range(1, generations+1), mean_fitness_values, label=f'Pop: {population_size}, Gerações: {generations}, Taxa Mutação: {mutation_rate}')
    
    # Plot settings
    plt.title('Curva de Adaptação')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(False)
    plt.xticks(np.arange(1, generations+1, 1))

    # Save the plot as PNG and JPEG files
    plt.savefig(f'curva_adaptacao_{population_size}_{generations}_{mutation_rate}.png')
    
    # Clear the plot for the next iteration
    plt.clf()

# Show the plots
plt.show()
