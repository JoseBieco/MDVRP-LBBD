import gurobipy as grb
from gurobipy import GRB
from itertools import combinations, product
import math
import os
import time
from collections import defaultdict

# Constantes
INSTANCES_DIR = r'../datasets/C-mdvrp/Ajustados'
RESULTS_DIR = r'./Resultados/Gurobi'

# Função para ler instância personalizada de MDVRP conforme o formato especificado
def read_mdvrp_instance(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Primeira linha: tipo de problema, número de veículos, número de clientes, número de depósitos
    problem_type, m, n, t = map(int, lines[0].split())
    
    # Dados dos depósitos
    depots = []
    for i in range(1, t + 1):
        D, Q = map(int, lines[i].split())
        depots.append({'max_duration': D, 'max_load': Q})

    # Dados dos clientes
    customers = []
    for i in range(t + 1, len(lines)):
        data = list(map(int, lines[i].split()))
        customer = {
            'id': data[0],
            'x': data[1],
            'y': data[2],
            'service_duration': data[3],
            'demand': data[4],
            'frequency': data[5],
            'num_combinations': data[6],
            'visit_combinations': data[7:7 + data[6]],
            'time_window_start': data[-2] if len(data) > 7 + data[6] else None,
            'time_window_end': data[-1] if len(data) > 7 + data[6] else None,
        }
        customers.append(customer)
    
    # Lista os últimos t depósitos e adiciona os atributos `x` e `y` a cada depósito
    for depot, customer in zip(depots, customers[-t:]):
        depot.update({'x': customer['x'], 'y': customer['y']})

    return problem_type, m, n, t, depots, customers

def calculate_distance_matrix(customers, depots):
    # Combinar coordenadas dos clientes e depósitos
    all_points = [(cust['x'], cust['y']) for cust in customers] + [(dep['x'], dep['y']) for dep in depots]
    num_points = len(all_points)
    
    # Inicializar a matriz de custo
    distance_matrix = [[0] * num_points for _ in range(num_points)]
    
    # Calcular a distância euclidiana entre cada par de pontos
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                x1, y1 = all_points[i]
                x2, y2 = all_points[j]
                distance_matrix[i][j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    return distance_matrix

def display_results(model, customers, depots, vehicles, depots_ids, demands):
    zvalues = model.getAttr('X', model._z)

    # Estrutura para armazenar os resultados
    results = defaultdict(lambda: defaultdict(lambda: {'clients': [], 'load': 0}))
    
    # Preencher os resultados
    for (i, k, d) in zvalues.keys():
        if zvalues[i, k, d] > 0.5:
            results[d][k]['clients'].append(i + 1)
            results[d][k]['load'] += demands[i]
    
     # Soma distância total percorrida
    distancia_total = 0
    for d in depots_ids:
        for k in vehicles:
            percurso = results[d][k]['clients']
            for i in range(len(percurso)):
                if i == 0:
                    distancia_total += round(model._edges_weights[len(customers) + d][percurso[i]-1], 2)
                    #print(f'D{d+1} > {percurso[i]} - {round(model._edges_weights[len(customers) + d][percurso[i]-1], 2)}')
                if i + 1 == len(percurso):
                    distancia_total += round(model._edges_weights[percurso[i]-1][len(customers) + d], 2)
                    #print(f'{percurso[i]} > D{d+1} - {round(model._edges_weights[percurso[i]-1][len(customers) + d], 2)}')
                else:
                    distancia_total +=  round(model._edges_weights[percurso[i]-1][percurso[i+1]-1], 2)
                    #print(f'{percurso[i]} > {percurso[i+1]} - {round(model._edges_weights[percurso[i]-1][percurso[i+1]-1], 2)}')

    write_results = {
            'objVal': model.objVal,
            'Runtime': model.Runtime,
            'MIPGap': model.MIPGap,
            'lines': [],
            'distancia_total': round(distancia_total, 2)
            }

    # Imprimir a tabela de resultados
    print(f'Model Obj: {model.objVal}, Rumtime: {model.Runtime}, MIPGap: {model.MIPGap}')
    print(f'Distancia total calculada: {round(distancia_total, 2)}')
    print("-" * 60)
    print(f"{'Depósito':<10} {'Veículo':<10} {'Capacidade Utilizada':<20} {'Clientes Atendidos'}")
    print("-" * 60)

    for d in depots_ids:
        for k in vehicles:
            if results[d][k]['clients']:
                clients_sequence = ' -> '.join(map(str, results[d][k]['clients']))
                #print(f"-{results[d][k]['clients']}\n")
                print(f"{1 + d:<10} {1 + k:<10} {results[d][k]['load']:<20} {clients_sequence}")
                write_results['lines'].append(f"{1 + d:<10} {1 + k:<10} {results[d][k]['load']:<20} {clients_sequence}")

    return write_results

def new_filename(filename: str, localtime):
    # Procura a existência de um separador
    if filename.find('.') != -1:
        # Separa nome do arquivo da extensão
        filename, extension = filename.split(sep='.')
        filename = f'{filename}_{localtime.tm_mday}_{localtime.tm_mon}_{localtime.tm_hour}_{localtime.tm_min}_{localtime.tm_sec}'
        return f'{filename}.{extension}'

    filename = f'{filename}_{localtime.tm_mday}_{localtime.tm_mon}_{localtime.tm_hour}_{localtime.tm_min}_{localtime.tm_sec}'
    return f'{filename}'

def write_result_file(path, filename, results):
    # Cria a pasta para os Resuldos caso não exista
    if not os.path.exists(path):
        os.mkdir(path)

    # Verifica se já não existe um arquivo com o mesmo nome (Mesma instância executada)
    if os.path.exists(f'{path}/{filename}'):
        filename = new_filename(filename=filename, localtime=time.localtime())

    filePath = f'{path}/{filename}'

    if results == None:
        file = open(filePath, '+a')
        file.write(f"Não foram encontradas soluções para o modelo!\n")
        file.close()
    else:
        try:
            file = open(filePath, '+a')

            file.write(f"Tempo de Execucaoo total: {results['Runtime']}\n")
            file.write(f"MIPGAP: {results['MIPGap']}\n")
            file.write(f"Model Obj: {results['objVal']}\n")
            file.write(f"Distancia total: {results['distancia_total']}\n")
            file.write("-" * 60 + '\n')
            file.write(f"{'Depósito':<10} {'Veículo':<10} {'Capacidade Utilizada':<20} {'Clientes Atendidos'}\n")
            file.write("-" * 60 + '\n')

            for line in results['lines']:
                file.write(line + '\n')

            file.close()
        except:
            print(f'Erro ao escrever no arquivo {filePath}!!!')
        finally:
            return True   

def solve_model(filename, execution_minutes:int = 1, write_results: int = 0):
    problem_type, num_vehicles, num_customers, num_depots, p_depots, p_customers = read_mdvrp_instance(filename)
    
    # Capacidade dos veículos em cada depósito
    Q = [depot['max_load'] for depot in p_depots]
    # Demanda dos clientes
    demands = [customer['demand'] for customer in p_customers[:-num_depots]]
    
    customers = list(range(0, num_customers))
    vehicles = list(range(0, num_vehicles))
    depots = list(range(0, num_depots))

    nodes = customers + [depot + len(customers) for depot in depots]

    # Matriz de distâncias (clientes + depósitos)
    distance_matrix = calculate_distance_matrix(p_customers[:-num_depots], p_depots)

    # Inicializando o modelo
    model = grb.Model("MDVRP")
    model.Params.TimeLimit = 60 * execution_minutes
    model.Params.OutputFlag = 1

    # Variáveis de decisão
    z = model.addVars(product(customers, vehicles, depots), vtype=grb.GRB.BINARY, name="z")
    y = model.addVars(product(vehicles, depots), vtype=grb.GRB.BINARY, name="y")
    x = model.addVars(product(nodes, nodes, vehicles), vtype=grb.GRB.BINARY, name='x')

    # Penalidade por alocar um veículo a um depósito
    vehicle_penalty = 1000

    # Função objetivo: Minimizar a distância total percorrida e penalidade por veículo alocado
    model.setObjective(
        grb.quicksum(distance_matrix[i][j] * x[i, j, k] for i, j, k in product(nodes, nodes, vehicles)) +
        grb.quicksum(vehicle_penalty * y[k, d] for k, d in product(vehicles, depots)),
        grb.GRB.MINIMIZE
    )

    # Restrições
    # 1. Cada cliente deve ser atendido por exatamente um veículo de algum depósito
    model.addConstrs(
        (grb.quicksum(z[i, k, d] for k, d in product(vehicles, depots)) == 1)
        for i in customers
    )

    for i in nodes:
        for k in vehicles:
            x[i, i, k].setAttr(grb.GRB.Attr.UB, 0.0)

    model.addConstrs(
        (grb.quicksum(x[i, j, k] for j in nodes if i != j) == 1)  
        for i, k in product(nodes, vehicles)
    )

    # 2. Capacidade do veículo associada ao depósito
    model.addConstrs(
        (grb.quicksum(demands[i] * z[i, k, d] for i in customers) <= Q[d] * y[k, d])
        for k, d in product(vehicles, depots)
    )

    # 3. Restrição de alocação do veículo: um veículo só pode ser utilizado se estiver alocado a um depósito
    model.addConstrs(
        (z[i, k, d] <= y[k, d] for i in customers for k in vehicles for d in depots),
        name="link_z_y"
    )

    # 5. Subtour elimination constraints (eliminação de subtours para garantir rotas válidas)
    u = model.addVars(nodes, vtype=grb.GRB.CONTINUOUS, name="u")
    n = len(customers)
    model.addConstrs(
        (u[i] - u[j] + (n-1) * x[i, j, k] <= n-2 for i, j, k in product(customers, customers, vehicles) if i != j),
        name="subtour_elimination"
    )

    model._edges_weights = distance_matrix
    model._num_customers = num_customers
    model._demands = demands
    model._Q = Q
    model._z = z
    model._y = y

    # Otimizar o modelo
    model.optimize()

    try:
        # Exibir os resultados finais em forma de tabela
        results = display_results(model, customers, depots, vehicles, depots, demands)

        # Valida o parâmetro de criar um arquivo para os resultados
        if write_results == 1:
            write_result_file(path=RESULTS_DIR, filename=filename.split('/')[-1], results=results)
    except:
        # Valida o parâmetro de criar um arquivo para os resultados
        if write_results == 1:
            write_result_file(path=RESULTS_DIR, filename=filename.split('/')[-1], results=None)
    

def get_instancias(path: str):
    return os.listdir(path)

__name__ = str('__main__')

if __name__ == "__main__":
    instancias = get_instancias(path=INSTANCES_DIR)
    for instancia in instancias:
        filePath = f'{INSTANCES_DIR}/{instancia}'
        print(f'Instancia: {instancia}\n')
        solve_model(filename=filePath, write_results=1, execution_minutes=30)

if __name__ == '__teste__':
    filePath = r'../datasets/C-mdvrp/Ajustados/p01-medio'
    resultado = solve_model(filePath, execution_minutes=1, write_results=0)