import gurobipy as grb
from collections import defaultdict
import logging
from itertools import combinations, product
import math
import os
import time

# Constantes
INSTANCES_DIR = r'../datasets/C-mdvrp/Ajustados'
RESULTS_DIR = r'./Resultados/Benders'

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
    
def mycallback(model, where):
    if where == grb.GRB.Callback.MIPSOL:
        try:
            generate_opt_cuts(model)
        except Exception:
            logging.exception("Exception occurred in MIPSOL callback")
            model.terminate()


def generate_opt_cuts(model):
    zvalues = model.cbGetSolution(model._z)
    yvalues = model.cbGetSolution(model._y)
    clients = defaultdict(lambda: defaultdict(list))

    for (i, k, d) in zvalues.keys():
        if zvalues[i, k, d] > 0.5:
            clients[k][d].append(i)

    for (k, dep_clients) in clients.items():
        for d, cs in dep_clients.items():
            if len(cs) > 1:
                obj = solve_tsp(model._edges_weights, cs, d, model._num_customers, model._demands, model._Q[d])

                expr = grb.quicksum((1 - model._z[i, k, d]) for i in cs)
                model.cbLazy(model._alpha[k, d] >= obj - obj * expr)

                # Generate feasibility cuts for vehicle load
                load = sum(model._demands[i] * zvalues[i, k, d] for i in cs)
                if load >= model._Q[d]:
                    expr = grb.quicksum(model._demands[i] * model._z[i, k, d] for i in cs)
                    model.cbLazy(expr <= model._Q[d] * yvalues[k, d])


def solve_tsp(edges_weights, clients, depot, num_customers, demands, max_load):
    model = grb.Model("TSP-DL-subtours-Load")
    model.Params.OutputFlag = 0

    nodes = [num_customers + depot] + clients # Inicia-se com o depósito
    arcs = list(product(nodes, nodes))

    x = model.addVars(arcs, vtype=grb.GRB.BINARY, name='x')
    u = model.addVars(nodes, ub=len(nodes)-2, vtype=grb.GRB.CONTINUOUS, name='u')

    # Objective: Minimize total travel cost    
    model.setObjective(
    grb.quicksum(x[i, j] * round(edges_weights[i][j], 2) for (i, j) in x.keys()), 
    grb.GRB.MINIMIZE)

    for i in nodes:
        x[i, i].setAttr(grb.GRB.Attr.UB, 0.0)

    # Ensure that each node (customer or depot) is visited exactly once
    model.addConstrs(
        (grb.quicksum(x[i, j] for j in nodes) == 1)
        for i in nodes)

    model.addConstrs(
        (grb.quicksum(x[j, i] for j in nodes) == 1)
        for i in nodes)

    # Subtour elimination constraints
    n = len(nodes)
    model.addConstrs(
        (u[i] - u[j] + (n-1)*x[i, j] + (n-3)*x[j, i] <= n-2)
        for (i, j) in product(nodes[1:], nodes[1:])
        if i != j
    )

    model.addConstrs(
        (x[i, j] + x[j, i] <= 1)
        for (i, j) in combinations(nodes, 2)
    )

    # Vehicle load constraints
    for i in clients:
        model.addConstr(
            grb.quicksum(demands[i] * x[i, j] for j in nodes) <= max_load,
            name=f"vehicle_load_{i}")

    model.optimize()

    return model.objVal



def solve_model(filename, execution_minutes: int = 1, write_results: int = 1):
    # Reading instance
    problem_type, num_vehicles, num_customers, num_depots, p_depots, p_customers = read_mdvrp_instance(filename)
    
    Q = [depot['max_load'] for depot in p_depots]
    demands = [customer['demand'] for customer in p_customers[:-num_depots]]
    customers = list(range(0, num_customers))
    K = num_vehicles
    vehicles = list(range(0, K))
    depots = list(range(num_depots))

    # Distance matrix (customers + depots)
    edges_weights = calculate_distance_matrix(p_customers[:-num_depots], p_depots)

    model = grb.Model("MDVRP-Benders-Load")
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = 60 * execution_minutes
    model.Params.OutputFlag = 1 # Remover depois de montar o toy

    # Decision variables
    z = model.addVars(product(customers, vehicles, depots), vtype=grb.GRB.BINARY, name="z")
    y = model.addVars(product(vehicles, depots), vtype=grb.GRB.BINARY, name="y")
    alpha = model.addVars(product(vehicles, depots), lb=0, vtype=grb.GRB.CONTINUOUS, name="alpha")
    load = model.addVars(product(vehicles, depots), lb=0, ub=80.0, vtype=grb.GRB.CONTINUOUS, name="load")

    vehicle_penallity = 1000

    # Objective: Minimize the total cost (travel distance/time)
    model.setObjective(
        grb.quicksum(alpha[k, d] * y[k, d] + vehicle_penallity * y[k, d] for k, d in product(vehicles, depots)), grb.GRB.MINIMIZE)

    # Constraints
    # 1. Each customer must be served by exactly one vehicle from some depot
    model.addConstrs(
        (grb.quicksum(z[i, k, d] for k, d in product(vehicles, depots)) == 1)
        for i in customers)

    # 2. Vehicle load constraint: total demand assigned to a vehicle cannot exceed its capacity
    model.addConstrs(
        (grb.quicksum(demands[i] * z[i, k, d] for i in customers) <= Q[d] * y[k, d])
        for k, d in product(vehicles, depots))

    # 3. No more than the maximum number of vehicles can be allocated to a depot
    model.addConstrs(
        (grb.quicksum(y[v, d] for v in vehicles) <= num_vehicles)
        for d in depots)

    # 4. Linking alpha with travel costs
    for (c, v, d) in product(customers, vehicles, depots):
        obj = round(edges_weights[num_customers + d][c], 2) + round(edges_weights[c][num_customers + d], 2)
        model.addConstr(alpha[v, d] >= obj * z[c, v, d], name=f"bound_alpha_{c}_{v}_{d}")

    # 5. Ensure alpha is only activated if the vehicle is allocated to the depot
    model.addConstrs(
        (alpha[k, d] * y[k, d] >= 0)
        for k, d in product(vehicles, depots))

    # 6. Symmetry breaking constraints
    for d in depots:
        for k in range(1, len(vehicles)):
            model.addConstr(y[k - 1, d] >= y[k, d], name=f"symmetry_y_{k}_{d}")
            model.addConstr(alpha[k - 1, d] >= alpha[k, d], name=f"symmetry_alpha_{k}_{d}")

    model.addConstrs(
        (z[i, k, d] <= y[k, d] for i in customers for k in vehicles for d in depots),
        name="link_z_y")

    model._edges_weights = edges_weights
    model._num_customers = num_customers
    model._demands = demands
    model._Q = Q
    model._z = z
    model._y = y
    model._alpha = alpha
    model._load = load

    # Benders Callback
    model.optimize(mycallback)

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

__name__ = str('__teste__')

if __name__ == "__main__":
    instancias = get_instancias(path=INSTANCES_DIR)
    for instancia in instancias:
        filePath = f'{INSTANCES_DIR}/{instancia}'
        print(f'Instancia: {instancia}\n')
        solve_model(filename=filePath, write_results=1, execution_minutes=30)

if __name__ == '__teste__':
    filePath = r'../datasets/C-mdvrp/Ajustados/p01-pequeno'
    solve_model(filePath, execution_minutes=1, write_results=0)





