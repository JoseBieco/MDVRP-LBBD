import gurobipy as grb
from collections import defaultdict
import logging
from itertools import combinations, product
import math

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
    #return {'type': problem_type, 'm': m, 'n': n, 't': t, 'depots': depots, 'customers': customers}



# def calculate_distance_matrix(customers, depots):
#     Criação da matriz de distâncias
#     distancias_euclidianas = [
#         [
#             math.sqrt((depot['x'] - customer['x']) ** 2 + (depot['y'] - customer['y']) ** 2)
#             for customer in customers
#         ]
#         for depot in depots
#     ]
    
#     return distancias_euclidianas

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

# def vehicles_upperbound(instance):
#     return instance['dimension']


def route_cost_lowerbound(edges_weights):
    total = 0
    n = len(edges_weights)
    for j in range(n):
        total += min(round(edges_weights[i][j]) for i in range(n) if i != j)
    return total


def visit_upperbound(node_idx, edges_weights):
    total = 0
    n = len(edges_weights)
    total += max(round(edges_weights[node_idx][j]) for j in range(n))
    total += max(round(edges_weights[j][node_idx]) for j in range(n))
    return total


def mycallback(model, where):
    if where == grb.GRB.Callback.MIPSOL:
        try:
            generate_opt_cuts(model)
        except Exception:
            logging.exception("Exception occurred in MIPSOL callback")
            model.terminate()


def generate_opt_cuts(model):
    zvalues = model.cbGetSolution(model._z)
    clients = defaultdict(lambda: defaultdict(list))

    for (i, k, d) in zvalues.keys():
        if zvalues[i, k, d] > 0.5:
            clients[k][d].append(i)

    for (k, dep_clients) in clients.items():
        for d, cs in dep_clients.items():
            if len(cs) > 1:
                obj = solve_tsp(model._edges_weights, cs, d, model._num_customers)
                expr = grb.quicksum((1 - model._z[i, k, d]) for i in cs)
                model.cbLazy(model._alpha[k, d] >= obj - obj * expr)


def solve_tsp(edges_weights, clients, depot, num_customers):
    model = grb.Model("TSP-DL-subtours")
    model.Params.OutputFlag = 0
    nodes = [num_customers + depot] + clients
    arcs = list(set(product(nodes, nodes)))

    x = model.addVars(set(arcs), vtype=grb.GRB.BINARY, name=f'x')
    u = model.addVars(nodes, ub=len(nodes)-2, vtype=grb.GRB.CONTINUOUS, name=f'u')

    model.setObjective(
        grb.quicksum(x[i, j] * round(edges_weights[i][j]) for (i, j) in x.keys()), 
        grb.GRB.MINIMIZE)

    for i in nodes:
        x[i, i].setAttr(grb.GRB.Attr.UB, 0)

    model.addConstrs(
        (grb.quicksum(x[i, j] for j in nodes) == 1)
        for i in nodes)

    model.addConstrs(
        (grb.quicksum(x[j, i] for j in nodes) == 1)
        for i in nodes)

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

    model.optimize()

    # if model.status == grb.GRB.OPTIMAL:
    #     return model.objVal
    # else:
    #     return -1
    return model.ObjVal

def solve_model(filename):
   # Leitura da instância
    problem_type, num_vehicles, num_customers, num_depots, depots, p_customers = read_mdvrp_instance(filename)
    
    Q = [depot['max_load'] for depot in depots]
    demands = [customer['demand'] for customer in p_customers[:-4]]
    n = num_customers + num_depots  # Customers + depot
    customers = list(range(0, num_customers))
    K = num_vehicles # * num_depots
    vehicles = list(range(0, K))

    # p_customers[:-4] > retirando os depósitos da lista de p_customers para 
    # a matriz manter seu tamanho correto de (n + t) X (n + t)
    edges_weights = calculate_distance_matrix(p_customers[:-4], depots)

    #depots_edges_weights = calculate_distance_matrix(p_customers[:-4], depots)
    
    m = num_depots  # Number of depots
    depots = list(range(m))
    
    model = grb.Model("MDVRP-Benders-1")
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = 60 * 1
    
    # z[i, k, d] -> 1 if vehicle k serves customer i starting from depot d
    z = model.addVars(product(customers, vehicles, depots), vtype=grb.GRB.BINARY, name="z")
    
    # y[k, d] -> 1 if vehicle k is assigned to depot d
    y = model.addVars(product(vehicles, depots), vtype=grb.GRB.BINARY, name='y')
    
    # alpha[k, d] -> Cost associated with vehicle k assigned to depot d
    alpha = model.addVars(product(vehicles, depots), lb=0, vtype=grb.GRB.CONTINUOUS, name='alpha')
    
    # model.setObjective(
    #     grb.quicksum(alpha[k, d] for k, d in product(vehicles, depots)) + 
    #     1000 * grb.quicksum(y[v, d] for v, d in product(vehicles, depots)), grb.GRB.MINIMIZE)
    
    model.setObjective(
        grb.quicksum(alpha[v, d] * y[v, d] for v, d in product(vehicles, depots)), grb.GRB.MINIMIZE)
    

    # for d in depots:
    #     for v in vehicles:
    #         model.addConstr(y[v, d] <= 1)

    # # Um depósito não pode ter mais que num_veiculos alocados
    # model.addConstrs(
    #     (grb.quicksum(y[v, d] for v in vehicles) <= num_vehicles) for d in depots
    # )


    # Cada cliente deve ser atendido por exatamente um veículo de algum depósito
    model.addConstrs(
        (grb.quicksum(z[i, k, d] for k, d in product(vehicles, depots)) == 1)
        for i in customers)


    # Capacidade do depósito
    model.addConstrs(
        (grb.quicksum(demands[i] * z[i, k, d] for i in customers) <= Q[d])
        for k, d in product(vehicles, depots))

    # Restrição de capacidade do veículo
    model.addConstrs(
        (grb.quicksum(demands[i] * z[i, k, d] for i in customers) <= Q[d] * y[k, d])
        for k, d in product(vehicles, depots))

    
    # Vínculo entre `alpha` e os custos de viagem
    # `edges_weights[c][num_customers + d]` adicionado desta forma para "andar" corretamente entre a matriz
    # Restrições vinculando alpha ao custo de viagem e ao uso do veículo
    for (c, v, d) in product(customers, vehicles, depots):
        obj = round(edges_weights[num_customers + d][c], 2) + round(edges_weights[c][num_customers + d], 2)
        model.addConstr(alpha[v, d] >= obj * z[c, v, d], name=f"bound_alpha_{c}_{v}_{d}")

    ## Garantindo que alpha seja ativado apenas se o veículo for alocado ao depósito
    # 5. Ensure alpha is only activated if the vehicle is allocated to the depot
    model.addConstrs(
        (alpha[k, d] * y[k, d] >= 0)
        for k, d in product(vehicles, depots))

    # Restrições para quebrar a simetria
    for (k, d) in product(vehicles[1:], depots):
        model.addConstr(y[k - 1, d] >= y[k, d])
        model.addConstr(alpha[k - 1, d] >= alpha[k, d])

    
    # # Restrição de lower bound em alpha ** Talvez deva ser lower bound em y ?
    # model.addConstrs(
    #     (grb.quicksum(alpha[k, d] for k in vehicles) >= 0 * route_cost_lowerbound(edges_weights))
    #     for d in depots)
    # model.addConstrs(
    #     (grb.quicksum(y[v, d] for v in vehicles) <= num_vehicles) for d in depots
    # )


    model._edges_weights = edges_weights
    model._num_customers = num_customers
    model._z = z
    model._y = y
    model._alpha = alpha
    model.optimize(mycallback)

    # Tentar identificar restrições inconsistentes
    if model.status == grb.GRB.INFEASIBLE:
        print("Modelo Inviável! Identificando IIS...")
        model.computeIIS()
        model.write("model.ilp")

    values = model.getAttr('X', model.getVars())
    names = model.getAttr('VarName', model.getVars())

    print(f'Model Obj: {model.objVal}')
    for (val, name) in zip(values, names):
        if val > 0.5:
            print(name, val)

if __name__ == "__main__":
    filename = r'../datasets/C-mdvrp/p01'
    solve_model(filename)


