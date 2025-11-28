import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

# ==========================================
# CONFIGURACIÓN Y PARÁMETROS (TABLA I) [cite: 258]
# ==========================================


class SimParams:
    def __init__(self):
        self.dt = 0.01          # Delta t: 10 ms
        self.T = 10             # Scheduling period length
        self.BW = 180e3         # Bandwidth: 180 KHz
        self.N0_dBm = -174      # Noise spectral density dBm/Hz
        self.N0 = 10**((self.N0_dBm - 30)/10) * self.BW  # Noise power in Watts
        self.P_max_dBm = 35     # Max power dBm
        self.P_max = 10**((self.P_max_dBm - 30)/10)  # Max power in Watts
        self.R_th_dB = 10       # SINR threshold dB
        self.R_th = 10**(self.R_th_dB/10)  # SINR threshold linear
        self.o_max = 2          # Max allowed users per slot
        self.L = 5              # Vehicle length (m)
        self.d_des = 10         # Desired spacing (m)
        self.u_max = 2.0        # Max accel (m/s^2)
        self.u_min = -2.0       # Min accel (m/s^2)
        self.e_th = 0.01        # Error bound (m) [cite: 262]

        # Pesos para la función de costo (No especificados en el paper, asumidos estándar)
        self.W = np.diag([5.0, 1.0])  # Peso velocidad, Peso posición


PARAMS = SimParams()

# ==========================================
# CLASES DE VEHÍCULO Y CONTROLADOR
# ==========================================


class Vehicle:
    def __init__(self, id, initial_pos, initial_vel):
        self.id = id
        self.p = initial_pos
        self.v = initial_vel
        self.u = 0.0  # Control input actual

        # Historial para plots
        self.h_p = [self.p]
        self.h_v = [self.v]
        self.h_error = []

        # Estado estimado del predecesor (para cuando no hay comunicación) [cite: 159]
        self.pred_state_est = None  # {p, v, u, timestamp}

    def update_dynamics(self, u_cmd):
        """Ec. (5) y (6): Modelo cinemático discreto [cite: 97]"""
        # Saturación del control input
        self.u = np.clip(u_cmd, PARAMS.u_min, PARAMS.u_max)

        self.p = self.p + self.v * PARAMS.dt + 0.5 * self.u * (PARAMS.dt**2)
        self.v = self.v + self.u * PARAMS.dt

        self.h_p.append(self.p)
        self.h_v.append(self.v)

    def calculate_error(self, p_pred, v_pred):
        """Ec. (3) y (4): Tracking error real [cite: 93]"""
        e_v = self.v - v_pred
        e_p = p_pred - self.p - PARAMS.d_des - PARAMS.L
        return np.array([e_v, e_p])

# ==========================================
# LÓGICA DE CONTROL (DISTRIBUTED POLICY)
# ==========================================


def get_control_input(vehicle, predecessor, received_info_packet, current_time_step):
    """
    Calcula u_i^t resolviendo el problema de optimización (18).
    Si recibe paquete, actualiza estimación. Si no, estima basado en historia.
    """

    # 1. Actualizar conocimiento del predecesor
    if received_info_packet is not None:
        # Se recibió comunicación: {p, v, u, t_sent}
        vehicle.pred_state_est = received_info_packet

    if vehicle.pred_state_est is None:
        # Caso inicial: Asumir estado ideal si nunca hubo comunicación
        p_pred_est = vehicle.p + PARAMS.d_des + PARAMS.L
        v_pred_est = vehicle.v
        u_pred_est = 0
    else:
        # 2. Estimación del estado del predecesor [cite: 161-163]
        p_known, v_known, u_known, t_known = vehicle.pred_state_est
        delta_steps = current_time_step - t_known
        delta_t_est = delta_steps * PARAMS.dt

        # Ec (14) y (15)
        p_pred_est = p_known + v_known * delta_t_est + \
            0.5 * u_known * (delta_t_est**2)
        v_pred_est = v_known + u_known * delta_t_est
        u_pred_est = u_known  # Se asume constante

    # 3. Predicción del error en t+1 [cite: 167-168]
    # Función de costo para minimizar e_{est}^{t+1} * W * e_{est}^{t+1}
    def cost_function(u_try):
        # Dinámica propia proyectada
        p_next = vehicle.p + vehicle.v * \
            PARAMS.dt + 0.5 * u_try * (PARAMS.dt**2)
        v_next = vehicle.v + u_try * PARAMS.dt

        # Dinámica estimada del predecesor en t+1
        p_pred_next = p_pred_est + v_pred_est * \
            PARAMS.dt + 0.5 * u_pred_est * (PARAMS.dt**2)
        v_pred_next = v_pred_est + u_pred_est * PARAMS.dt

        e_v_est = v_next - v_pred_next
        e_p_est = p_pred_next - p_next - PARAMS.d_des - PARAMS.L

        e_vec = np.array([e_v_est, e_p_est])
        return e_vec.T @ PARAMS.W @ e_vec

    # Resolver optimización (QP simple)
    res = minimize(cost_function, x0=0.0, bounds=[
                   (PARAMS.u_min, PARAMS.u_max)])
    return res.x[0]

# ==========================================
# ALGORITMO DE ASIGNACIÓN DE RECURSOS
# ==========================================


def run_resource_allocation(platoon, current_period_start_step):
    """
    Ejecuta el Algoritmo de dos etapas (Scheduling + Power) antes de cada periodo T.
    [cite: 184]
    """
    N = len(platoon) - 1  # Número de MVs
    allocation_matrix = np.zeros((N, PARAMS.T))  # [Vehicle_idx, Time_slot]
    power_matrix = np.zeros((N, PARAMS.T))

    # --- ETAPA 1: USER SCHEDULING (Algoritmo 1)  ---
    for i in range(N):
        mv = platoon[i+1]  # MV index starts at 1
        pred = platoon[i]

        # Calcular error actual (al inicio del periodo)
        err = mv.calculate_error(pred.p, pred.v)
        e_p_abs = abs(err[1])

        if e_p_abs <= PARAMS.e_th:
            f_i = 0
        else:
            # Ec (9): Frecuencia basada en capacidad de corrección
            correction_ability = 0.5 * PARAMS.u_max * \
                ((PARAMS.T * PARAMS.dt)**2)
            # Evitar división por cero o valores muy pequeños
            if e_p_abs < 1e-6:
                f_i = 0
            else:
                f_i = min(np.ceil(e_p_abs / correction_ability), PARAMS.T)

        if f_i > 0:
            c_i = int(PARAMS.T / f_i)
            # Determinar t_1 (primer slot) basado en historia (simplificado a 0 para simulación inicial)
            t_n = 0
            count = 0

            # Lógica de asignación y búsqueda "Z" [cite: 206]
            candidates = list(range(PARAMS.T))  # Todos los slots del periodo

            while count < f_i and t_n < PARAMS.T:
                # Buscar slot candidato
                search_order = [t_n]
                # Agregar vecinos (Búsqueda Z: t_n-1, t_n+1, ...)
                for offset in range(1, PARAMS.T):
                    if t_n - offset >= 0:
                        search_order.append(t_n - offset)
                    if t_n + offset < PARAMS.T:
                        search_order.append(t_n + offset)

                slot_found = -1
                for slot in search_order:
                    users_in_slot = np.sum(allocation_matrix[:, slot])
                    if users_in_slot < PARAMS.o_max:
                        slot_found = slot
                        break

                if slot_found != -1:
                    allocation_matrix[i, slot_found] = 1
                    count += 1
                    t_n = slot_found + c_i
                else:
                    # No se encontró espacio, avanzar al siguiente ideal
                    t_n += 1

    # --- ETAPA 2: POWER ALLOCATION [cite: 213-247] ---
    total_power_period = 0
    successful_comms = 0

    for t in range(PARAMS.T):
        users_scheduled = np.where(allocation_matrix[:, t] == 1)[0]
        if len(users_scheduled) == 0:
            continue

        # Simular ganancias de canal (Path loss + Rayleigh)
        # Asumimos distancia d_des + error para path loss
        h_gains = []       # Ganancia directa h_{i, i-1}
        h_interf = []      # Matriz de interferencias

        # Generar canales para este slot
        # Nota: El paper no da formula exacta de canal, usamos modelo estándar log-distancia
        # h = d^(-3) * |Rayleigh|^2

        user_channel_data = []

        for u_idx in users_scheduled:
            mv = platoon[u_idx+1]
            pred = platoon[u_idx]
            dist = pred.p - mv.p
            # Rayleigh fading power
            h_direct = (dist**-3) * np.random.exponential(1.0)
            user_channel_data.append(
                {'id': u_idx, 'h_direct': h_direct, 'dist': dist})

        # Orden de Decodificación NOMA: Ascendente de ganancia de canal de INTERFERENCIA [cite: 221]
        # Para simplificar (como dice el paper para 2 usuarios), asumimos que
        # mayor distancia = menor canal.
        # Si hay 2 usuarios (j, k) con j < k (j está atrás de k), la interferencia
        # de (k-1) sobre j es h_{j, k-1}.

        # Ordenamos por posición en el pelotón para aproximar la lógica de interferencia
        # El paper dice: "optimal decoding order is the ascending order in interference channel gain"
        sorted_users = sorted(
            user_channel_data, key=lambda x: x['dist'], reverse=True)
        # Mayor distancia (más atrás en pelotón) -> Menor interferencia -> Primero en orden de decodificación

        # Asignación de potencia Greedy [cite: 244]
        current_interference = 0  # Interferencia acumulada de usuarios con más potencia

        for u_data in sorted_users:
            u_idx = u_data['id']
            h = u_data['h_direct']

            # P_i = R_th * (Interferencia + Ruido) / h
            # Interferencia incluye señales de otros usuarios NOMA (aún no calculados en orden inverso?
            # NOMA downlink vs uplink logic. El paper usa Eq 11.)
            # En V2V broadcast/unicast, el paper asume SIC.
            # Ec (11): Gamma = (p*h) / (I + N0).
            # I incluye otros usuarios en el mismo slot que NO son cancelados.
            # Asumimos que el usuario actual cancela a los que tienen MENOR ganancia de interferencia (los anteriores en la lista)

            # Nota: El paper simplifica la ec. 244 asumiendo orden.
            # P_1 (más debil) = R_th * N / h1
            # P_2 (más fuerte) = R_th * (P1*h12 + N) / h2

            # Implementación simple basada en eq 244 para 2 usuarios
            required_p = 0
            if len(sorted_users) == 1:
                required_p = (PARAMS.R_th * PARAMS.N0) / h
            else:
                # Si hay multiples, este loop necesita ajuste estricto a ec 244.
                # Asumimos sorted_users[0] es user 1 (menor interferencia), sorted_users[1] user 2.
                if u_data == sorted_users[0]:
                    required_p = (PARAMS.R_th * PARAMS.N0) / h
                else:
                    prev_p = power_matrix[sorted_users[0]['id'], t]
                    # Estimamos interferencia cruzada h_12 (distancia entre user 1 y emisor para user 2? No, es h_{1,2})
                    # Simplificación: Usamos una ganancia de canal aleatoria para la interferencia cruzada
                    h_cross = ((PARAMS.d_des*2)**-3) * \
                        np.random.exponential(1.0)
                    required_p = (
                        PARAMS.R_th * (prev_p * h_cross + PARAMS.N0)) / h

            if required_p <= PARAMS.P_max:
                power_matrix[u_idx, t] = required_p
                total_power_period += required_p
                successful_comms += 1
            else:
                power_matrix[u_idx, t] = 0  # Falla comunicación por potencia

    return allocation_matrix, power_matrix, total_power_period, successful_comms

# ==========================================
# LOOP PRINCIPAL DE SIMULACIÓN
# ==========================================


def run_simulation():
    # Inicialización del Pelotón
    N_vehicles = 12  # [cite: 262]
    platoon = []

    # Posicionar vehículos separados por d_des
    for i in range(N_vehicles + 1):  # +1 por el líder
        pos = (N_vehicles - i) * (PARAMS.L + PARAMS.d_des)
        vel = 20.0  # Velocidad crucero 20 m/s
        platoon.append(Vehicle(i, pos, vel))

    total_steps = int(30.0 / PARAMS.dt)  # 30 segundos [cite: 254]

    # Métricas
    total_power_history = []
    avg_spacing_error_history = []

    current_allocation = None
    current_power = None

    # Loop temporal
    print("Iniciando simulación del paper...")
    for t_step in range(total_steps):
        time = t_step * PARAMS.dt

        # 1. Perfil de Velocidad del Líder (Perturbación) [cite: 254-255]
        # 0-10s: Decelerar de 20 a 15 (a = -0.5 m/s^2 para cuadrar distancias o -1 segun paper?)
        # El paper dice decelerar de 20 a 15 en 10s. (20-15)/10 = 0.5 m/s^2. El paper dice "constant acceleration as -1",
        # lo cual llevaría a 10m/s. Usaremos la descripción de velocidades: a = -0.5.
        if 0 <= time < 10:
            platoon[0].u = -0.5
        elif 10 <= time < 15:
            platoon[0].u = 0  # Steady state
        elif 15 <= time < 25:
            platoon[0].u = 0.5  # Acelerar de 15 a 20
        else:
            platoon[0].u = 0  # Steady state

        # Actualizar dinámica del líder
        platoon[0].update_dynamics(platoon[0].u)

        # 2. Asignación de Recursos (Cada T slots)
        slot_in_period = t_step % PARAMS.T
        if slot_in_period == 0:
            current_allocation, current_power, p_consum, _ = run_resource_allocation(
                platoon, t_step)
            # Acumular consumo de potencia para métricas (simplificado a lineal)
            if len(total_power_history) > 0:
                total_power_history.append(total_power_history[-1] + p_consum)
            else:
                total_power_history.append(p_consum)
        else:
            # Mantener valor anterior para plot
            total_power_history.append(total_power_history[-1])

        # 3. Control de Miembros (MVs)
        step_errors = []
        for i in range(1, len(platoon)):
            mv = platoon[i]
            pred = platoon[i-1]

            # Verificar si tiene asignado slot de comunicación en este t_step
            # Indices en allocation son [i-1] porque allocation es solo para MVs
            has_comm = False
            packet = None

            if current_allocation[i-1, slot_in_period] == 1:
                # Verificar si la potencia fue suficiente (éxito de transmisión)
                if current_power[i-1, slot_in_period] > 0:
                    has_comm = True
                    # Enviar paquete {p, v, u, timestamp}
                    packet = (pred.p, pred.v, pred.u, t_step)

            # Ejecutar controlador distribuido
            u_cmd = get_control_input(mv, pred, packet, t_step)
            mv.update_dynamics(u_cmd)

            # Calcular error para métricas
            err = mv.calculate_error(pred.p, pred.v)
            step_errors.append(abs(err[1]))  # Spacing error absoluto

        avg_spacing_error_history.append(np.mean(step_errors))

    return avg_spacing_error_history, total_power_history


# ==========================================
# EJECUCIÓN Y GRÁFICAS
# ==========================================
spacing_error, power_consum = run_simulation()

time_axis = np.arange(len(spacing_error)) * PARAMS.dt

plt.figure(figsize=(12, 5))

# Plot Error (Replica Fig. 4)
plt.subplot(1, 2, 1)
plt.plot(time_axis, spacing_error, label='Proposed Scheme (Python Sim)')
plt.axhline(y=PARAMS.e_th, color='r', linestyle='--', label='Error Bound')
plt.xlabel('Time (s)')
plt.ylabel('Average Absolute Spacing Error (m)')
plt.title('Performance de Control (Fig. 4)')
plt.legend()
plt.grid(True)

# Plot Power (Replica Fig. 3)
plt.subplot(1, 2, 2)
plt.plot(time_axis, power_consum, label='Proposed Scheme')
plt.xlabel('Time (s)')
plt.ylabel('Total Power Consumption (J)')
plt.title('Consumo de Potencia (Fig. 3)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
