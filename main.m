%% Joint Platoon Control and Resource Allocation (NOMA-V2V)
% Replicación del Paper - Esqueleto en MATLAB
clear; clc; close all;

%% 1. CONFIGURACIÓN Y PARÁMETROS (Tabla I)
% Parámetros de Simulación 
Sim.dt = 0.01;              % 10 ms
Sim.T_scheduling = 10;      % Periodo de agendamiento (slots)
Sim.TotalTime = 30;         % 30 segundos
Sim.Steps = Sim.TotalTime / Sim.dt;

% Parámetros de Comunicación
Comm.BW = 180e3;            % 180 KHz
Comm.N0_dBm = -174;         % dBm/Hz
Comm.N0 = 10^((Comm.N0_dBm - 30)/10) * Comm.BW; % Potencia de ruido en Watts
Comm.P_max_dBm = 35;        
Comm.P_max = 10^((Comm.P_max_dBm - 30)/10);     % Watts
Comm.R_th_dB = 10;
Comm.R_th = 10^(Comm.R_th_dB/10);               % Umbral SINR lineal
Comm.O_max = 2;             % Máx usuarios NOMA por slot

% Parámetros del Platoon y Vehículo
Platoon.N = 12;             % Número de vehículos miembros (MVs)
Platoon.d_des = 10;         % Espaciamiento deseado (m)
Platoon.len = 5;            % Longitud del vehículo (m)
Platoon.e_th = 0.01;        % Error bound
Platoon.u_max = 2;          % Aceleración máx (m/s^2)
Platoon.u_min = -2;         % Aceleración mín (m/s^2)

%% 2. INICIALIZACIÓN DE ESTADOS
% Matrices de estado: Filas = Vehículos (0=Líder, 1..N=Miembros), Col = Tiempo
% Inicializamos con velocidad crucero 20 m/s y posiciones separadas por d_des+len
POS = zeros(Platoon.N + 1, Sim.Steps);
VEL = zeros(Platoon.N + 1, Sim.Steps);
ACC = zeros(Platoon.N + 1, Sim.Steps);

% Estado Inicial (t=1)
v_cruise = 20; 
for i = 1:(Platoon.N + 1)
    veh_idx = i - 1; % 0 es líder
    VEL(i, 1) = v_cruise;
    % El líder está en la posición más adelantada
    POS(i, 1) = (Platoon.N - veh_idx) * (Platoon.d_des + Platoon.len);
end

% Memoria de "Última Información Recibida" para cada MV
% Estructura: [Pos, Vel, Acc, Tiempo_de_paquete]
LastInfo = zeros(Platoon.N, 4); 
% Inicializar con datos verdaderos al inicio
for i = 1:Platoon.N
    leader_idx = i; % Predecesor es i (porque i+1 es el MV actual en matriz 1-based)
    LastInfo(i, :) = [POS(leader_idx,1), VEL(leader_idx,1), 0, 0];
end

%% 3. BUCLE PRINCIPAL DE SIMULACIÓN
fprintf('Iniciando simulación de %d pasos...\n', Sim.Steps);

for t = 1 : Sim.Steps - 1
    current_time = t * Sim.dt;
    
    % --- A. DINÁMICA DEL LÍDER (Perfil de Perturbación) ---
    % 0-10s: Desacelera a -1 m/s^2
    % 15-25s: Acelera a 1 m/s^2
    u_leader = 0;
    if current_time <= 10
        u_leader = -1;
    elseif current_time >= 15 && current_time <= 25
        u_leader = 1;
    end
    ACC(1, t) = u_leader;
    
    % --- B. ASIGNACIÓN DE RECURSOS (Cada T slots) ---
    % Se ejecuta al inicio de cada periodo de agendamiento
    if mod(t-1, Sim.T_scheduling) == 0
        % 1. Calcular errores actuales de todos los MVs
        errors = calculate_errors(POS(:,t), VEL(:,t), Platoon);
        
        % 2. Ejecutar Algoritmo 1: User Scheduling 
        UserSchedule = run_stage_1_scheduling(errors, Sim, Platoon, Comm);
    end
    
    % --- C. COMUNICACIÓN Y NOMA (Cada slot) ---
    % Determinar qué usuarios transmiten en este slot relativo (1 a T)
    slot_idx = mod(t-1, Sim.T_scheduling) + 1;
    scheduled_users = find(UserSchedule(:, slot_idx) == 1);
    
    if ~isempty(scheduled_users)
        % 1. Generar Canal (H) y Ruido
        % [TODO]: Implementar modelo Rayleigh Fading aquí
        % H = abs((randn(N,1) + 1i*randn(N,1))/sqrt(2)); ...
        H_gains = ones(length(scheduled_users), 1); % Placeholder: Ganancia Unitaria
        
        % 2. Ejecutar Stage 2: Power Allocation
        % Asigna potencias basadas en SINR target y NOMA decoding order
        allocated_powers = run_stage_2_power(scheduled_users, H_gains, Comm);
        
        % 3. Actualizar información si la comunicación es exitosa
        for k = 1:length(scheduled_users)
            u_id = scheduled_users(k);
            % Si potencia > 0 y < Pmax, asumimos éxito (simplificado)
            if allocated_powers(k) > 0
                % El MV 'u_id' recibe info de su predecesor 'u_id-1'
                pred_real_idx = u_id; % En matriz MATLAB, predecesor es índice u_id
                LastInfo(u_id, :) = [POS(pred_real_idx, t), VEL(pred_real_idx, t), ACC(pred_real_idx, t), current_time];
            end
        end
    end
    
    % --- D. CONTROL DISTRIBUIDO Y FÍSICA DE MVs ---
    % Actualizar líder físicamente
    [POS(1,t+1), VEL(1,t+1)] = update_physics(POS(1,t), VEL(1,t), ACC(1,t), Sim.dt);
    
    for i = 1:Platoon.N
        mv_idx = i + 1; % Índice en matriz (2 a N+1)
        
        % 1. Estimación del estado del predecesor 
        % Si no hubo comunicación hoy, proyectar usando LastInfo
        est_state = estimate_predecessor_state(LastInfo(i,:), current_time, Sim.dt);
        
        % 2. Calcular Control Óptimo (QP)
        % Resolver Ec. 18 usando quadprog
        u_opt = solve_control_qp(POS(mv_idx,t), VEL(mv_idx,t), est_state, Platoon, Sim.dt);
        
        % Guardar input y actualizar física
        ACC(mv_idx, t) = u_opt;
        [POS(mv_idx,t+1), VEL(mv_idx,t+1)] = update_physics(POS(mv_idx,t), VEL(mv_idx,t), u_opt, Sim.dt);
    end
end

%% GRÁFICAS DE RESULTADOS (Fig 2, 3, 4)
figure; 
subplot(3,1,1); plot(POS'); title('Posiciones'); grid on;
subplot(3,1,2); plot(VEL'); title('Velocidades'); grid on;
% Calcular error de espaciamiento real para graficar
spacing_errors = zeros(Platoon.N, Sim.Steps);
for t=1:Sim.Steps
    for i=1:Platoon.N
        spacing_errors(i,t) = (POS(i,t) - POS(i+1,t)) - (Platoon.d_des + Platoon.len);
    end
end
subplot(3,1,3); plot(mean(abs(spacing_errors), 1)); 
title('Average Absolute Spacing Error (Fig 4 replication)'); 
xlabel('Time Slot'); ylabel('Error (m)'); grid on;


%% --- FUNCIONES LOCALES ---

function [p_next, v_next] = update_physics(p, v, u, dt)
    % Ec. 5 y 6 
    v_next = v + u * dt;
    p_next = p + v * dt + 0.5 * u * dt^2;
end

function u_opt = solve_control_qp(p_self, v_self, pred_est, Platoon, dt)
    % Implementación de Ec. 18: min e(t+1)' * W * e(t+1)
    % Esto es equivalente a min 0.5 * u' * H * u + f' * u
    
    % Definir pesos W (diagonal)
    W = diag([1, 1]); % Pesos para [error_vel, error_pos]
    
    % Recuperar estimado del predecesor para t+1
    p_pred_next = pred_est.p_next;
    v_pred_next = pred_est.v_next;
    
    % Expresar e(t+1) en función de u(t)
    % e_v(t+1) = v_pred(t+1) - (v_self + u*dt)
    % e_p(t+1) = p_pred(t+1) - (p_self + v_self*dt + 0.5*u*dt^2) - d - l
    
    % Definir constantes A y B tal que Error = A*u + B
    % Error Vector = [e_v; e_p]
    A_vec = [-dt; -0.5*dt^2];
    
    term_v = v_pred_next - v_self;
    term_p = p_pred_next - p_self - v_self*dt - Platoon.d_des - Platoon.len;
    B_vec = [term_v; term_p];
    
    % Formular matrices para quadprog: min (Au+B)'W(Au+B)
    % Expansion: u'(A'WA)u + 2(B'WA)u + constant
    H_qp = 2 * (A_vec' * W * A_vec);
    f_qp = 2 * (A_vec' * W * B_vec);
    
    % Restricciones: u_min <= u <= u_max
    lb = Platoon.u_min;
    ub = Platoon.u_max;
    
    % Opciones para silenciar output del optimizador
    options = optimoptions('quadprog','Display','off');
    
    % Resolver
    u_opt = quadprog(H_qp, f_qp, [], [], [], [], lb, ub, [], options);
end

function est = estimate_predecessor_state(last_info, t_now, dt)
    % Ec. 14 y 15 
    p_last = last_info(1);
    v_last = last_info(2);
    u_last = last_info(3);
    t_rx = last_info(4);
    
    delta_t = t_now - t_rx;
    
    % Proyectar al tiempo actual t
    % Asunción del paper: el predecesor mantuvo u_last constante
    v_est_now = v_last + u_last * delta_t;
    p_est_now = p_last + v_last * delta_t + 0.5 * u_last * delta_t^2;
    
    % Proyectar un paso adelante t+1 (para el controlador)
    est.v_next = v_est_now + u_last * dt;
    est.p_next = p_est_now + v_est_now * dt + 0.5 * u_last * dt^2;
end

function sched = run_stage_1_scheduling(errors, Sim, Platoon, Comm)
    % Implementación del Algoritmo 1 
    sched = zeros(Platoon.N, Sim.T_scheduling);

    % Contador de cuántos usuarios hay en cada slot
    slot_load = zeros(1, Sim.T_scheduling);

    for i = 1:N
        e_i = abs(errors(i));

        % Si el error está bajo el umbral, no se agenda al MV i
        if e_i <= Platoon.e_th
            continue;
        end

        % f_i según Ec. (9): número de slots asignados en un período
        avg_err_corr_per_slot = 0.5 * Platoon.u_max * Sim.T_scheduling * (Sim.dt^2);
        f_real = e_i / avg_err_corr_per_slot;
        f_max  = Comm.O_max * Sim.T_scheduling / Platoon.N;

        f_i = ceil(min(f_real, f_max));   % redondeo hacia arriba para asegurar corrección
        if f_i <= 0
            continue;
        end

        % c_i según Ec. (21): separación promedio entre comunicaciones
        c_i = max(1, floor(Sim.T_scheduling / f_i));

        allocated = 0;
        tn = 1;   % primer slot deseado dentro del período

        while (allocated < f_i) && (tn <= Sim.T_scheduling)
            desired = min(tn, Sim.T_scheduling);

            if slot_load(desired) < Comm.O_max
                % Slot deseado disponible
                sched(i, desired) = 1;
                slot_load(desired) = slot_load(desired) + 1;
                allocated = allocated + 1;
                tn = desired + c_i;
            else
                % Búsqueda en "Z" alrededor del slot deseado
                found = false;
                for d = 1:(Sim.T_scheduling - 1)
                    left  = desired - d;
                    right = desired + d;

                    if left >= 1 && slot_load(left) < Comm.O_max
                        desired = left;
                        found = true;
                        break;
                    end
                    if right <= Sim.T_scheduling && slot_load(right) < Comm.O_max
                        desired = right;
                        found = true;
                        break;
                    end
                end

                if ~found
                    % No quedan slots libres en este período
                    break;
                end

                sched(i, desired) = 1;
                slot_load(desired) = slot_load(desired) + 1;
                allocated = allocated + 1;
                tn = desired + c_i;
            end
        end
    end

end

function powers = run_stage_2_power(users, h_gains, Comm)
    % Implementación Teorema 1 y Asignación Greedy 
    num_users = length(users);
    powers = zeros(num_users, 1);
    p_sorted = zeros(num_users, 1);
    
    % Ordenar usuarios por ganancia (para NOMA decoding order)
    % El paper indica ordenar por "interference channel gain"

    if num_users == 0
        return;
    end

    % Orden aproximado para decodificación NOMA (por ganancia de canal)
    [h_sorted, sort_idx] = sort(h_gains(:), 'ascend');   % orden creciente

    for k = 1:num_users
        if k == 1
            % Primer usuario: sólo ruido
            interference = 0;
        else
            % Interferencia aproximada de los usuarios ya asignados
            interference = sum(p_sorted(1:k-1) .* h_sorted(1:k-1));
        end

        % Potencia mínima para cumplir SINR >= R_th
        p_k = Comm.R_th * (Comm.N0 + interference) / h_sorted(k);

        % Respeta cota de potencia
        if p_k > Comm.P_max
            p_k = Comm.P_max;
        elseif p_k < 0
            p_k = 0;
        end

        p_sorted(k) = p_k;
    end

    % Reordenar a la correspondencia original de 'users'
    powers(sort_idx) = p_sorted;

end

function e = calculate_errors(pos_t, vel_t, Platoon)
    % Calcula error de posición para User Scheduling
    e = zeros(Platoon.N, 1);
    for i=1:Platoon.N
        % Error = (Pos Predecesor - Pos Propia) - (Distancia Deseada + Longitud)
        val = (pos_t(i) - pos_t(i+1)) - (Platoon.d_des + Platoon.len);
        e(i) = val;
    end
end