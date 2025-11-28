%% Joint Platoon Control and Resource Allocation (NOMA-V2V)
% Replicación del Paper - Código Corregido
clear; clc; close all;
%% 1. CONFIGURACIÓN Y PARÁMETROS (Tabla I)
Sim.dt = 10e-3; % 10 ms
Sim.T_scheduling = 10; % Periodo de agendamiento (slots)
Sim.TotalTime = 30; % 30 segundos
Sim.Steps = Sim.TotalTime / Sim.dt;
Comm.BW = 180e3; % 180 KHz
Comm.N0_dBm = -174; % dBm/Hz
Comm.N0 = 10^((Comm.N0_dBm - 30)/10) * Comm.BW; % Watts
Comm.P_max_dBm = 35;
Comm.P_max = 10^((Comm.P_max_dBm - 30)/10); % Watts
Comm.R_th_dB = 10;
Comm.R_th = 10^(Comm.R_th_dB/10); % Lineal
Comm.O_max = 2; % Máx usuarios NOMA por slot
Platoon.N = 12;
Platoon.d_des = 10;
Platoon.len = 5;
Platoon.e_th = 0.01;
Platoon.u_max = 2;
Platoon.u_min = -2;
%% 2. INICIALIZACIÓN DE ESTADOS
POS = zeros(Platoon.N + 1, Sim.Steps);
VEL = zeros(Platoon.N + 1, Sim.Steps);
ACC = zeros(Platoon.N + 1, Sim.Steps);
% Estado Inicial (t=1)
v_cruise = 20;
for i = 1:(Platoon.N + 1)
    veh_idx = i - 1; % 0 es líder
    VEL(i, 1) = v_cruise;
    POS(i, 1) = (Platoon.N - veh_idx) * (Platoon.d_des + Platoon.len);
end
LastInfo = zeros(Platoon.N, 4);
for i = 1:Platoon.N
    leader_idx = i;
    LastInfo(i, :) = [POS(leader_idx,1), VEL(leader_idx,1), 0, 0];
end
LastAllocated = zeros(Platoon.N, 1); % tl por MV, inicial 0
%% 3. BUCLE PRINCIPAL DE SIMULACIÓN
fprintf('Iniciando simulación de %d pasos...\n', Sim.Steps);
% Inicializamos UserSchedule fuera del loop para evitar errores en t=1
UserSchedule = zeros(Platoon.N, Sim.T_scheduling);
CumulComm = zeros(1, Sim.Steps);
CumulEnergy = zeros(1, Sim.Steps);
TotalComm = zeros(1, Sim.Steps);
TotalPower = zeros(1, Sim.Steps);
for t = 1 : Sim.Steps - 1
    current_time = t * Sim.dt;
% --- A. DINÁMICA DEL LÍDER ---
if current_time < 10
        u_leader = -0.5; % Corregido a -0.5 m/s² para coincidir con Δv=-5 en 10s
elseif current_time < 15
        u_leader = 0;
elseif current_time < 25
        u_leader = 0.5; % Corregido a 0.5 m/s²
else
        u_leader = 0;
end
    ACC(1, t) = u_leader;
% --- B. ASIGNACIÓN DE RECURSOS (Cada T slots) ---
if mod(t-1, Sim.T_scheduling) == 0
        errors = calculate_errors(POS(:,t), VEL(:,t), Platoon);
        UserSchedule = run_stage_1_scheduling(errors, Sim, Platoon, Comm, LastAllocated);
        % Actualizar LastAllocated después de scheduling
        for i = 1:Platoon.N
            assigned_slots = find(UserSchedule(i, :) == 1);
            if ~isempty(assigned_slots)
                LastAllocated(i) = max(assigned_slots);
            end
        end
        total_slots_assigned = sum(UserSchedule(:));
end
% --- C. COMUNICACIÓN Y NOMA (Cada slot) ---
    slot_idx = mod(t-1, Sim.T_scheduling) + 1;
    scheduled_users = find(UserSchedule(:, slot_idx) == 1);
if ~isempty(scheduled_users)
        num_users = length(scheduled_users);
% ====== Construcción de h_signal y matriz completa de interferencias ======
        H_signal = zeros(num_users,1);
        H_interf_matrix = zeros(num_users,num_users);
for m = 1:num_users
            rx = scheduled_users(m); % MV i
            tx_self = rx; % Predecesor es rx (índice MV 1..N, pred 1 es leader 0, pero índice POS tx_self)
            pos_rx = POS(rx + 1, t); % MV rx es POS(rx+1)
            pos_tx_self = POS(rx, t); % Predecesor POS(rx)
            % Canal útil h_{i-1,i}
            H_signal(m) = calculate_channel_gain(pos_tx_self, pos_rx);
for n = 1:num_users
if n == m
                    H_interf_matrix(m,n) = 0;
else
                    tx_other = scheduled_users(n);
                    pos_tx_other = POS(tx_other, t);
                    H_interf_matrix(m,n) = calculate_channel_gain(pos_tx_other, pos_rx);
end
end
end
% ====== Llamada NUEVA a run_stage_2_power (consistente con el paper) ======
        allocated_powers = run_stage_2_power(scheduled_users, H_signal, H_interf_matrix, Comm);
        TotalComm(t) = num_users;
        TotalPower(t) = sum(allocated_powers(allocated_powers > 0));
% Actualizar información si la comunicación es exitosa
for k = 1:length(scheduled_users)
            rx_id = scheduled_users(k); % receptor (MV i)
            tx_id = rx_id; % transmisor = predecesor POS(tx_id)
            if allocated_powers(k) > 0
% Obtener estados del transmisor real (predecesor)
if tx_id == 1
                    pos_tx = POS(1, t);
                    vel_tx = VEL(1, t);
                    acc_tx = ACC(1, t);
else
                    pos_tx = POS(tx_id, t);
                    vel_tx = VEL(tx_id, t);
                    acc_tx = ACC(tx_id, t); % Ahora control antes? No, ver abajo
end
                LastInfo(rx_id, :) = [pos_tx, vel_tx, acc_tx, current_time];
end
end
end
% --- D. CONTROL DISTRIBUIDO Y FÍSICA ---
    [POS(1,t+1), VEL(1,t+1)] = update_physics(POS(1,t), VEL(1,t), ACC(1,t), Sim.dt);
    for i = 1:Platoon.N
        mv_idx = i + 1;
        est_state = estimate_predecessor_state(LastInfo(i,:), current_time, Sim.dt);
        u_opt = solve_control_qp(POS(mv_idx,t), VEL(mv_idx,t), est_state, Platoon, Sim.dt);
        ACC(mv_idx, t) = u_opt;
        [POS(mv_idx,t+1), VEL(mv_idx,t+1)] = update_physics(POS(mv_idx,t), VEL(mv_idx,t), u_opt, Sim.dt);
    end
end
CumulComm = cumsum(TotalComm);
CumulEnergy = cumsum(TotalPower) * Sim.dt;
% Verifica si LastInfo quedó siempre con t_rx == 0 (nunca actualizó)
n_updates = nnz(LastInfo(:,4) > 0);
fprintf('Número de LastInfo actualizados al menos una vez: %d/%d\n', n_updates, Platoon.N);
%% GRÁFICAS
figure;
subplot(4,1,1); plot(POS'); title('Posiciones'); grid on; ylabel('m');
subplot(4,1,2); plot(VEL'); title('Velocidades'); grid on; ylabel('m/s');
spacing_errors = zeros(Platoon.N, Sim.Steps);
for t=1:Sim.Steps
for i=1:Platoon.N
        spacing_errors(i,t) = (POS(i,t) - POS(i+1,t)) - (Platoon.d_des + Platoon.len);
end
end
subplot(4,1,3); plot(1:Sim.Steps, mean(abs(spacing_errors), 1));
title('Average Absolute Spacing Error'); xlabel('Time Slot'); ylabel('Error (m)'); grid on;
subplot(4,1,4); yyaxis left; plot(1:Sim.Steps, CumulComm); ylabel('Total Allocated Comm');
yyaxis right; plot(1:Sim.Steps, CumulEnergy); ylabel('Total Energy (J)'); xlabel('Time Slot'); grid on;
%% --- FUNCIONES LOCALES ---
function powers = run_stage_2_power(users, h_signal, H_interf_full, Comm)
    num_users = length(users);
    powers = zeros(num_users,1);
if num_users == 1
        p_req = (Comm.R_th * Comm.N0) / h_signal(1);
if p_req > Comm.P_max
            powers(1) = 0;
else
            powers(1) = p_req;
end
return;
end
    interf_sum = sum(H_interf_full, 2);
    [~, order] = sort(interf_sum, 'ascend');
    hS_ord = h_signal(order);
    H_ord = H_interf_full(order, order);
    p_ord = zeros(num_users,1);
    N0 = Comm.N0;
    Rth = Comm.R_th;
    p_ord(1) = (Rth * N0) / hS_ord(1);
for m = 2:num_users
        interf_from_prev = 0;
for j = 1:(m-1)
            interf_from_prev = interf_from_prev + p_ord(j) * H_ord(m,j);
end
        p_ord(m) = Rth * (N0 + interf_from_prev) / hS_ord(m);
end
    powers(order) = p_ord;
    over_idx = powers > Comm.P_max;
    powers(over_idx) = 0;
end
function sched = run_stage_1_scheduling(errors, Sim, Platoon, Comm, last_alloc)
    sched = zeros(Platoon.N, Sim.T_scheduling);
    slot_load = zeros(1, Sim.T_scheduling);
for i = 1:Platoon.N
        e_i = abs(errors(i));
if e_i <= Platoon.e_th
continue;
end
        delta_e_max = 0.5 * Platoon.u_max * (Sim.T_scheduling * Sim.dt)^2;
        delta_e_max = max(delta_e_max, 1e-9);
        avg_max = Comm.O_max * Sim.T_scheduling / Platoon.N;
        f_raw = e_i / delta_e_max;
        f_i = min(f_raw, avg_max);
        f_i = round(f_i); % Redondear a entero para número de slots
        if f_i == 0
            continue;
        end
        c_i = Sim.T_scheduling / f_i; % Real, como en paper
        c_i = max(1, round(c_i));
        tl = last_alloc(i);
        tn = mod(tl + c_i, Sim.T_scheduling);
        if tn == 0
            tn = Sim.T_scheduling;
        end
        a = 0;
        while a < f_i && tn <= Sim.T_scheduling
            [sched, slot_load] = try_assign_slot(sched, slot_load, i, tn, Sim, Comm);
            if sched(i, tn) == 1 % Asignado exitosamente?
                a = a + 1;
                tn = tn + c_i;
                if tn > Sim.T_scheduling
                    tn = tn - Sim.T_scheduling;
                end
            else
                % Buscar candidato próximo
                tn = tn + 1; % Simple avance, o implementar Z full
                if tn > Sim.T_scheduling
                    break;
                end
            end
        end
end
end
function [sched, slot_load] = try_assign_slot(sched, slot_load, u_idx, tn, Sim, Comm)
    desired = min(max(tn,1), Sim.T_scheduling);
if slot_load(desired) < Comm.O_max
        sched(u_idx, desired) = 1;
        slot_load(desired) = slot_load(desired) + 1;
return;
end
for d = 1:(Sim.T_scheduling-1)
        left = desired - d;
        right = desired + d;
if left >= 1 && slot_load(left) < Comm.O_max
                sched(u_idx, left) = 1;
                slot_load(left) = slot_load(left) + 1;
return;
end
if right <= Sim.T_scheduling && slot_load(right) < Comm.O_max
                sched(u_idx, right) = 1;
                slot_load(right) = slot_load(right) + 1;
return;
end
end
end
function [p_next, v_next] = update_physics(p, v, u, dt)
    v_next = v + u * dt;
    p_next = p + v * dt + 0.5 * u * dt^2;
end
function u = solve_control_qp(p_i, v_i, pred, Platoon, dt)
    term_p_const = pred.p_next - p_i - v_i*dt - (Platoon.d_des + Platoon.len);
    term_v_const = pred.v_next - v_i;
    B_vec = [term_p_const; term_v_const];
    A_vec = [-0.5 * dt^2; -dt];
    W = diag([1, 1]);
    H = 2 * (A_vec' * W * A_vec);
    f = 2 * (A_vec' * W * B_vec);
    eps_reg = 1e-9;
    H = H + eps_reg;
    H_mat = double(H);
    f_vec = double(f);
    lb = Platoon.u_min;
    ub = Platoon.u_max;
    u = [];
try
        options = optimoptions('quadprog','Display','off');
        u = quadprog(H_mat, f_vec, [], [], [], [], lb, ub, [], options);
catch
        u = [];
end
if isempty(u) || ~isfinite(u)
        u_unclamped = - f / H;
        u = min(max(u_unclamped, lb), ub);
end
if ~isfinite(u)
        u = 0;
end
end
function est = estimate_predecessor_state(last_info, t_now, dt)
    p_t_prime = last_info(1);
    v_t_prime = last_info(2);
    u_t_prime = last_info(3);
    time_prime = last_info(4);
    delta_time = t_now - time_prime;
    v_est_t = v_t_prime + u_t_prime * delta_time;
    p_est_t = p_t_prime + v_t_prime * delta_time + 0.5 * u_t_prime * (delta_time^2);
    est.v_next = v_est_t + u_t_prime * dt;
    est.p_next = p_est_t + v_est_t * dt + 0.5 * u_t_prime * dt^2;
end
function e = calculate_errors(pos_t, vel_t, Platoon)
    e = zeros(Platoon.N, 1);
for i=1:Platoon.N
        val = (pos_t(i) - pos_t(i+1)) - (Platoon.d_des + Platoon.len);
        e(i) = val;
end
end
function h_gain = calculate_channel_gain(pos_tx, pos_rx)
    alpha = 3;
    G0 = 1;
    epsilon = 1e-6;
    dist = abs(pos_tx - pos_rx);
    dist = max(dist, epsilon);
    path_loss = G0 * (dist .^ (-alpha));
    h_small = (randn(1,1) + 1i * randn(1,1)) / sqrt(2);
    rayleigh_gain = abs(h_small)^2;
    h_gain = path_loss * rayleigh_gain;
end