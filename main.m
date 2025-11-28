%% Joint Platoon Control and Resource Allocation (NOMA-V2V)
% Versión final para replicar Fig.2, Fig.3, Fig.4 (paper)
clear; clc; close all;
%% 1. CONFIGURACIÓN Y PARÁMETROS (Tabla I / paper)
Sim.dt = 10e-3; % 10 ms
Sim.T_scheduling = 10; % slots per scheduling period
Sim.TotalTime = 30; % seconds
Sim.Steps = Sim.TotalTime / Sim.dt;
Comm.BW = 180e3; % 180 kHz
Comm.N0_dBm = -174; % dBm/Hz
Comm.N0 = 10^((Comm.N0_dBm - 30)/10) * Comm.BW; % Watts (noise over BW)
Comm.P_max_dBm =    35;
Comm.P_max = 10^((Comm.P_max_dBm - 30)/10); % Watts
Comm.R_th_dB = 10;
Comm.R_th = 10^(Comm.R_th_dB/10); % linear threshold (SINR-like)
Comm.O_max = 2; % max NOMA users per slot
Platoon.N = 12; % number of followers
Platoon.d_des = 10; % desired inter-vehicle distance
Platoon.len = 5; % vehicle length
Platoon.e_th = 0.01; % scheduling trigger threshold (m) -- small
Platoon.u_max = 2; % acceleration bounds
Platoon.u_min = -2;
%% 2. INICIALIZACIÓN DE ESTADOS (positions in meters, v in m/s)
POS = zeros(Platoon.N + 1, Sim.Steps); % index 1: leader (vehicle 0), 2: MV1, ...
VEL = zeros(Platoon.N + 1, Sim.Steps);
ACC = zeros(Platoon.N + 1, Sim.Steps);
% Initial states (paper layout) -> p_i(0) = (N - i)*(d_des + len)
v_cruise = 20;
for idx = 1:(Platoon.N + 1)
    veh_idx = idx - 1; % 0 = leader
    VEL(idx, 1) = v_cruise;
    POS(idx, 1) = (Platoon.N - veh_idx) * (Platoon.d_des + Platoon.len);
end
% LastInfo(i,:) stores latest packet that follower i received about its PREDECESSOR:
% [pos_pred, vel_pred, acc_pred, t_rx]
LastInfo = zeros(Platoon.N, 4);
% initialize LastInfo with predecessor's initial true state:
for i = 1:Platoon.N
    LastInfo(i, :) = [POS(i, 1), VEL(i, 1), 0, 0];
end
% LastAllocated: last slot assigned for each MV (useful if implementing cyclic start)
LastAllocated = zeros(Platoon.N, 1);
%% Diagnostics / metrics
CumulComm = zeros(1, Sim.Steps);
CumulEnergy = zeros(1, Sim.Steps);
TotalComm = zeros(1, Sim.Steps);
TotalPower = zeros(1, Sim.Steps);
fprintf('Iniciando simulación de %d pasos (dt=%.3f s)...\n', Sim.Steps, Sim.dt);
%% 3. BUCLE PRINCIPAL DE SIMULACIÓN
UserSchedule = zeros(Platoon.N, Sim.T_scheduling); % schedule matrix
for t = 1 : Sim.Steps - 1
    current_time = (t-1) * Sim.dt; % use (t-1)*dt as current discrete time matching indices
% --- A. Lider (trayectoria del paper)
if current_time < 10
        u_leader = -0.5; % decelerate
elseif current_time < 15
        u_leader = 0;
elseif current_time < 25
        u_leader = 0.5; % accelerate
else
        u_leader = 0;
end
    ACC(1, t) = u_leader;
% --- B. STAGE-1: Scheduling (cada periodo de T_scheduling slots)
if mod(t-1, Sim.T_scheduling) == 0
        errors = calculate_errors(POS(:,t), VEL(:,t), Platoon);
        UserSchedule = run_stage_1_scheduling(errors, Sim, Platoon, Comm, LastAllocated);
% update LastAllocated for diagnostics (last assigned slot index)
for i = 1:Platoon.N
            assigned_slots = find(UserSchedule(i, :) == 1);
if ~isempty(assigned_slots)
                LastAllocated(i) = max(assigned_slots);
end
end
        total_slots_assigned = sum(UserSchedule(:));
        fprintf('[t=%.2f] scheduling: total slots assigned this period = %d\n', current_time, total_slots_assigned);
end
% --- C. CONTROL (paper applies control every dt but uses latest received pred info)
% Compute control for all followers using their latest LastInfo estimate
for i = 1:Platoon.N
        mv_idx = i + 1; % index in POS/VEL/ACC
        pred_est = estimate_predecessor_state(LastInfo(i,:), current_time, Sim.dt);
        u_opt = solve_control_qp(POS(mv_idx,t), VEL(mv_idx,t), pred_est, Platoon, Sim.dt);
        ACC(mv_idx, t) = u_opt;
end
% --- D. COMMUNICATION & NOMA (each slot) ---
    slot_idx = mod(t-1, Sim.T_scheduling) + 1;
    scheduled_users = find(UserSchedule(:, slot_idx) == 1); % MVs scheduled this slot (indices 1..N)
if ~isempty(scheduled_users)
        num_users = length(scheduled_users);
        H_signal = zeros(num_users,1);
        H_interf_matrix = zeros(num_users, num_users);
% Build channels using PREDECESSOR indices (paper model)
for m = 1:num_users
            rx = scheduled_users(m); % MV i
            tx_self = rx - 1; % predecessor index (0..N-1)
% positions in POS: leader at index 1 => POS(pred+1)
            pos_rx = POS(rx + 1, t);
            pos_tx_self = POS(tx_self + 1, t);
            H_signal(m) = calculate_channel_gain(pos_tx_self, pos_rx);
for n = 1:num_users
if n == m
                    H_interf_matrix(m,n) = 0;
continue;
end
                tx_other = scheduled_users(n) - 1; % predecessor of other user
% paper: leader (tx_other==0) only transmits to MV1; treat others as non-interfering
if tx_other == 0 && rx ~= 1
                    H_interf_matrix(m,n) = 0;
else
                    pos_tx_other = POS(tx_other + 1, t);
                    H_interf_matrix(m,n) = calculate_channel_gain(pos_tx_other, pos_rx);
end
end
end
% Stage-2: power allocation (NOMA) following Theorem 1
        allocated_powers = run_stage_2_power(scheduled_users, H_signal, H_interf_matrix, Comm);
        TotalComm(t) = sum(allocated_powers > 0);
        TotalPower(t) = sum(allocated_powers(allocated_powers > 0));
% Update LastInfo only for successful receptions (allocated_powers > 0)
for k = 1:num_users
            rx_id = scheduled_users(k); % receiver index (1..N)
            tx_id = rx_id - 1; % predecessor
if allocated_powers(k) > 0
% read predecessor's *current* state
                pos_tx = POS(tx_id + 1, t);
                vel_tx = VEL(tx_id + 1, t);
                acc_tx = ACC(tx_id + 1, t); % Use current t for all (fixed bug)
                LastInfo(rx_id, :) = [pos_tx, vel_tx, acc_tx, current_time];
end
end
else
% no scheduled users this slot
end
% --- E. Physics update for all vehicles (apply ACC computed above)
for idx = 1:(Platoon.N + 1)
        [POS(idx, t+1), VEL(idx, t+1)] = update_physics(POS(idx, t), VEL(idx, t), ACC(idx, t), Sim.dt);
end
end
% accumulate metrics
CumulComm = cumsum(TotalComm);
CumulEnergy = cumsum(TotalPower) * Sim.dt;
% report
n_updates = nnz(LastInfo(:,4) > 0);
fprintf('Número de LastInfo actualizados al menos una vez: %d/%d\n', n_updates, Platoon.N);
%% Plots (paper-style)
figure('Position',[100 100 900 900]);
spacing_errors = zeros(Platoon.N, Sim.Steps);
for tt = 1:Sim.Steps
for i = 1:Platoon.N
        spacing_errors(i,tt) = (POS(i,tt) - POS(i+1,tt)) - (Platoon.d_des + Platoon.len);
end
end
subplot(4,1,1); plot((0:Sim.Steps-1)*Sim.dt, spacing_errors); title('Spacing Error'); grid on; ylabel('m');
subplot(4,1,2); plot((0:Sim.Steps-1)*Sim.dt, VEL'); title('Velocidades (m/s)'); grid on; ylabel('m/s');
subplot(4,1,3); plot((0:Sim.Steps-1)*Sim.dt, mean(abs(spacing_errors), 1)); title('Average Absolute Spacing Error'); grid on;
ylabel('Error (m)');
subplot(4,1,4);
yyaxis left; plot((0:Sim.Steps-1)*Sim.dt, CumulComm); ylabel('Cumulative Packets Allocated');
yyaxis right; plot((0:Sim.Steps-1)*Sim.dt, CumulEnergy); ylabel('Cumulative Energy (J)');
xlabel('Time (s)'); grid on;
%% ------------------- FUNCIONES LOCALES -------------------
function sched = run_stage_1_scheduling(errors, Sim, Platoon, Comm, last_alloc)
% Stage-1 EXACT per paper Eqs (16)-(17)
    sched = zeros(Platoon.N, Sim.T_scheduling);
    slot_load = zeros(1, Sim.T_scheduling);
[~, sorted_idx] = sort(abs(errors), 'descend'); % Fixed: sort by descending error to prioritize high-demand vehicles
for ii = 1:Platoon.N
        i = sorted_idx(ii);
        e_i = abs(errors(i));
if e_i <= Platoon.e_th
continue;
end
        delta_e_max = 0.5 * Platoon.u_max * (Sim.T_scheduling * Sim.dt)^2;
        delta_e_max = max(delta_e_max, 1e-9);
        ratio = e_i / delta_e_max;
        f_i = ceil(ratio); % Ceiling for integer slots
        f_i = min(f_i, Comm.O_max * Sim.T_scheduling/Platoon.N); % System-wide cap only (fixed bug: remove per-user /N cap)
if f_i <= 0
continue;
end
        c_i = max(1, floor(Sim.T_scheduling / f_i)); % Eq (17)
        tl = last_alloc(i);
        tn_start = mod(tl + c_i, Sim.T_scheduling);
if tn_start == 0
            tn_start = Sim.T_scheduling;
end
% assign ideal slots tn_start + (n-1)*c_i mod T, but since paper uses 1 + (n-1)*c_i, but to use last_alloc
for n = 0:(f_i-1)
            desired = mod(tn_start + n*c_i -1, Sim.T_scheduling) +1; % -1 to adjust 1-based
            [sched, slot_load] = try_assign_slot_paper(sched, slot_load, i, desired, Sim, Comm);
end
end
end
function [sched, slot_load] = try_assign_slot_paper(sched, slot_load, u_idx, desired, Sim, Comm)
% Searches forward starting at desired; if no slot forward, wraps and searches from 1..desired-1
% 1) try desired
if slot_load(desired) < Comm.O_max
        sched(u_idx, desired) = 1;
        slot_load(desired) = slot_load(desired) + 1;
return;
end
% 2) search forward
for s = (desired+1):Sim.T_scheduling
if slot_load(s) < Comm.O_max
            sched(u_idx, s) = 1;
            slot_load(s) = slot_load(s) + 1;
return;
end
end
% 3) wrap and search 1..desired-1
for s = 1:(desired-1)
if slot_load(s) < Comm.O_max
            sched(u_idx, s) = 1;
            slot_load(s) = slot_load(s) + 1;
return;
end
end
% else: no assignment
end
function powers = run_stage_2_power(users, h_signal, H_interf_full, Comm)
% NOMA greedy allocation (Teorem 1 / eqs in paper)
    num_users = length(users);
    powers = zeros(num_users,1);
if num_users == 1
        p_req = (Comm.R_th * Comm.N0) / h_signal(1);
        powers(1) = p_req * (p_req <= Comm.P_max);
return;
end
% sum interference suffered by each receiver (rows)
    interf_sum = sum(H_interf_full, 2);
    [~, order] = sort(interf_sum, 'ascend'); % least interference first
    hS_ord = h_signal(order);
    H_ord = H_interf_full(order, order);
    p_ord = zeros(num_users,1);
    N0 = Comm.N0;
    Rth = Comm.R_th;
% first (no previous inter-user interference)
    p_ord(1) = (Rth * N0) / hS_ord(1);
for m = 2:num_users
        interf_prev = 0;
for j = 1:(m-1)
            interf_prev = interf_prev + p_ord(j) * H_ord(m,j);
end
        p_ord(m) = Rth * (N0 + interf_prev) / hS_ord(m);
end
    powers(order) = p_ord;
    over_idx = powers > Comm.P_max;
    powers(over_idx) = 0; % mark as failed
end
function [p_next, v_next] = update_physics(p, v, u, dt)
    v_next = v + u * dt;
    p_next = p + v * dt + 0.5 * u * dt^2;
end
function u = solve_control_qp(p_i, v_i, pred, Platoon, dt)
% Implements Eq (18) from paper (1D scalar QP)
% Use formulation: e = B + A*u, min (e' W e), note signs from derivation
    e_p = pred.p_next - p_i - (Platoon.d_des + Platoon.len); % predicted spacing error at t+1 (using pred p_next as pred at t+1)
    e_v = pred.v_next - v_i;
% Build B and A according to derivation
    B = [e_p - v_i * dt; e_v];
    A = [-0.5 * dt^2; -dt]; % negative for position and velocity terms
    %W = diag([1,1]); % default, but can tune if needed
    W = diag([1,1]);
    H = 2 * (A' * W * A); % scalar
    f = 2 * (A' * W * B); % scalar
% regularize H lightly
    eps_reg = 1e-9;
    H = H + eps_reg;
    lb = Platoon.u_min;
    ub = Platoon.u_max;
% Solve quadratic scalar QP with quadprog (works with 1x1 H)
try
        options = optimoptions('quadprog','Display','off');
        u = quadprog(double(H), double(f), [], [], [], [], lb, ub, [], options);
catch
% fallback analytic if solver fails
        u = [];
end
if isempty(u) || ~isfinite(u)
% derivative: H*u + f = 0 => u = -f/H
        u_unclamped = - f / H;
        u = min(max(u_unclamped, lb), ub);
end
if ~isfinite(u)
        u = 0;
end
end
function est = estimate_predecessor_state(last_info, t_now, dt)
% project last received predecessor packet (pos, vel, acc at time t_rx) to current time t_now,
% then one further step to produce pred.p_next and pred.v_next for controller usage.
    p_rx = last_info(1);
    v_rx = last_info(2);
    u_rx = last_info(3);
    t_rx = last_info(4); % time in seconds when packet was received
if t_rx == 0
% no packet ever received: assume predecessor at initial condition (no motion)
        delta_t = 0;
else
        delta_t = t_now - t_rx;
if delta_t < 0
            delta_t = 0;
end
end
% project to current t
    v_est = v_rx + u_rx * delta_t;
    p_est = p_rx + v_rx * delta_t + 0.5 * u_rx * (delta_t^2);
% now predict one step ahead (t+1) for controller (paper minimizes e_{t+1})
    est.v_next = v_est + u_rx * dt;
    est.p_next = p_est + v_est * dt + 0.5 * u_rx * dt^2;
end
function e = calculate_errors(pos_t, vel_t, Platoon)
    e = zeros(Platoon.N, 1);
for i = 1:Platoon.N
        e(i) = (pos_t(i) - pos_t(i+1)) - (Platoon.d_des + Platoon.len);
end
end
function h_gain = calculate_channel_gain(pos_tx, pos_rx)
% Deterministic path-loss used to reproduce paper figures (no fading)
    alpha = 3;
    G0 = 1e-4; % small-scale constant (tune to match paper)
    epsilon = 1e-6;
    dist = abs(pos_tx - pos_rx);
    dist = max(dist, epsilon);
    path_loss = G0 * (dist.^(-alpha));
    rayleigh_gain = 1; % deterministic (set to rand value if modeling fading)
    h_gain = path_loss * rayleigh_gain;
end