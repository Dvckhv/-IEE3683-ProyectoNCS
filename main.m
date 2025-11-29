%% Joint Platoon Control and Resource Allocation (NOMA-V2V)
clear; clc; close all;
%% Configuration and parameters
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
Comm.R_th = 10^(Comm.R_th_dB/10);
Comm.O_max = 2; % max NOMA users per slot
Platoon.N = 12; % number of followers
Platoon.d_des = 10; % desired inter-vehicle distance
Platoon.len = 5; % vehicle length
Platoon.e_th = 0.05; % scheduling trigger threshold (m)
Platoon.u_max = 2; % acceleration bounds
Platoon.u_min = -2;
Platoon.m = 1500; % Vehicle weight (kg)
lims.Fmax = Platoon.u_max * Platoon.m; % Force bounds
lims.Fmin = Platoon.u_min * Platoon.m;
lims.max_jerk = 0.9; % Jerk bound
lims.dFmax = lims.max_jerk * Platoon.m; % delta Force bounds
lims.dFmin = -lims.max_jerk * Platoon.m;

%% STATE INITIALIZATION (positions in meters, v in m/s, a in m/s^2, F in N, DF in N/s)
POS = zeros(Platoon.N + 1, Sim.Steps); % index 1: leader (vehicle 0), 2: MV1, ...
VEL = zeros(Platoon.N + 1, Sim.Steps);
ACC = zeros(Platoon.N + 1, Sim.Steps);
FOR = zeros(Platoon.N + 1, Sim.Steps);
DFOR = zeros(Platoon.N + 1, Sim.Steps);
% Initial states -> p_i(0) = (N - i)*(d_des + len)
v_cruise = 20;
for idx = 1:(Platoon.N + 1)
    veh_idx = idx - 1; % 0 = leader
    VEL(idx, 1) = v_cruise;
    POS(idx, 1) = (Platoon.N - veh_idx) * (Platoon.d_des + Platoon.len);
end
% LastInfo(i,:) stores latest packet that follower i received about its PREDECESSOR:
% [pos_pred, vel_pred, acc_pred, for_pred, dfor_pred, t_rx]
LastInfo = zeros(Platoon.N, 6);
for i = 1:Platoon.N
    LastInfo(i, :) = [POS(i, 1), VEL(i, 1), 0, 0, 0, 0];
end
% LastAllocated: last slot assigned for each MV
LastAllocated = zeros(Platoon.N, 1);
%% Metrics
CumulComm = zeros(1, Sim.Steps);
CumulEnergy = zeros(1, Sim.Steps);
TotalComm = zeros(1, Sim.Steps);
TotalPower = zeros(1, Sim.Steps);
TotalSchedule = zeros(Platoon.N, Sim.Steps);
fprintf('Iniciando simulación de %d pasos (dt=%.3f s)...\n', Sim.Steps, Sim.dt);
%% Main simulation loop
UserSchedule = zeros(Platoon.N, Sim.T_scheduling);
for t = 1 : Sim.Steps - 1
    current_time = (t-1) * Sim.dt;
% Leader trajectory
if current_time < 1
    dF_leader = 0; 
elseif current_time < 6
    dF_leader = lims.dFmin / 10; % decelerate
else
    dF_leader = 500 * (v_cruise - VEL(1,t)) - 1400 * ACC(1,t); % Simple PD controller for leader
    if dF_leader < lims.dFmin
        dF_leader = lims.dFmin;
    elseif dF_leader > lims.dFmax
        dF_leader = lims.dFmax;
    end
end
    DFOR(1, t) = dF_leader;
% Scheduling
% Stage 1: Transmition moments for each vehicle
if mod(t-1, Sim.T_scheduling) == 0
        errors = calculate_errors(POS(:,t), Platoon);
        UserSchedule = run_stage_1_scheduling(errors, Sim, Platoon, Comm, LastAllocated, FOR(:,t), Platoon.m, lims);
        TotalSchedule(:,t:t+9) = UserSchedule;

for i = 1:Platoon.N
            assigned_slots = find(UserSchedule(i, :) == 1);
if ~isempty(assigned_slots)
                LastAllocated(i) = max(assigned_slots);
end
end
        total_slots_assigned = sum(UserSchedule(:));
        fprintf('[t=%.2f] scheduling: total slots assigned this period = %d\n', current_time, total_slots_assigned);
end
% CONTROL
% Compute control for all followers using their latest LastInfo estimate
for i = 1:Platoon.N
        mv_idx = i + 1; % index in POS/VEL/ACC/FOR/DFOR
        pred_est = estimate_predecessor_state(Platoon.m, LastInfo(i,:), current_time, Sim.dt, lims);
        u_opt = solve_control_qp(POS(mv_idx,t), VEL(mv_idx,t), FOR(mv_idx,t), pred_est, Platoon, Sim.dt, lims);
        DFOR(mv_idx, t) = u_opt;
end
% COMMUNICATION & NOMA
    slot_idx = mod(t-1, Sim.T_scheduling) + 1;
    scheduled_users = find(UserSchedule(:, slot_idx) == 1);
if ~isempty(scheduled_users)
        num_users = length(scheduled_users);
        H_signal = zeros(num_users,1);
        H_interf_matrix = zeros(num_users, num_users);
% Build channels using PREDECESSOR indices
for m = 1:num_users
            rx = scheduled_users(m);
            tx_self = rx - 1; % predecessor index
            pos_rx = POS(rx + 1, t);
            pos_tx_self = POS(tx_self + 1, t);
            H_signal(m) = calculate_channel_gain(pos_tx_self, pos_rx);
for n = 1:num_users
if n == m
                    H_interf_matrix(m,n) = 0;
continue;
end
                tx_other = scheduled_users(n) - 1;
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
            rx_id = scheduled_users(k);
            tx_id = rx_id - 1;
if allocated_powers(k) > 0
                pos_tx = POS(tx_id + 1, t);
                vel_tx = VEL(tx_id + 1, t);
                acc_tx = ACC(tx_id + 1, t);
                for_tx = FOR(tx_id + 1, t);
                dfor_tx = DFOR(tx_id + 1, t);
                LastInfo(rx_id, :) = [pos_tx, vel_tx, acc_tx, for_tx, dfor_tx,current_time];
end
end
else
% no scheduled users this slot
end
% Physics update for all vehicles (apply dF computed above)
for idx = 1:(Platoon.N + 1)
        [POS(idx, t+1), VEL(idx, t+1), FOR(idx, t+1), ACC(idx, t+1)] = update_physics_smooth(POS(idx, t), VEL(idx, t), FOR(idx, t), DFOR(idx, t), Platoon.m, Sim.dt, lims);
end
end
% accumulate metrics
CumulComm = cumsum(TotalComm);
CumulEnergy = cumsum(TotalPower) * Sim.dt;
% report
n_updates = nnz(LastInfo(:,6) > 0);
fprintf('Número de LastInfo actualizados al menos una vez: %d/%d\n', n_updates, Platoon.N);



%% Plots
figure('Position',[100 100 900 900]);
spacing_errors = zeros(Platoon.N, Sim.Steps);
for tt = 1:Sim.Steps
for i = 1:Platoon.N
        spacing_errors(i,tt) = (POS(i,tt) - POS(i+1,tt)) - (Platoon.d_des + Platoon.len);
end
end
subplot(4,1,1); plot((0:Sim.Steps-1)*Sim.dt, spacing_errors); title('Spacing Error'); grid on; ylabel('m'); legend('1','2','3','4','5','6','7','8','9','10','11','12');
subplot(4,1,2); plot((0:Sim.Steps-1)*Sim.dt, VEL'); title('Velocidades (m/s)'); grid on; ylabel('m/s'), legend('1','2','3','4','5','6','7','8','9','10','11','12');
subplot(4,1,3); plot((0:Sim.Steps-1)*Sim.dt, mean(abs(spacing_errors), 1)); title('Average Absolute Spacing Error'); grid on;
ylabel('Error (m)');
subplot(4,1,4);
yyaxis left; plot((0:Sim.Steps-1)*Sim.dt, CumulComm); ylabel('Cumulative Packets Allocated');
yyaxis right; plot((0:Sim.Steps-1)*Sim.dt, CumulEnergy); ylabel('Cumulative Energy (J)');
xlabel('Time (s)'); grid on;

    
[N, T] = size(TotalSchedule);
figure;
for i = 1:N
    subplot(N,1,i);
    stem((0:Sim.Steps-1)*Sim.dt, TotalSchedule(i,:), 'filled', 'MarkerSize', 3);
    ylim([-0.2, 1.2]);
    grid on;
    title(sprintf('Vehicle %d', i));
    ylabel('Tx');
    if i == N
        xlabel('Time (s)');
    else
        set(gca, 'XTickLabel', []);
    end
end

sgtitle('Vehicle transmition schedulle');

figure;
subplot(3,1,1); plot((0:Sim.Steps-1)*Sim.dt, POS(1,:)); title('Leader position'); grid on; ylabel('m');
subplot(3,1,2); plot((0:Sim.Steps-1)*Sim.dt, VEL(1,:)); title('Leader speed (m/s)'); grid on; ylabel('m/s');
subplot(3,1,3); plot((0:Sim.Steps-1)*Sim.dt, ACC(1,:)); title('Leader acceleration (m/s)'); grid on; ylabel('m/s^2');
xlabel('Time (s)'); grid on;

acc_leader = ACC(1,:);
save("Leader_Acceleration_Data","acc_leader")



%% ------------------- LOCAL FUNCTIONS -------------------
function sched = run_stage_1_scheduling(errors, Sim, Platoon, Comm, last_alloc, F, m, lims)
% Stage 1 
    sched = zeros(Platoon.N, Sim.T_scheduling);
    slot_load = zeros(1, Sim.T_scheduling);
[~, sorted_idx] = sort(abs(errors), 'descend'); % Sort by descending error to prioritize high-demand vehicles
for ii = 1:Platoon.N
        i = sorted_idx(ii);
        e_i = abs(errors(i));
if e_i <= Platoon.e_th
continue;
end
        F_max_actual = F(i) + lims.dFmax*(Sim.T_scheduling * Sim.dt);
        a_max_actual = F_max_actual/m;

        delta_e_max = 0.5 * a_max_actual * (Sim.T_scheduling * Sim.dt)^2;
        delta_e_max = max(delta_e_max, 1e-9);
        ratio = e_i / delta_e_max;
        f_i = ceil(ratio); % Ceiling for integer slots
        f_i = min(f_i, Comm.O_max * Sim.T_scheduling/Platoon.N);
if f_i <= 0
continue;
end
        c_i = max(1, floor(Sim.T_scheduling / f_i));
        tl = last_alloc(i);
        tn_start = mod(tl + c_i, Sim.T_scheduling);
if tn_start == 0
            tn_start = Sim.T_scheduling;
end
for n = 0:(f_i-1)
            desired = mod(tn_start + n*c_i -1, Sim.T_scheduling) +1;
            [sched, slot_load] = try_assign_slot_paper(sched, slot_load, i, desired, Sim, Comm);
end
end
end
function [sched, slot_load] = try_assign_slot_paper(sched, slot_load, u_idx, desired, Sim, Comm)
if slot_load(desired) < Comm.O_max
        sched(u_idx, desired) = 1;
        slot_load(desired) = slot_load(desired) + 1;
return;
end
for s = (desired+1):Sim.T_scheduling
if slot_load(s) < Comm.O_max
            sched(u_idx, s) = 1;
            slot_load(s) = slot_load(s) + 1;
return;
end
end
for s = 1:(desired-1)
if slot_load(s) < Comm.O_max
            sched(u_idx, s) = 1;
            slot_load(s) = slot_load(s) + 1;
return;
end
end
% No assignment
end
function powers = run_stage_2_power(users, h_signal, H_interf_full, Comm)
% NOMA greedy allocation
    num_users = length(users);
    powers = zeros(num_users,1);
if num_users == 1
        p_req = (Comm.R_th * Comm.N0) / h_signal(1);
        powers(1) = p_req * (p_req <= Comm.P_max);
return;
end
% sum interference suffered by each receiver
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

function [p_next, v_next, F_next, a_next] = update_physics_smooth(p, v, F, u_F, m, dt, lims)

    F_next = F + u_F * dt;

    if F_next < lims.Fmin
        F_next = lims.Fmin;
    elseif F_next > lims.Fmax
        F_next = lims.Fmax;
    end

    a_next = F_next / m;

    v_next = v + a_next * dt;
    p_next = p + v * dt + 0.5 * a_next * dt^2;
end

function u = solve_control_qp(p_i, v_i, F_i, pred, Platoon, dt, lims)

m   = Platoon.m;
D   = Platoon.d_des + Platoon.len;

e_p = pred.p_next - p_i - D;
e_v = pred.v_next - v_i;
e_a = pred.a_next - (F_i / m);  
% e(t+1) = B + A*u, with u = dF/dt
B_p = e_p - v_i*dt - 0.5*(F_i/m)*dt^2;
B_v = e_v - (F_i/m)*dt;
B_a = e_a;

A_p = -0.5*(dt^3/m);
A_v = - (dt^2/m);
A_a = - (dt/m);

B = [B_p; B_v; B_a];
A = [A_p; A_v; A_a];

% Cost weights
W = diag([800, 100, 1]);

% QP matrices: min (B + A*u)' W (B + A*u)
H = 2 * (A' * W * A);
f = 2 * (A' * W * B);

% Bounds for u = dF/dt
lb = lims.dFmin;
ub = lims.dFmax;

% regularize H lightly
    eps_reg = 1e-9;
    H = H + eps_reg;

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
function est = estimate_predecessor_state(m, last_info, t_now, dt, lims)
% project last received predecessor packet (pos, vel, acc at time t_rx) to current time t_now,
% then one further step to produce pred.p_next and pred.v_next for controller usage.
    p_rx = last_info(1);
    v_rx = last_info(2);
    a_rx = last_info(3);
    f_rx = last_info(4);
    df_rx = last_info(5);
    t_rx = last_info(6); % time in seconds when packet was received
    delta_t = t_now - t_rx;
    if delta_t < 0
        delta_t = 0;
    end
    % project to current t 
    t_steps = floor(delta_t/dt);
    if t_steps > 0
        p_est = p_rx; v_est = v_rx; F_est = f_rx; a_est = a_rx;
        for k = 1:t_steps
            [p_est, v_est, F_est, a_est] = update_physics_smooth(p_est, v_est, F_est, df_rx, m, dt, lims);
        end
    else
        p_est = p_rx; v_est = v_rx; F_est = f_rx; a_est = a_rx;
    end
% now predict one step ahead (t+1) for controller
    est.a_next = a_est;
    est.v_next = v_est + a_est * dt;
    est.p_next = p_est + v_est * dt + 0.5 * a_est * dt^2;
    
end
function e = calculate_errors(pos_t, Platoon)
    e = zeros(Platoon.N, 1);
for i = 1:Platoon.N
        e(i) = (pos_t(i) - pos_t(i+1)) - (Platoon.d_des + Platoon.len);
end
end
function h_gain = calculate_channel_gain(pos_tx, pos_rx)
% Deterministic path-loss used to reproduce paper figures (no fading)
    alpha = 3;
    G0 = 1e-4; % small-scale constant
    epsilon = 1e-6;
    dist = abs(pos_tx - pos_rx);
    dist = max(dist, epsilon);
    path_loss = G0 * (dist.^(-alpha));
    rayleigh_gain = 1; % deterministic 
    h_gain = path_loss * rayleigh_gain;
end