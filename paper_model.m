%% Joint Platoon Control and Resource Allocation (NOMA-V2V)
clear; clc; close all;

%% Configuration and parameters
% Define simulation parameters
Sim.dt           = 10e-3;                   % simulation step (s)
Sim.T_scheduling = 10;                      % number of slots per scheduling period
Sim.TotalTime    = 30;                      % total simulation time (s)
Sim.Steps        = Sim.TotalTime / Sim.dt;  % total number of time steps

% Define communication parameters
Comm.BW       = 180e3;                     % system bandwidth (Hz)
Comm.N0_dBm   = -174;                      % noise power spectral density (dBm/Hz)
Comm.N0       = 10^((Comm.N0_dBm - 30)/10) * Comm.BW; % noise power over BW (W)
Comm.P_max_dBm = 35;                       % maximum transmit power (dBm)
Comm.P_max    = 10^((Comm.P_max_dBm - 30)/10); % maximum transmit power (W)
Comm.R_th_dB  = 10;                        % SINR threshold (dB)
Comm.R_th     = 10^(Comm.R_th_dB/10);      % SINR threshold (linear scale)
Comm.O_max    = 2;                         % maximum NOMA users per time slot

% Define platoon parameters
Platoon.N     = 12;                        % number of follower vehicles
Platoon.d_des = 10;                        % desired inter-vehicle distance (m)
Platoon.len   = 5;                         % vehicle length (m)
Platoon.e_th  = 0.05;                      % scheduling trigger threshold (m)
Platoon.u_max = 2;                         % maximum acceleration (m/s^2)
Platoon.u_min = -2;                        % minimum acceleration (m/s^2)

%% State initialization
% State matrices: positions (m), velocities (m/s), accelerations (m/s^2)
POS = zeros(Platoon.N + 1, Sim.Steps);     % row 1 = leader (vehicle 0), rows 2..N+1 = followers
VEL = zeros(Platoon.N + 1, Sim.Steps);
ACC = zeros(Platoon.N + 1, Sim.Steps);

% Initialize vehicles in equilibrium platoon configuration
v_cruise = 20;                             % desired cruise speed (m/s)
for idx = 1:(Platoon.N + 1)
    veh_idx     = idx - 1;                                % convert row index to vehicle ID (0 = leader)
    VEL(idx, 1) = v_cruise;
    POS(idx, 1) = (Platoon.N - veh_idx) * (Platoon.d_des + Platoon.len); % spacing from tail to head
end

% Initialize LastInfo: latest packet received by each follower about its predecessor
% LastInfo(i,:) = [pos_pred, vel_pred, acc_pred, t_rx]
LastInfo = zeros(Platoon.N, 4);
for i = 1:Platoon.N
    LastInfo(i, :) = [POS(i, 1), VEL(i, 1), 0, 0];  % initial guess for predecessor state
end

% Initialize LastAllocated: last slot index assigned to each follower
LastAllocated = zeros(Platoon.N, 1);

% Load reference leader acceleration profile (from previous simulation)
data       = load("Leader_Acceleration_Data.mat", "acc_leader");
acc_leader = data.acc_leader;              % leader acceleration profile (m/s^2)

%% Metrics initialization
% Communication and energy metrics over time
CumulComm    = zeros(1, Sim.Steps);        % cumulative number of successful transmissions
CumulEnergy  = zeros(1, Sim.Steps);        % cumulative energy consumption (J)
TotalComm    = zeros(1, Sim.Steps);        % successful transmissions per time step
TotalPower   = zeros(1, Sim.Steps);        % total transmit power per time step (W)
TotalSchedule = zeros(Platoon.N, Sim.Steps); % per-vehicle scheduling over time

fprintf('Starting simulation with %d steps (dt = %.3f s)...\n', Sim.Steps, Sim.dt);

%% Main simulation loop
% UserSchedule(k,s) = 1 if follower k transmits in slot s of the current scheduling period
UserSchedule = zeros(Platoon.N, Sim.T_scheduling); % schedule matrix within a period

for t = 1:(Sim.Steps - 1)
    current_time = (t - 1) * Sim.dt;  % current simulation time (s), aligned with indices
    
    %% Leader trajectory
    % Use leader acceleration from precomputed profile
    u_leader     = acc_leader(t);  % leader acceleration (m/s^2)
    ACC(1, t)    = u_leader;       % row 1 corresponds to leader
    
    %% Stage 1 scheduling (every T_scheduling slots)
    if mod(t - 1, Sim.T_scheduling) == 0
        % Compute spacing errors at current time
        errors = calculate_errors(POS(:, t), Platoon);
        
        % Run stage-1 scheduling
        UserSchedule = run_stage_1_scheduling(errors, Sim, Platoon, Comm, LastAllocated);
        
        % Expand per-period schedule over global time axis
        TotalSchedule(:, t:(t + Sim.T_scheduling - 1)) = UserSchedule; % copy schedule window
        
        % Update LastAllocated for diagnostics (last assigned slot index)
        for i = 1:Platoon.N
            assigned_slots = find(UserSchedule(i, :) == 1);
            if ~isempty(assigned_slots)
                LastAllocated(i) = max(assigned_slots); % store last allocated slot index
            end
        end
        
        % Report number of assigned slots in this scheduling period
        total_slots_assigned = sum(UserSchedule(:));
        fprintf('[t = %.2f s] scheduling: total slots assigned this period = %d\n', ...
                current_time, total_slots_assigned);
    end
    
    %% Control update for followers
    % Apply local controller for each follower using its latest predecessor estimate
    for i = 1:Platoon.N
        mv_idx   = i + 1;  % follower i is stored at row i+1 (row 1 = leader)
        
        % Estimate predecessor state for follower i
        pred_est = estimate_predecessor_state(LastInfo(i, :), current_time, Sim.dt);
        
        % Solve local QP-based control for follower i
        u_opt    = solve_control_qp(POS(mv_idx, t), VEL(mv_idx, t), pred_est, Platoon, Sim.dt);
        
        % Store acceleration command (m/s^2)
        ACC(mv_idx, t) = u_opt;
    end
    
    %% Communication and NOMA (per slot within scheduling period)
    % Determine current slot index within the scheduling period
    slot_idx        = mod(t - 1, Sim.T_scheduling) + 1; % slot index in [1, T_scheduling]
    
    % Get users scheduled in this slot
    scheduled_users = find(UserSchedule(:, slot_idx) == 1); % follower indices (1..N)
    
    if ~isempty(scheduled_users)
        num_users       = length(scheduled_users);
        H_signal        = zeros(num_users, 1);          % desired channel gains
        H_interf_matrix = zeros(num_users, num_users);  % interference channel gains
        
        % Build channels using predecessor-based links
        for m = 1:num_users
            rx      = scheduled_users(m);  % receiver follower index (1..N)
            tx_self = rx - 1;             % predecessor ID (0..N-1, 0 = leader)
            
            % Positions in POS: leader at row 1, follower k at row k+1
            pos_rx      = POS(rx + 1, t);      % follower rx stored at row rx+1
            pos_tx_self = POS(tx_self + 1, t); % predecessor stored at row (tx_self+1)
            
            % Desired link: predecessor -> this follower
            H_signal(m) = calculate_channel_gain(pos_tx_self, pos_rx);
            
            % Interfering links from other scheduled users
            for n = 1:num_users
                if n == m
                    H_interf_matrix(m, n) = 0;  % no self-interference
                    continue;
                end
                
                tx_other = scheduled_users(n) - 1; % other predecessor ID (0 = leader)
              
                % treat leader as non-interfering for other receivers
                if tx_other == 0 && rx ~= 1
                    H_interf_matrix(m, n) = 0;
                else
                    pos_tx_other = POS(tx_other + 1, t); % map transmitter ID to row index
                    H_interf_matrix(m, n) = calculate_channel_gain(pos_tx_other, pos_rx);
                end
            end
        end
        
        % Stage 2: NOMA power allocation
        allocated_powers = run_stage_2_power(scheduled_users, H_signal, H_interf_matrix, Comm);
        
        % Store communication metrics
        TotalComm(t)  = sum(allocated_powers > 0);                   % successful receptions
        TotalPower(t) = sum(allocated_powers(allocated_powers > 0)); % total TX power used (W)
        
        % Update LastInfo only for successful receptions
        for k = 1:num_users
            rx_id = scheduled_users(k); % receiver follower index (1..N)
            tx_id = rx_id - 1;          % predecessor ID (0..N-1)
            
            if allocated_powers(k) > 0
                % Read predecessor's current state at time t
                pos_tx = POS(tx_id + 1, t); % map predecessor ID to state row
                vel_tx = VEL(tx_id + 1, t);
                acc_tx = ACC(tx_id + 1, t);
                
                % Store latest received predecessor state and timestamp
                LastInfo(rx_id, :) = [pos_tx, vel_tx, acc_tx, current_time];
            end
        end
    else
        % No scheduled users in this slot (no communication update)
    end
    
    %% Physics update
    % Update positions and velocities for all vehicles using simple kinematic model
    for idx = 1:(Platoon.N + 1)
        [POS(idx, t + 1), VEL(idx, t + 1)] = update_physics(POS(idx, t), ...
                                                            VEL(idx, t), ...
                                                            ACC(idx, t), ...
                                                            Sim.dt);
    end
end

%% Post-processing of metrics
% Accumulate communication and energy metrics over time
CumulComm   = cumsum(TotalComm);               % cumulative successful transmissions
CumulEnergy = cumsum(TotalPower) * Sim.dt;     % cumulative energy (J)

% Count how many followers updated LastInfo at least once
n_updates = nnz(LastInfo(:, 4) > 0);
fprintf('Number of followers with at least one LastInfo update: %d/%d\n', ...
        n_updates, Platoon.N);

%% Plots
figure('Position', [100 100 900 900]);

% Compute spacing errors for all vehicles and times
spacing_errors = zeros(Platoon.N, Sim.Steps);
for tt = 1:Sim.Steps
    for i = 1:Platoon.N
        spacing_errors(i, tt) = (POS(i, tt) - POS(i + 1, tt)) ...
                                - (Platoon.d_des + Platoon.len);
    end
end

subplot(4, 1, 1);
plot((0:Sim.Steps - 1) * Sim.dt, spacing_errors);
title('Spacing Error');
grid on;
ylabel('m');
legend('1','2','3','4','5','6','7','8','9','10','11','12');

subplot(4, 1, 2);
plot((0:Sim.Steps - 1) * Sim.dt, VEL');
title('Vehicle Speeds');
grid on;
ylabel('m/s');
legend('1','2','3','4','5','6','7','8','9','10','11','12');

subplot(4, 1, 3);
plot((0:Sim.Steps - 1) * Sim.dt, mean(abs(spacing_errors), 1));
title('Average Absolute Spacing Error');
grid on;
ylabel('Error (m)');

subplot(4, 1, 4);
yyaxis left;
plot((0:Sim.Steps - 1) * Sim.dt, CumulComm);
ylabel('Cumulative Packets Allocated');
yyaxis right;
plot((0:Sim.Steps - 1) * Sim.dt, CumulEnergy);
ylabel('Cumulative Energy (J)');
xlabel('Time (s)');
grid on;

fprintf("Total power used: %.2e\n", CumulEnergy(end))
fprintf("Total packets: %.1f\n", CumulComm(end))

%% Indepent plots 

figure;
plot((0:Sim.Steps - 1) * Sim.dt, spacing_errors);
title('Spacing Error');
grid on;
ylabel('m');
legend('1','2','3','4','5','6','7','8','9','10','11','12');

figure;
plot((0:Sim.Steps - 1) * Sim.dt, VEL');
title('Vehicle Speeds');
grid on;
ylabel('m/s');
legend('1','2','3','4','5','6','7','8','9','10','11','12');

figure;
plot((0:Sim.Steps - 1) * Sim.dt, mean(abs(spacing_errors), 1));
title('Average Absolute Spacing Error');
grid on;
ylabel('Error (m)');

figure;
yyaxis left;
plot((0:Sim.Steps - 1) * Sim.dt, CumulComm);
ylabel('Cumulative Packets Allocated');
yyaxis right;
plot((0:Sim.Steps - 1) * Sim.dt, CumulEnergy);
ylabel('Cumulative Energy (J)');
xlabel('Time (s)');
grid on;

%% Plot per-vehicle transmission schedule
[N, ~] = size(TotalSchedule);

figure;
for i = 1:N
    subplot(N, 1, i);
    stem((0:Sim.Steps - 1) * Sim.dt, TotalSchedule(i, :), ...
         'filled', 'MarkerSize', 3);
    ylim([-0.2, 1.2]);
    grid on;
    title(sprintf('Vehicle %d', i));
    ylabel('Tx');
    if i == N
        xlabel('Time (s)');
    else
        set(gca, 'XTickLabel', []);  % hide x labels for intermediate subplots
    end
end
sgtitle('Vehicle Transmission Schedule');



%% Plot leader trajectory (position, speed, acceleration)
figure;
subplot(3, 1, 1);
plot((0:Sim.Steps - 1) * Sim.dt, POS(1, :));
title('Leader position');
grid on;
ylabel('m');

subplot(3, 1, 2);
plot((0:Sim.Steps - 1) * Sim.dt, VEL(1, :));
title('Leader speed (m/s)');
grid on;
ylabel('m/s');

subplot(3, 1, 3);
plot((0:Sim.Steps - 1) * Sim.dt, ACC(1, :));
title('Leader acceleration (m/s^2)');
grid on;
ylabel('m/s^2');
xlabel('Time (s)');

%% ------------------- LOCAL FUNCTIONS -------------------

function sched = run_stage_1_scheduling(errors, Sim, Platoon, Comm, last_alloc)
% Stage-1 scheduling 

    sched     = zeros(Platoon.N, Sim.T_scheduling); % per-follower schedule within period
    slot_load = zeros(1, Sim.T_scheduling);         % number of users assigned to each slot
    
    % Sort vehicles by absolute spacing error (descending)
    [~, sorted_idx] = sort(abs(errors), 'descend');
    
    for ii = 1:Platoon.N
        i   = sorted_idx(ii);   % follower index in [1..N]
        e_i = abs(errors(i));   % absolute spacing error for this follower
        
        if e_i <= Platoon.e_th
            continue;           % skip if error below threshold
        end
        
        % Maximum spacing error reduction within scheduling horizon
        delta_e_max = 0.5 * Platoon.u_max * (Sim.T_scheduling * Sim.dt)^2;
        delta_e_max = max(delta_e_max, 1e-9);         % avoid division by zero
        
        ratio = e_i / delta_e_max;
        f_i   = ceil(ratio);                          % required number of slots (rounded up)
        
        % Limit slots by system-wide NOMA capacity
        f_i   = min(f_i, Comm.O_max * Sim.T_scheduling / Platoon.N);
        if f_i <= 0
            continue;
        end
        
        % Compute slot spacing for this follower within the period
        c_i = max(1, floor(Sim.T_scheduling / f_i));
        
        % Last allocated slot index for this follower (1..T_scheduling)
        tl = last_alloc(i);                       % last slot in previous period
        tn_start = mod(tl + c_i, Sim.T_scheduling);
        if tn_start == 0
            tn_start = Sim.T_scheduling;
        end
        
        % Assign ideal slots tn_start + n*c_i (wrapped within the period)
        for n = 0:(f_i - 1)
            % Convert to 1-based slot index in [1..T_scheduling]
            desired = mod(tn_start + n * c_i - 1, Sim.T_scheduling) + 1;
            [sched, slot_load] = try_assign_slot(sched, slot_load, i, ...
                                                       desired, Sim, Comm);
        end
    end
end

function [sched, slot_load] = try_assign_slot(sched, slot_load, u_idx, desired, Sim, Comm)
% Helper for stage 1: tries to assign a slot to user u_idx with wrapping if needed

    % 1) Try desired slot
    if slot_load(desired) < Comm.O_max
        sched(u_idx, desired) = 1;
        slot_load(desired)    = slot_load(desired) + 1;
        return;
    end
    
    % 2) Search forward from desired+1 to end of period
    for s = (desired + 1):Sim.T_scheduling
        if slot_load(s) < Comm.O_max
            sched(u_idx, s) = 1;
            slot_load(s)    = slot_load(s) + 1;
            return;
        end
    end
    
    % 3) Wrap and search from slot 1 to desired-1
    for s = 1:(desired - 1)
        if slot_load(s) < Comm.O_max
            sched(u_idx, s) = 1;
            slot_load(s)    = slot_load(s) + 1;
            return;
        end
    end
    
    % No slot available: user not scheduled in this period
end

function powers = run_stage_2_power(users, h_signal, H_interf_full, Comm)
% Stage 2: NOMA greedy power allocation

    num_users = length(users);
    powers    = zeros(num_users, 1);
    
    % Single-user case: simple SNR-based allocation
    if num_users == 1
        p_req     = (Comm.R_th * Comm.N0) / h_signal(1); % required power for SINR threshold
        powers(1) = p_req * (p_req <= Comm.P_max);       % only allocate if within power budget
        return;
    end
    
    % Multi-user case: order users by interference and allocate incrementally
    interf_sum = sum(H_interf_full, 2);               % total interference channel gain per user
    [~, order] = sort(interf_sum, 'ascend');          % users with less interference first
    hS_ord     = h_signal(order);
    H_ord      = H_interf_full(order, order);
    p_ord      = zeros(num_users, 1);
    N0         = Comm.N0;
    Rth        = Comm.R_th;
    
    % First user (no prior inter-user interference)
    p_ord(1) = (Rth * N0) / hS_ord(1);
    
    % Subsequent users: account for already allocated users as interference
    for m = 2:num_users
        interf_prev = 0;
        for j = 1:(m - 1)
            interf_prev = interf_prev + p_ord(j) * H_ord(m, j);
        end
        p_ord(m) = Rth * (N0 + interf_prev) / hS_ord(m);
    end
    
    % Map ordered powers back to original user order
    powers(order) = p_ord;
    
    % Enforce maximum power constraint
    over_idx         = powers > Comm.P_max;
    powers(over_idx) = 0;  % users exceeding power budget are considered failed
end

function [p_next, v_next] = update_physics(p, v, u, dt)
% Vehicle kinematics with constant acceleration over time step

    v_next = v + u * dt;               % update velocity (m/s)
    p_next = p + v * dt + 0.5 * u * dt^2; % update position (m)
end

function u = solve_control_qp(p_i, v_i, pred, Platoon, dt)
% Local scalar QP controller implementing

    % Predicted spacing error at t+1
    e_p = pred.p_next - p_i - (Platoon.d_des + Platoon.len); % spacing error (m)
    e_v = pred.v_next - v_i;                                 % relative speed error (m/s)
    
    % Build B and A according to derivation: e(t+1) = B + A*u
    B = [e_p - v_i * dt; e_v];
    A = [-0.5 * dt^2; -dt];              % position and velocity contributions
    
    % Quadratic cost weights
    W = diag([8, 1]);
    
    % QP formulation: min (B + A*u)' W (B + A*u)
    H = 2 * (A' * W * A);               % quadratic term (scalar)
    f = 2 * (A' * W * B);               % linear term (scalar)
    
    % Light regularization for numerical robustness
    eps_reg = 1e-9;
    H       = H + eps_reg;
    
    % Bounds for acceleration input u
    lb = Platoon.u_min;
    ub = Platoon.u_max;
    
    % Solve scalar QP using quadprog
    try
        options = optimoptions('quadprog', 'Display', 'off');
        u       = quadprog(double(H), double(f), [], [], [], [], lb, ub, [], options);
    catch
        % Fallback if solver fails
        u = [];
    end
    
    % Analytic fallback: derivative of cost = 0 → H*u + f = 0 → u = -f/H
    if isempty(u) || ~isfinite(u)
        u_unclamped = -f / H;
        u           = min(max(u_unclamped, lb), ub);
    end
    
    % Final safety check
    if ~isfinite(u)
        u = 0;
    end
end

function est = estimate_predecessor_state(last_info, t_now, dt)
% Estimate predecessor state at current time and one-step-ahead prediction

    % Unpack last received packet
    p_rx = last_info(1);  % last received position (m)
    v_rx = last_info(2);  % last received velocity (m/s)
    u_rx = last_info(3);  % last received acceleration (m/s^2)
    t_rx = last_info(4);  % reception time of last packet (s)
    
    % Compute time elapsed since last update
    if t_rx == 0
        % No packet ever received: assume no evolution
        delta_t = 0;
    else
        delta_t = t_now - t_rx;
        if delta_t < 0
            delta_t = 0;
        end
    end
    
    % Project predecessor state from t_rx to current time t_now using kinematics
    v_est = v_rx + u_rx * delta_t;                       % estimated velocity at t_now (m/s)
    p_est = p_rx + v_rx * delta_t + 0.5 * u_rx * delta_t^2; % estimated position at t_now (m)
    
    % One-step-ahead prediction for controller
    est.v_next = v_est + u_rx * dt;                      % predicted velocity at t+1 (m/s)
    est.p_next = p_est + v_est * dt + 0.5 * u_rx * dt^2; % predicted position at t+1 (m)
end

function e = calculate_errors(pos_t, Platoon)
% Compute spacing errors between consecutive vehicles

    e = zeros(Platoon.N, 1);
    for i = 1:Platoon.N
        % Row i = predecessor, row i+1 = follower
        e(i) = (pos_t(i) - pos_t(i + 1)) - (Platoon.d_des + Platoon.len);
    end
end

function h_gain = calculate_channel_gain(pos_tx, pos_rx)
% Deterministic path-loss model

    alpha   = 3;       % path-loss exponent
    G0      = 1e-4;    % reference gain constant
    epsilon = 1e-6;    % minimum distance to avoid singularity
    
    dist = abs(pos_tx - pos_rx);   % inter-vehicle distance (m)
    dist = max(dist, epsilon);     % enforce minimum distance
    
    path_loss     = G0 * (dist.^(-alpha)); % large-scale path loss
    rayleigh_gain = 1;                     % no fading (set to random for fading models)
    
    h_gain = path_loss * rayleigh_gain;    % effective channel gain
end
