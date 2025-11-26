# Joint Platoon Control and Resource Allocation (NOMA-V2V)

A Python simulation framework for joint optimization of vehicle platoon control and Non-Orthogonal Multiple Access (NOMA) based Vehicle-to-Vehicle (V2V) communication resource allocation.

## Overview

This project implements a comprehensive simulation environment for studying the interaction between:
- **Vehicle Platoon Control**: Cooperative Adaptive Cruise Control (CACC) for maintaining formation
- **NOMA-V2V Communications**: Non-Orthogonal Multiple Access for efficient spectrum utilization
- **Resource Allocation**: Joint subchannel assignment and power allocation optimization

## Features

### Platoon Control
- Predecessor-Following (PF) control
- Leader-Predecessor Following (LPF) control
- Bidirectional (BD) control
- Communication-aware adaptive control
- String stability analysis

### NOMA System Model
- Power-domain NOMA with Successive Interference Cancellation (SIC)
- Multiple subchannels with user clustering
- V2V channel model with path loss, shadow fading, and fast fading

### Resource Allocation
- Greedy subchannel assignment
- Hungarian algorithm-based optimal assignment
- Water-filling power allocation
- Fixed-ratio power allocation
- QoS-aware power allocation

## Installation

```bash
# Clone the repository
git clone https://github.com/Dvckhv/-IEE3683-ProyectoNCS.git
cd -IEE3683-ProyectoNCS

# Install dependencies
pip install numpy scipy matplotlib
```

## Usage

### Running the Simulation

```bash
python main.py
```

### Running Tests

```bash
python tests/test_all.py
```

### Programmatic Usage

```python
from src.simulation import PlatoonNOMASimulation, SimulationConfig

# Configure simulation
config = SimulationConfig(
    num_vehicles=5,
    desired_spacing=20.0,  # meters
    desired_velocity=20.0,  # m/s
    num_subchannels=4,
    max_tx_power=0.2,  # Watts
    qos_rate=1e6,  # 1 Mbps
    simulation_time=30.0,  # seconds
    control_method='comm_aware',
    allocation_method='greedy',
    power_method='water_filling'
)

# Run simulation
sim = PlatoonNOMASimulation(config, random_seed=42)
results = sim.run(verbose=True)

# Access results
print(f"Average Sum Rate: {results['avg_sum_rate_mbps']:.2f} Mbps")
print(f"Average Spacing Error: {results['avg_spacing_error']:.2f} m")
```

## Project Structure

```
├── main.py                    # Main simulation script
├── src/
│   ├── __init__.py           # Package initialization
│   ├── vehicle.py            # Vehicle and Platoon classes
│   ├── channel.py            # V2V channel model
│   ├── noma.py               # NOMA system model
│   ├── resource_allocation.py # Resource allocation algorithms
│   ├── platoon_control.py    # Platoon control algorithms
│   └── simulation.py         # Main simulation environment
├── tests/
│   ├── __init__.py
│   └── test_all.py           # Unit tests
└── README.md
```

## System Model

### V2V Channel Model
The V2V channel follows a log-distance path loss model with:
- Carrier frequency: 5.9 GHz (ITS band)
- Path loss exponent: 3.0 (urban environment)
- Log-normal shadow fading (σ = 3 dB)
- Rayleigh fast fading

### NOMA Multiplexing
- Multiple users share subchannels via power-domain multiplexing
- SIC decoding at receivers (strong users decode weak users first)
- Power allocation favors weak channel users for fairness

### Platoon Control
The constant time gap policy maintains spacing:
```
d_desired = d_standstill + τ × v
```
where τ is the time headway parameter.

## Example Results

```
Communication Performance:
  - Average Sum Rate: 104.07 Mbps
  - QoS Satisfaction Ratio: 57.4%

Platoon Control Performance:
  - Average Spacing Error: 0.14 m
  - String Stability Ratio: 100.0%

Comparison of Methods:
Method                    Sum Rate  QoS Sat.  Spacing Err
---------------------------------------------------------
CommAware + QoS-Aware     111.11    100.0%    0.00 m
CommAware + Hungarian     110.65    75.2%     0.00 m
PF + Greedy               104.10    57.4%     0.00 m
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_vehicles` | Number of vehicles in platoon | 5 |
| `desired_spacing` | Target inter-vehicle spacing (m) | 20.0 |
| `desired_velocity` | Target platoon velocity (m/s) | 20.0 |
| `num_subchannels` | NOMA subchannels | 4 |
| `max_users_per_subchannel` | Max NOMA cluster size | 2 |
| `max_tx_power` | Max TX power (Watts) | 0.2 |
| `qos_rate` | QoS data rate requirement (bps) | 1e6 |
| `control_method` | Control algorithm | 'comm_aware' |
| `allocation_method` | Subchannel allocation | 'greedy' |
| `power_method` | Power allocation | 'water_filling' |

## License

This project is for educational purposes as part of course IEE3683.

## References

1. 3GPP TR 36.885: Study on LTE-based V2X Services
2. NOMA for V2X Communications: A Survey
3. Cooperative Adaptive Cruise Control: A Survey