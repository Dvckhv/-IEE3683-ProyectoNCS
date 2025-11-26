#!/usr/bin/env python3
"""
Main script for Joint Platoon Control and Resource Allocation (NOMA-V2V).

This script demonstrates the joint optimization of:
1. Platoon control for vehicle spacing and velocity
2. NOMA resource allocation for V2V communications
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.simulation import PlatoonNOMASimulation, SimulationConfig, run_comparison_study


def main():
    """Run the main simulation demonstration."""
    print("=" * 70)
    print("Joint Platoon Control and Resource Allocation (NOMA-V2V)")
    print("=" * 70)
    print()
    
    # Configuration
    config = SimulationConfig(
        num_vehicles=5,
        desired_spacing=20.0,  # meters
        desired_velocity=20.0,  # m/s (72 km/h)
        num_subchannels=4,
        max_users_per_subchannel=2,
        max_tx_power=0.2,  # Watts (23 dBm)
        qos_rate=1e6,  # 1 Mbps
        simulation_time=30.0,  # seconds
        time_step=0.1,  # seconds
        control_method='comm_aware',
        allocation_method='greedy',
        power_method='water_filling'
    )
    
    print("Simulation Configuration:")
    print(f"  - Number of vehicles: {config.num_vehicles}")
    print(f"  - Desired spacing: {config.desired_spacing} m")
    print(f"  - Desired velocity: {config.desired_velocity} m/s ({config.desired_velocity * 3.6:.1f} km/h)")
    print(f"  - Number of subchannels: {config.num_subchannels}")
    print(f"  - Max TX power: {10 * np.log10(config.max_tx_power) + 30:.1f} dBm")
    print(f"  - QoS rate requirement: {config.qos_rate / 1e6:.1f} Mbps")
    print(f"  - Simulation time: {config.simulation_time} s")
    print(f"  - Control method: {config.control_method}")
    print(f"  - Allocation method: {config.allocation_method}")
    print(f"  - Power method: {config.power_method}")
    print()
    
    # Run simulation
    print("Running simulation...")
    print("-" * 40)
    
    sim = PlatoonNOMASimulation(config, random_seed=42)
    
    # Run with progress output
    num_steps = int(config.simulation_time / config.time_step)
    for step_idx in range(num_steps):
        comm_metrics, control_metrics = sim.step()
        
        # Introduce leader braking disturbance at t=10s
        if 10.0 <= sim.current_time < 12.0:
            sim.platoon.leader.acceleration = -3.0
        elif 12.0 <= sim.current_time < 12.1:
            sim.platoon.leader.acceleration = 0.0
        
        if step_idx % 50 == 0 or step_idx == num_steps - 1:
            print(f"  t = {sim.current_time:5.1f}s: "
                  f"Sum Rate = {comm_metrics['sum_rate_mbps']:6.2f} Mbps, "
                  f"Spacing Err = {control_metrics['mean_spacing_error']:5.2f} m, "
                  f"String Stable = {control_metrics['string_stable']}")
    
    # Get final results
    results = sim.get_results()
    
    print()
    print("-" * 40)
    print("Simulation Results:")
    print("-" * 40)
    print()
    
    # Communication Performance
    print("Communication Performance:")
    print(f"  - Average Sum Rate: {results['avg_sum_rate_mbps']:.2f} Mbps")
    print(f"  - Final Sum Rate: {results['final_sum_rate_mbps']:.2f} Mbps")
    print(f"  - QoS Satisfaction Ratio: {results['avg_qos_satisfaction'] * 100:.1f}%")
    print()
    
    # Platoon Control Performance
    print("Platoon Control Performance:")
    print(f"  - Average Spacing Error: {results['avg_spacing_error']:.2f} m")
    print(f"  - Average Velocity Error: {results['avg_velocity_error']:.2f} m/s")
    print(f"  - String Stability Ratio: {results['string_stable_ratio'] * 100:.1f}%")
    print()
    
    # Final vehicle positions
    final_positions = results['positions_history'][-1]
    print("Final Vehicle Positions (x-coordinate):")
    for i, pos in enumerate(final_positions):
        role = "Leader" if i == 0 else f"Follower {i}"
        print(f"  - {role}: {pos[0]:.1f} m")
    
    print()
    print("=" * 70)
    print("Comparison Study: Different Control and Allocation Methods")
    print("=" * 70)
    print()
    
    # Run comparison study
    comparison_results = run_comparison_study(
        num_vehicles=5,
        simulation_time=30.0,
        random_seed=42
    )
    
    print(f"{'Method':<30} {'Sum Rate (Mbps)':<16} {'QoS Sat.':<12} {'Spacing Err':<12} {'String Stable':<12}")
    print("-" * 82)
    
    for method_name, metrics in comparison_results.items():
        print(f"{method_name:<30} "
              f"{metrics['avg_sum_rate_mbps']:<16.2f} "
              f"{metrics['avg_qos_satisfaction'] * 100:<12.1f}% "
              f"{metrics['avg_spacing_error']:<12.2f} "
              f"{metrics['string_stable_ratio'] * 100:<12.1f}%")
    
    print()
    print("=" * 70)
    print("Simulation completed successfully!")
    print("=" * 70)
    
    return results, comparison_results


if __name__ == "__main__":
    main()
