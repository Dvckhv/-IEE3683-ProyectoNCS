"""
Joint Platoon Control and Resource Allocation Simulation.

This module provides the main simulation environment that integrates:
- Vehicle platoon dynamics
- V2V NOMA communication system
- Resource allocation
- Platoon control with communication awareness
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .vehicle import Platoon, Vehicle
from .channel import V2VChannel, ChannelParameters
from .noma import NOMASystem, NOMAUser
from .resource_allocation import ResourceAllocator
from .platoon_control import PlatoonController, ControlParameters


@dataclass
class SimulationConfig:
    """
    Configuration for the joint platoon control and resource allocation simulation.
    
    Attributes:
        num_vehicles: Number of vehicles in the platoon
        desired_spacing: Desired inter-vehicle spacing (meters)
        desired_velocity: Desired platoon velocity (m/s)
        num_subchannels: Number of NOMA subchannels
        max_users_per_subchannel: Maximum users per NOMA cluster
        max_tx_power: Maximum transmission power (Watts)
        qos_rate: QoS data rate requirement (bits/s)
        simulation_time: Total simulation time (seconds)
        time_step: Simulation time step (seconds)
        control_method: Platoon control method
        allocation_method: Resource allocation method
        power_method: Power allocation method
    """
    num_vehicles: int = 5
    desired_spacing: float = 20.0
    desired_velocity: float = 20.0
    num_subchannels: int = 4
    max_users_per_subchannel: int = 2
    max_tx_power: float = 0.2
    qos_rate: float = 1e6
    simulation_time: float = 30.0
    time_step: float = 0.1
    control_method: str = 'comm_aware'
    allocation_method: str = 'greedy'
    power_method: str = 'water_filling'


class PlatoonNOMASimulation:
    """
    Main simulation class for joint platoon control and NOMA resource allocation.
    
    This simulation models:
    1. A platoon of vehicles with CACC-style control
    2. V2V communication using NOMA on shared subchannels
    3. Joint optimization of communication resources and platoon control
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the simulation.
        
        Args:
            config: Simulation configuration
            random_seed: Random seed for reproducibility
        """
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize platoon
        self.platoon = Platoon(
            num_vehicles=self.config.num_vehicles,
            desired_spacing=self.config.desired_spacing,
            desired_velocity=self.config.desired_velocity
        )
        
        # Initialize NOMA system
        channel_params = ChannelParameters()
        self.noma_system = NOMASystem(
            num_subchannels=self.config.num_subchannels,
            max_users_per_subchannel=self.config.max_users_per_subchannel,
            max_tx_power=self.config.max_tx_power,
            channel_params=channel_params
        )
        
        # Initialize resource allocator
        self.allocator = ResourceAllocator(self.noma_system)
        
        # Initialize platoon controller
        control_params = ControlParameters()
        self.controller = PlatoonController(self.platoon, control_params)
        
        # Define V2V links (each vehicle communicates with its neighbors)
        self.v2v_links = self._create_v2v_links()
        
        # Simulation state
        self.current_time = 0.0
        self.time_history = []
        self.metrics_history = []
        self.positions_history = []
        self.data_rates_history = []
    
    def _create_v2v_links(self) -> List[Tuple[int, int]]:
        """
        Create V2V communication links for the platoon.
        
        Each follower vehicle needs to receive information from:
        - Its predecessor (for CACC)
        - The leader (for enhanced stability)
        
        Returns:
            List of (tx_id, rx_id) tuples
        """
        links = []
        
        # Predecessor-to-follower links (essential for CACC)
        for i in range(self.config.num_vehicles - 1):
            links.append((i, i + 1))  # Vehicle i transmits to vehicle i+1
        
        # Leader-to-all-followers links (for LPF control)
        for i in range(2, self.config.num_vehicles):
            links.append((0, i))  # Leader transmits to vehicle i
        
        return links
    
    def update_v2v_channels(self) -> Dict:
        """
        Update V2V channel states based on current vehicle positions.
        
        Returns:
            Dictionary with channel and allocation metrics
        """
        # Get current vehicle positions
        positions = self.platoon.get_positions()
        
        # Create V2V links with current distances
        qos_rates = [self.config.qos_rate] * len(self.v2v_links)
        
        self.noma_system.create_v2v_links(
            vehicle_positions=positions,
            link_pairs=self.v2v_links,
            qos_rates=qos_rates,
            rng=self.rng
        )
        
        # Perform resource allocation
        metrics = self.allocator.joint_allocation(
            method=self.config.allocation_method,
            power_method=self.config.power_method
        )
        
        return metrics
    
    def get_link_data_rates(self) -> np.ndarray:
        """
        Get data rates for predecessor-follower links.
        
        Returns:
            Array of data rates for each follower vehicle
        """
        # Find predecessor-follower links
        rates = []
        for i in range(1, self.config.num_vehicles):
            # Find the link from predecessor (i-1) to follower (i)
            for user in self.noma_system.users:
                if user.tx_vehicle_id == i - 1 and user.rx_vehicle_id == i:
                    rates.append(user.data_rate)
                    break
            else:
                rates.append(0.0)  # Link not found
        
        return np.array(rates)
    
    def step(self) -> Tuple[Dict, Dict]:
        """
        Execute one simulation step.
        
        Returns:
            Tuple of (comm_metrics, control_metrics)
        """
        # Update V2V channels and allocate resources
        comm_metrics = self.update_v2v_channels()
        
        # Get data rates for control
        data_rates = self.get_link_data_rates()
        
        # Execute platoon control step
        _, control_metrics = self.controller.step(
            dt=self.config.time_step,
            control_method=self.config.control_method,
            data_rates=data_rates
        )
        
        # Update time
        self.current_time += self.config.time_step
        
        # Record history
        self.time_history.append(self.current_time)
        self.metrics_history.append({
            'comm': comm_metrics,
            'control': control_metrics
        })
        self.positions_history.append(self.platoon.get_positions().copy())
        self.data_rates_history.append(data_rates.copy())
        
        return comm_metrics, control_metrics
    
    def run(self, verbose: bool = False) -> Dict:
        """
        Run the full simulation.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary with simulation results
        """
        num_steps = int(self.config.simulation_time / self.config.time_step)
        
        for step_idx in range(num_steps):
            comm_metrics, control_metrics = self.step()
            
            if verbose and step_idx % 50 == 0:
                print(f"Step {step_idx}/{num_steps}: "
                      f"Sum Rate = {comm_metrics['sum_rate_mbps']:.2f} Mbps, "
                      f"Spacing Error = {control_metrics['mean_spacing_error']:.2f} m")
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """
        Get comprehensive simulation results.
        
        Returns:
            Dictionary with all simulation metrics and history
        """
        if len(self.metrics_history) == 0:
            return {}
        
        # Aggregate metrics
        comm_metrics_list = [m['comm'] for m in self.metrics_history]
        control_metrics_list = [m['control'] for m in self.metrics_history]
        
        avg_sum_rate = np.mean([m['sum_rate_mbps'] for m in comm_metrics_list])
        avg_qos_satisfaction = np.mean([m['qos_satisfaction_ratio'] for m in comm_metrics_list])
        avg_spacing_error = np.mean([m['mean_spacing_error'] for m in control_metrics_list])
        avg_velocity_error = np.mean([m['mean_velocity_error'] for m in control_metrics_list])
        
        # String stability analysis
        string_stable_ratio = np.mean([m['string_stable'] for m in control_metrics_list])
        
        return {
            'config': self.config,
            'simulation_time': self.current_time,
            'num_steps': len(self.time_history),
            
            # Communication metrics
            'avg_sum_rate_mbps': avg_sum_rate,
            'final_sum_rate_mbps': comm_metrics_list[-1]['sum_rate_mbps'],
            'avg_qos_satisfaction': avg_qos_satisfaction,
            
            # Control metrics
            'avg_spacing_error': avg_spacing_error,
            'avg_velocity_error': avg_velocity_error,
            'string_stable_ratio': string_stable_ratio,
            
            # History
            'time_history': np.array(self.time_history),
            'positions_history': np.array(self.positions_history),
            'data_rates_history': np.array(self.data_rates_history),
            'metrics_history': self.metrics_history
        }
    
    def introduce_leader_disturbance(
        self,
        disturbance_type: str = 'brake',
        magnitude: float = -3.0,
        duration: float = 2.0
    ):
        """
        Introduce a disturbance to the leader vehicle.
        
        Args:
            disturbance_type: Type of disturbance ('brake', 'accelerate')
            magnitude: Acceleration magnitude (m/s^2)
            duration: Duration of disturbance (seconds)
        """
        leader = self.platoon.leader
        
        if disturbance_type == 'brake':
            leader.acceleration = magnitude
        elif disturbance_type == 'accelerate':
            leader.acceleration = abs(magnitude)
    
    def reset(self):
        """Reset the simulation to initial state."""
        self.platoon = Platoon(
            num_vehicles=self.config.num_vehicles,
            desired_spacing=self.config.desired_spacing,
            desired_velocity=self.config.desired_velocity
        )
        
        self.controller = PlatoonController(
            self.platoon,
            ControlParameters()
        )
        
        self.current_time = 0.0
        self.time_history = []
        self.metrics_history = []
        self.positions_history = []
        self.data_rates_history = []


def run_comparison_study(
    num_vehicles: int = 5,
    simulation_time: float = 30.0,
    random_seed: int = 42
) -> Dict:
    """
    Run a comparison study of different control and allocation methods.
    
    Args:
        num_vehicles: Number of vehicles in platoon
        simulation_time: Simulation duration
        random_seed: Random seed
        
    Returns:
        Dictionary with comparison results
    """
    methods = [
        ('pf', 'greedy', 'water_filling', 'PF + Greedy'),
        ('lpf', 'greedy', 'water_filling', 'LPF + Greedy'),
        ('comm_aware', 'greedy', 'water_filling', 'CommAware + Greedy'),
        ('comm_aware', 'hungarian', 'water_filling', 'CommAware + Hungarian'),
        ('comm_aware', 'greedy', 'qos_aware', 'CommAware + QoS-Aware'),
    ]
    
    results = {}
    
    for control, alloc, power, name in methods:
        config = SimulationConfig(
            num_vehicles=num_vehicles,
            simulation_time=simulation_time,
            control_method=control,
            allocation_method=alloc,
            power_method=power
        )
        
        sim = PlatoonNOMASimulation(config, random_seed)
        sim.run(verbose=False)
        result = sim.get_results()
        
        results[name] = {
            'avg_sum_rate_mbps': result['avg_sum_rate_mbps'],
            'avg_qos_satisfaction': result['avg_qos_satisfaction'],
            'avg_spacing_error': result['avg_spacing_error'],
            'string_stable_ratio': result['string_stable_ratio']
        }
    
    return results
