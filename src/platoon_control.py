"""
Platoon Control Module for NOMA-V2V Systems.

This module implements platoon control algorithms including:
- Cooperative Adaptive Cruise Control (CACC)
- Communication-aware platoon control
- String stability analysis
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .vehicle import Platoon, Vehicle


@dataclass
class ControlParameters:
    """
    Parameters for platoon control.
    
    Attributes:
        k_p: Position/spacing error gain
        k_v: Velocity error gain
        k_a: Acceleration feedforward gain
        tau: Time headway (constant time gap)
        max_acceleration: Maximum acceleration in m/s^2
        max_deceleration: Maximum deceleration in m/s^2
        comm_delay: Communication delay in seconds
    """
    k_p: float = 0.3
    k_v: float = 0.5
    k_a: float = 0.1
    tau: float = 0.5
    max_acceleration: float = 3.0
    max_deceleration: float = -5.0
    comm_delay: float = 0.01


class PlatoonController:
    """
    Platoon controller implementing CACC with communication awareness.
    
    Implements various control strategies:
    - Predecessor-Following (PF): Follow only the vehicle ahead
    - Bidirectional (BD): Consider both predecessor and successor
    - Leader-Predecessor Following (LPF): Follow both leader and predecessor
    """
    
    def __init__(
        self,
        platoon: Platoon,
        params: Optional[ControlParameters] = None
    ):
        """
        Initialize the platoon controller.
        
        Args:
            platoon: The platoon to control
            params: Control parameters
        """
        self.platoon = platoon
        self.params = params or ControlParameters()
        
        # Communication quality indicators (1.0 = perfect, 0.0 = no comm)
        self.comm_quality = np.ones(len(platoon.vehicles))
        
        # State history for stability analysis
        self.spacing_history = []
        self.velocity_history = []
        self.acceleration_history = []
    
    def set_communication_quality(self, quality: np.ndarray):
        """
        Set communication quality based on V2V link performance.
        
        Args:
            quality: Array of communication quality indicators per vehicle
        """
        self.comm_quality = np.clip(quality, 0.0, 1.0)
    
    def calculate_desired_spacing(self, vehicle_idx: int) -> float:
        """
        Calculate desired spacing using constant time gap policy.
        
        Args:
            vehicle_idx: Index of the follower vehicle
            
        Returns:
            Desired spacing in meters
        """
        if vehicle_idx == 0:
            return 0.0  # Leader has no predecessor
        
        vehicle = self.platoon.vehicles[vehicle_idx]
        speed = vehicle.get_speed()
        
        # Constant time gap policy: d = d_standstill + tau * v
        d_standstill = 5.0  # Minimum standstill distance
        desired_spacing = d_standstill + self.params.tau * speed
        
        return max(desired_spacing, self.platoon.desired_spacing)
    
    def predecessor_following_control(self) -> np.ndarray:
        """
        Predecessor-Following (PF) control strategy.
        
        Each follower adjusts acceleration based on spacing error
        and relative velocity to the predecessor.
        
        Returns:
            Array of accelerations for all vehicles
        """
        accelerations = np.zeros(len(self.platoon.vehicles))
        
        # Leader maintains constant velocity (or follows external command)
        accelerations[0] = 0.0
        
        for i in range(1, len(self.platoon.vehicles)):
            follower = self.platoon.vehicles[i]
            predecessor = self.platoon.vehicles[i - 1]
            
            # Spacing error
            actual_spacing = predecessor.get_distance_to(follower)
            desired_spacing = self.calculate_desired_spacing(i)
            spacing_error = actual_spacing - desired_spacing
            
            # Relative velocity
            relative_velocity = predecessor.get_speed() - follower.get_speed()
            
            # PF control law with communication quality
            comm_factor = self.comm_quality[i - 1]
            
            acceleration = (
                self.params.k_p * spacing_error +
                self.params.k_v * relative_velocity * comm_factor +
                self.params.k_a * predecessor.acceleration * comm_factor
            )
            
            # Apply limits
            acceleration = np.clip(
                acceleration,
                self.params.max_deceleration,
                self.params.max_acceleration
            )
            
            accelerations[i] = acceleration
        
        return accelerations
    
    def leader_predecessor_following_control(self) -> np.ndarray:
        """
        Leader-Predecessor Following (LPF) control strategy.
        
        Uses information from both the leader and the immediate predecessor
        for improved string stability.
        
        Returns:
            Array of accelerations for all vehicles
        """
        accelerations = np.zeros(len(self.platoon.vehicles))
        leader = self.platoon.leader
        
        accelerations[0] = 0.0
        
        for i in range(1, len(self.platoon.vehicles)):
            follower = self.platoon.vehicles[i]
            predecessor = self.platoon.vehicles[i - 1]
            
            # Spacing error to predecessor
            actual_spacing = predecessor.get_distance_to(follower)
            desired_spacing = self.calculate_desired_spacing(i)
            spacing_error = actual_spacing - desired_spacing
            
            # Velocity errors
            rel_vel_pred = predecessor.get_speed() - follower.get_speed()
            rel_vel_leader = leader.get_speed() - follower.get_speed()
            
            # Acceleration feedforward
            pred_acc = predecessor.acceleration * self.comm_quality[i - 1]
            leader_acc = leader.acceleration * self.comm_quality[0]
            
            # LPF control law
            w_pred = 0.7  # Weight for predecessor
            w_leader = 0.3  # Weight for leader
            
            acceleration = (
                self.params.k_p * spacing_error +
                self.params.k_v * (w_pred * rel_vel_pred + w_leader * rel_vel_leader) +
                self.params.k_a * (w_pred * pred_acc + w_leader * leader_acc)
            )
            
            acceleration = np.clip(
                acceleration,
                self.params.max_deceleration,
                self.params.max_acceleration
            )
            
            accelerations[i] = acceleration
        
        return accelerations
    
    def bidirectional_control(self) -> np.ndarray:
        """
        Bidirectional (BD) control strategy.
        
        Uses information from both predecessor and successor vehicles.
        
        Returns:
            Array of accelerations for all vehicles
        """
        accelerations = np.zeros(len(self.platoon.vehicles))
        n = len(self.platoon.vehicles)
        
        accelerations[0] = 0.0
        
        for i in range(1, n):
            follower = self.platoon.vehicles[i]
            predecessor = self.platoon.vehicles[i - 1]
            
            # Spacing and velocity error to predecessor
            spacing_pred = predecessor.get_distance_to(follower)
            desired_spacing = self.calculate_desired_spacing(i)
            spacing_error_pred = spacing_pred - desired_spacing
            vel_error_pred = predecessor.get_speed() - follower.get_speed()
            
            # Consider successor if exists
            if i < n - 1:
                successor = self.platoon.vehicles[i + 1]
                spacing_succ = follower.get_distance_to(successor)
                spacing_error_succ = spacing_succ - self.calculate_desired_spacing(i + 1)
                vel_error_succ = successor.get_speed() - follower.get_speed()
                
                # Bidirectional combination
                w_front = 0.6
                w_back = 0.4
                spacing_error = w_front * spacing_error_pred - w_back * spacing_error_succ
                vel_error = w_front * vel_error_pred * self.comm_quality[i - 1]
            else:
                spacing_error = spacing_error_pred
                vel_error = vel_error_pred * self.comm_quality[i - 1]
            
            acceleration = (
                self.params.k_p * spacing_error +
                self.params.k_v * vel_error +
                self.params.k_a * predecessor.acceleration * self.comm_quality[i - 1]
            )
            
            acceleration = np.clip(
                acceleration,
                self.params.max_deceleration,
                self.params.max_acceleration
            )
            
            accelerations[i] = acceleration
        
        return accelerations
    
    def communication_aware_control(
        self,
        data_rates: np.ndarray,
        rate_threshold: float = 1e6
    ) -> np.ndarray:
        """
        Communication-aware control that adapts based on V2V link quality.
        
        When communication quality degrades, falls back to more conservative
        control strategies.
        
        Args:
            data_rates: Data rates for each V2V link
            rate_threshold: Threshold for acceptable communication quality
            
        Returns:
            Array of accelerations for all vehicles
        """
        # Update communication quality based on data rates
        if len(data_rates) > 0:
            self.comm_quality = np.clip(data_rates / rate_threshold, 0.0, 1.0)
        
        # Use LPF control as default with communication weighting
        accelerations = np.zeros(len(self.platoon.vehicles))
        leader = self.platoon.leader
        
        accelerations[0] = 0.0
        
        for i in range(1, len(self.platoon.vehicles)):
            follower = self.platoon.vehicles[i]
            predecessor = self.platoon.vehicles[i - 1]
            
            # Calculate spacing error
            actual_spacing = predecessor.get_distance_to(follower)
            desired_spacing = self.calculate_desired_spacing(i)
            spacing_error = actual_spacing - desired_spacing
            
            # Velocity errors with communication weighting
            rel_vel_pred = predecessor.get_speed() - follower.get_speed()
            rel_vel_leader = leader.get_speed() - follower.get_speed()
            
            # Adaptive weights based on communication quality
            comm_idx = min(i - 1, len(self.comm_quality) - 1)
            q = self.comm_quality[comm_idx] if comm_idx >= 0 else 1.0
            
            # When comm is good, use more aggressive control
            # When comm is poor, rely more on local sensing
            k_p_adaptive = self.params.k_p * (1 + 0.5 * q)
            k_v_adaptive = self.params.k_v * q
            k_a_adaptive = self.params.k_a * q
            
            acceleration = (
                k_p_adaptive * spacing_error +
                k_v_adaptive * rel_vel_pred +
                k_a_adaptive * predecessor.acceleration
            )
            
            # Add leader info if communication is good
            if q > 0.5:
                acceleration += 0.2 * self.params.k_v * rel_vel_leader
            
            acceleration = np.clip(
                acceleration,
                self.params.max_deceleration,
                self.params.max_acceleration
            )
            
            accelerations[i] = acceleration
        
        return accelerations
    
    def step(
        self,
        dt: float,
        control_method: str = 'pf',
        data_rates: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Execute one control step.
        
        Args:
            dt: Time step in seconds
            control_method: Control method ('pf', 'lpf', 'bd', 'comm_aware')
            data_rates: Optional data rates for communication-aware control
            
        Returns:
            Tuple of (accelerations, metrics)
        """
        # Calculate accelerations based on control method
        if control_method == 'lpf':
            accelerations = self.leader_predecessor_following_control()
        elif control_method == 'bd':
            accelerations = self.bidirectional_control()
        elif control_method == 'comm_aware' and data_rates is not None:
            accelerations = self.communication_aware_control(data_rates)
        else:
            accelerations = self.predecessor_following_control()
        
        # Update platoon states
        self.platoon.update_states(dt, accelerations)
        
        # Record history
        self.spacing_history.append(self.platoon.get_inter_vehicle_distances())
        self.velocity_history.append([v.get_speed() for v in self.platoon.vehicles])
        self.acceleration_history.append(accelerations.copy())
        
        # Calculate metrics
        spacing_errors = self.platoon.get_spacing_errors()
        velocity_errors = self.platoon.get_velocity_errors()
        
        metrics = {
            'mean_spacing_error': np.mean(np.abs(spacing_errors)),
            'max_spacing_error': np.max(np.abs(spacing_errors)),
            'mean_velocity_error': np.mean(np.abs(velocity_errors)),
            'max_velocity_error': np.max(np.abs(velocity_errors)),
            'mean_acceleration': np.mean(np.abs(accelerations[1:])),
            'string_stable': self.check_string_stability()
        }
        
        return accelerations, metrics
    
    def check_string_stability(self, window: int = 10) -> bool:
        """
        Check if platoon maintains string stability.
        
        String stability means disturbances do not amplify as they
        propagate through the platoon.
        
        Args:
            window: Number of recent steps to analyze
            
        Returns:
            True if string stable
        """
        if len(self.spacing_history) < window:
            return True
        
        recent_spacing = np.array(self.spacing_history[-window:])
        
        # Check if spacing errors are not amplifying
        # Compare variance at front vs back of platoon
        if recent_spacing.shape[1] < 2:
            return True
        
        front_var = np.var(recent_spacing[:, 0])
        back_var = np.var(recent_spacing[:, -1])
        
        # String stable if back variance is not much larger than front
        return back_var <= 1.5 * front_var + 0.1
    
    def get_performance_metrics(self) -> dict:
        """
        Get comprehensive platoon control performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if len(self.spacing_history) == 0:
            return {}
        
        spacing_history = np.array(self.spacing_history)
        velocity_history = np.array(self.velocity_history)
        
        # Spacing metrics
        spacing_errors = spacing_history - self.platoon.desired_spacing
        
        # Velocity metrics
        velocity_errors = velocity_history - self.platoon.desired_velocity
        
        return {
            'avg_spacing_error': np.mean(np.abs(spacing_errors)),
            'std_spacing_error': np.std(spacing_errors),
            'max_spacing_error': np.max(np.abs(spacing_errors)),
            'avg_velocity_error': np.mean(np.abs(velocity_errors)),
            'std_velocity_error': np.std(velocity_errors),
            'string_stable': self.check_string_stability(),
            'history_length': len(self.spacing_history)
        }
    
    def reset_history(self):
        """Clear control history."""
        self.spacing_history = []
        self.velocity_history = []
        self.acceleration_history = []
