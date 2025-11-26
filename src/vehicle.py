"""
Vehicle and Platoon models for V2V communication simulation.

This module defines the Vehicle and Platoon classes that model
vehicles in a platoon formation with their positions, velocities,
and communication capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Vehicle:
    """
    Represents a vehicle in the platoon.
    
    Attributes:
        vehicle_id: Unique identifier for the vehicle
        position: Current position (x, y) in meters
        velocity: Current velocity (vx, vy) in m/s
        acceleration: Current acceleration in m/s^2
        tx_power: Transmission power in dBm
        is_leader: Whether this vehicle is the platoon leader
    """
    vehicle_id: int
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([20.0, 0.0]))
    acceleration: float = 0.0
    tx_power: float = 23.0  # dBm, typical V2V transmission power
    is_leader: bool = False
    
    def __post_init__(self):
        """Ensure position and velocity are numpy arrays."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=float)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=float)
    
    def update_state(self, dt: float, new_acceleration: Optional[float] = None):
        """
        Update vehicle state based on kinematics.
        
        Args:
            dt: Time step in seconds
            new_acceleration: New acceleration value (if None, keeps current)
        """
        if new_acceleration is not None:
            self.acceleration = new_acceleration
        
        # Update velocity (assuming acceleration in direction of motion)
        speed = np.linalg.norm(self.velocity)
        if speed > 0:
            direction = self.velocity / speed
        else:
            direction = np.array([1.0, 0.0])
        
        new_speed = max(0, speed + self.acceleration * dt)
        self.velocity = direction * new_speed
        
        # Update position
        self.position = self.position + self.velocity * dt
    
    def get_distance_to(self, other: 'Vehicle') -> float:
        """Calculate Euclidean distance to another vehicle."""
        return np.linalg.norm(self.position - other.position)
    
    def get_speed(self) -> float:
        """Get current speed magnitude."""
        return np.linalg.norm(self.velocity)


class Platoon:
    """
    Represents a platoon of vehicles with a leader and followers.
    
    The platoon maintains formation with specified inter-vehicle spacing.
    
    Attributes:
        vehicles: List of vehicles in the platoon (leader first)
        desired_spacing: Desired inter-vehicle spacing in meters
        desired_velocity: Desired platoon velocity in m/s
    """
    
    def __init__(
        self,
        num_vehicles: int = 5,
        desired_spacing: float = 20.0,
        desired_velocity: float = 20.0,
        initial_position: np.ndarray = None
    ):
        """
        Initialize a platoon with specified parameters.
        
        Args:
            num_vehicles: Number of vehicles in platoon
            desired_spacing: Desired spacing between consecutive vehicles (meters)
            desired_velocity: Desired velocity of the platoon (m/s)
            initial_position: Starting position of the leader vehicle
        """
        self.desired_spacing = desired_spacing
        self.desired_velocity = desired_velocity
        self.vehicles: List[Vehicle] = []
        
        if initial_position is None:
            initial_position = np.array([0.0, 0.0])
        
        # Create vehicles with initial positions
        for i in range(num_vehicles):
            position = initial_position.copy()
            position[0] -= i * desired_spacing  # Vehicles behind leader
            
            vehicle = Vehicle(
                vehicle_id=i,
                position=position,
                velocity=np.array([desired_velocity, 0.0]),
                is_leader=(i == 0)
            )
            self.vehicles.append(vehicle)
    
    @property
    def leader(self) -> Vehicle:
        """Get the platoon leader."""
        return self.vehicles[0]
    
    @property
    def followers(self) -> List[Vehicle]:
        """Get the follower vehicles."""
        return self.vehicles[1:]
    
    def get_spacing_errors(self) -> np.ndarray:
        """
        Calculate spacing errors for all followers.
        
        Returns:
            Array of spacing errors (actual - desired) for each follower
        """
        errors = np.zeros(len(self.vehicles) - 1)
        for i, follower in enumerate(self.followers):
            predecessor = self.vehicles[i]
            actual_spacing = predecessor.get_distance_to(follower)
            errors[i] = actual_spacing - self.desired_spacing
        return errors
    
    def get_velocity_errors(self) -> np.ndarray:
        """
        Calculate velocity errors for all vehicles.
        
        Returns:
            Array of velocity errors (actual - desired) for each vehicle
        """
        errors = np.zeros(len(self.vehicles))
        for i, vehicle in enumerate(self.vehicles):
            errors[i] = vehicle.get_speed() - self.desired_velocity
        return errors
    
    def update_states(self, dt: float, accelerations: Optional[np.ndarray] = None):
        """
        Update all vehicle states.
        
        Args:
            dt: Time step in seconds
            accelerations: Array of accelerations for each vehicle
        """
        if accelerations is None:
            accelerations = [None] * len(self.vehicles)
        
        for vehicle, acc in zip(self.vehicles, accelerations):
            vehicle.update_state(dt, acc)
    
    def get_positions(self) -> np.ndarray:
        """Get positions of all vehicles."""
        return np.array([v.position for v in self.vehicles])
    
    def get_velocities(self) -> np.ndarray:
        """Get velocities of all vehicles."""
        return np.array([v.velocity for v in self.vehicles])
    
    def get_inter_vehicle_distances(self) -> np.ndarray:
        """Get distances between consecutive vehicles."""
        distances = np.zeros(len(self.vehicles) - 1)
        for i in range(len(self.vehicles) - 1):
            distances[i] = self.vehicles[i].get_distance_to(self.vehicles[i + 1])
        return distances
