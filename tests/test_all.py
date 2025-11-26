"""
Tests for Joint Platoon Control and Resource Allocation (NOMA-V2V).

This module contains unit tests for all components of the simulation.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vehicle import Vehicle, Platoon
from src.channel import V2VChannel, ChannelParameters
from src.noma import NOMASystem, NOMAUser, NOMACluster
from src.resource_allocation import ResourceAllocator
from src.platoon_control import PlatoonController, ControlParameters
from src.simulation import PlatoonNOMASimulation, SimulationConfig


def test_vehicle():
    """Test Vehicle class functionality."""
    print("Testing Vehicle class...")
    
    # Create vehicle
    v = Vehicle(
        vehicle_id=0,
        position=np.array([0.0, 0.0]),
        velocity=np.array([20.0, 0.0])
    )
    
    assert v.vehicle_id == 0
    assert np.allclose(v.position, [0.0, 0.0])
    assert np.allclose(v.velocity, [20.0, 0.0])
    assert v.get_speed() == 20.0
    
    # Test state update
    v.update_state(dt=1.0, new_acceleration=2.0)
    # Position: x = x0 + v_new * dt where v_new = v0 + a*dt = 20 + 2*1 = 22
    # So position should be 22 m (velocity updated then position uses new velocity)
    assert np.allclose(v.position, [22.0, 0.0], atol=1.0)
    assert v.get_speed() == 22.0
    
    # Test distance calculation
    v2 = Vehicle(vehicle_id=1, position=np.array([30.0, 0.0]))
    # v is at [22, 0], v2 is at [30, 0] => distance = 8m
    assert np.isclose(v.get_distance_to(v2), 8.0, atol=1.0)
    
    print("  Vehicle tests passed!")


def test_platoon():
    """Test Platoon class functionality."""
    print("Testing Platoon class...")
    
    platoon = Platoon(
        num_vehicles=5,
        desired_spacing=20.0,
        desired_velocity=20.0
    )
    
    assert len(platoon.vehicles) == 5
    assert platoon.leader.is_leader
    assert len(platoon.followers) == 4
    
    # Check initial spacing
    distances = platoon.get_inter_vehicle_distances()
    assert len(distances) == 4
    assert np.allclose(distances, 20.0, atol=0.1)
    
    # Check spacing errors (should be near zero initially)
    errors = platoon.get_spacing_errors()
    assert np.allclose(errors, 0.0, atol=0.1)
    
    print("  Platoon tests passed!")


def test_channel():
    """Test V2VChannel class functionality."""
    print("Testing V2VChannel class...")
    
    channel = V2VChannel()
    
    # Test path loss calculation
    pl_10m = channel.calculate_path_loss(10.0)
    pl_100m = channel.calculate_path_loss(100.0)
    
    # Path loss should increase with distance
    assert pl_100m > pl_10m
    
    # Test SNR calculation
    snr = channel.calculate_snr(
        tx_power_dBm=23.0,
        distance=20.0,
        include_shadow=False,
        include_fast_fading=False
    )
    
    # SNR should be positive for reasonable distances
    assert snr > 0
    
    # Test channel matrix
    tx_pos = np.array([[0.0, 0.0], [20.0, 0.0]])
    rx_pos = np.array([[10.0, 0.0], [30.0, 0.0]])
    
    matrix = channel.get_channel_matrix(
        tx_pos, rx_pos,
        include_shadow=False,
        include_fast_fading=False
    )
    
    assert matrix.shape == (2, 2)
    assert np.all(matrix > 0)
    
    print("  V2VChannel tests passed!")


def test_noma_system():
    """Test NOMASystem class functionality."""
    print("Testing NOMASystem class...")
    
    noma = NOMASystem(
        num_subchannels=4,
        max_users_per_subchannel=2,
        max_tx_power=0.2
    )
    
    assert noma.num_subchannels == 4
    assert len(noma.clusters) == 4
    
    # Create test users
    positions = np.array([
        [0.0, 0.0],
        [20.0, 0.0],
        [40.0, 0.0],
        [60.0, 0.0]
    ])
    
    links = [(0, 1), (1, 2), (2, 3)]
    
    users = noma.create_v2v_links(
        vehicle_positions=positions,
        link_pairs=links,
        rng=np.random.default_rng(42)
    )
    
    assert len(users) == 3
    assert all(u.distance > 0 for u in users)
    
    print("  NOMASystem tests passed!")


def test_resource_allocator():
    """Test ResourceAllocator class functionality."""
    print("Testing ResourceAllocator class...")
    
    noma = NOMASystem(num_subchannels=4, max_users_per_subchannel=2)
    allocator = ResourceAllocator(noma)
    
    # Create test users
    positions = np.array([
        [0.0, 0.0],
        [20.0, 0.0],
        [40.0, 0.0],
        [60.0, 0.0],
        [80.0, 0.0]
    ])
    
    links = [(0, 1), (1, 2), (2, 3), (3, 4)]
    noma.create_v2v_links(positions, links, rng=np.random.default_rng(42))
    
    # Test greedy allocation
    clusters = allocator.allocate_subchannels_greedy()
    assert len(clusters) == 4
    
    # Test joint allocation
    metrics = allocator.joint_allocation(method='greedy', power_method='water_filling')
    
    assert 'sum_rate' in metrics
    assert 'sum_rate_mbps' in metrics
    assert metrics['sum_rate'] >= 0
    assert metrics['num_users'] == 4
    
    print("  ResourceAllocator tests passed!")


def test_platoon_controller():
    """Test PlatoonController class functionality."""
    print("Testing PlatoonController class...")
    
    platoon = Platoon(num_vehicles=5, desired_spacing=20.0, desired_velocity=20.0)
    controller = PlatoonController(platoon)
    
    # Test predecessor following control
    accelerations = controller.predecessor_following_control()
    assert len(accelerations) == 5
    assert accelerations[0] == 0.0  # Leader has no control
    
    # Test control step
    accel, metrics = controller.step(dt=0.1, control_method='pf')
    
    assert 'mean_spacing_error' in metrics
    assert 'string_stable' in metrics
    
    # After step, history should have one entry
    assert len(controller.spacing_history) == 1
    
    print("  PlatoonController tests passed!")


def test_simulation():
    """Test full simulation."""
    print("Testing PlatoonNOMASimulation class...")
    
    config = SimulationConfig(
        num_vehicles=4,
        desired_spacing=20.0,
        desired_velocity=20.0,
        simulation_time=5.0,
        time_step=0.1
    )
    
    sim = PlatoonNOMASimulation(config, random_seed=42)
    
    # Run simulation
    results = sim.run(verbose=False)
    
    assert 'avg_sum_rate_mbps' in results
    assert 'avg_spacing_error' in results
    assert 'string_stable_ratio' in results
    assert np.isclose(results['simulation_time'], 5.0, atol=0.01)
    
    print("  PlatoonNOMASimulation tests passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running all tests for NOMA-V2V Platoon Control")
    print("=" * 50)
    print()
    
    test_vehicle()
    test_platoon()
    test_channel()
    test_noma_system()
    test_resource_allocator()
    test_platoon_controller()
    test_simulation()
    
    print()
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
