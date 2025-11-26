"""
NOMA (Non-Orthogonal Multiple Access) System Model for V2V.

This module implements the NOMA system model where multiple V2V links
share the same time-frequency resources using power-domain multiplexing.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .channel import V2VChannel, ChannelParameters


@dataclass
class NOMAUser:
    """
    Represents a NOMA user (V2V link).
    
    Attributes:
        user_id: Unique identifier
        tx_vehicle_id: ID of transmitting vehicle
        rx_vehicle_id: ID of receiving vehicle
        distance: Distance between TX and RX
        channel_gain: Current channel gain (linear)
        allocated_power: Allocated transmit power (linear)
        allocated_subchannel: Assigned subchannel index
        data_rate: Achieved data rate (bits/s)
        qos_rate: Required QoS data rate (bits/s)
    """
    user_id: int
    tx_vehicle_id: int
    rx_vehicle_id: int
    distance: float = 0.0
    channel_gain: float = 1.0
    allocated_power: float = 0.0
    allocated_subchannel: int = -1
    data_rate: float = 0.0
    qos_rate: float = 1e6  # 1 Mbps default requirement


@dataclass
class NOMACluster:
    """
    Represents a NOMA cluster sharing a subchannel.
    
    In NOMA, multiple users can share the same subchannel using
    power-domain multiplexing with SIC (Successive Interference Cancellation).
    
    Attributes:
        cluster_id: Cluster identifier (same as subchannel index)
        users: List of users in this cluster, sorted by channel gain
        total_power: Total power allocated to this cluster
    """
    cluster_id: int
    users: List[NOMAUser] = field(default_factory=list)
    total_power: float = 0.0
    
    def add_user(self, user: NOMAUser):
        """Add a user to the cluster."""
        self.users.append(user)
        user.allocated_subchannel = self.cluster_id
        
    def sort_by_channel_gain(self):
        """Sort users by channel gain (descending for SIC order)."""
        self.users.sort(key=lambda u: u.channel_gain, reverse=True)


class NOMASystem:
    """
    NOMA system model for V2V platoon communications.
    
    Implements power-domain NOMA where multiple V2V links share
    subchannels using Successive Interference Cancellation (SIC).
    
    Attributes:
        num_subchannels: Number of available subchannels
        subchannel_bandwidth: Bandwidth per subchannel in Hz
        max_users_per_subchannel: Maximum users sharing a subchannel
        max_tx_power: Maximum transmission power per vehicle (Watts)
    """
    
    def __init__(
        self,
        num_subchannels: int = 4,
        subchannel_bandwidth: float = 2.5e6,  # 2.5 MHz
        max_users_per_subchannel: int = 2,
        max_tx_power: float = 0.2,  # 200 mW (23 dBm)
        channel_params: Optional[ChannelParameters] = None
    ):
        """
        Initialize the NOMA system.
        
        Args:
            num_subchannels: Number of subchannels for resource allocation
            subchannel_bandwidth: Bandwidth of each subchannel in Hz
            max_users_per_subchannel: Maximum users per NOMA cluster
            max_tx_power: Maximum TX power per vehicle in Watts
            channel_params: V2V channel parameters
        """
        self.num_subchannels = num_subchannels
        self.subchannel_bandwidth = subchannel_bandwidth
        self.max_users_per_subchannel = max_users_per_subchannel
        self.max_tx_power = max_tx_power
        
        # Initialize channel model
        if channel_params is None:
            channel_params = ChannelParameters(bandwidth=subchannel_bandwidth)
        else:
            channel_params.bandwidth = subchannel_bandwidth
        
        self.channel = V2VChannel(channel_params)
        
        # Initialize clusters
        self.clusters: List[NOMACluster] = [
            NOMACluster(cluster_id=i) for i in range(num_subchannels)
        ]
        
        # Users list
        self.users: List[NOMAUser] = []
        
        # Noise power per subchannel (linear)
        self.noise_power = 10 ** (self.channel.noise_power / 10) * 1e-3  # Convert to Watts
    
    def reset_allocation(self):
        """Reset all resource allocations."""
        self.clusters = [
            NOMACluster(cluster_id=i) for i in range(self.num_subchannels)
        ]
        for user in self.users:
            user.allocated_power = 0.0
            user.allocated_subchannel = -1
            user.data_rate = 0.0
    
    def add_user(self, user: NOMAUser):
        """Add a user to the system."""
        self.users.append(user)
    
    def create_v2v_links(
        self,
        vehicle_positions: np.ndarray,
        link_pairs: List[Tuple[int, int]],
        qos_rates: Optional[List[float]] = None,
        rng: Optional[np.random.Generator] = None
    ) -> List[NOMAUser]:
        """
        Create V2V communication links.
        
        Args:
            vehicle_positions: Positions of all vehicles (N x 2)
            link_pairs: List of (tx_id, rx_id) tuples defining V2V links
            qos_rates: QoS rate requirements for each link
            rng: Random number generator
            
        Returns:
            List of created NOMA users
        """
        self.users = []
        
        if qos_rates is None:
            qos_rates = [1e6] * len(link_pairs)  # Default 1 Mbps
        
        for i, ((tx_id, rx_id), qos_rate) in enumerate(zip(link_pairs, qos_rates)):
            distance = np.linalg.norm(vehicle_positions[tx_id] - vehicle_positions[rx_id])
            _, channel_gain = self.channel.calculate_channel_gain(
                distance,
                include_shadow=True,
                include_fast_fading=True,
                rng=rng
            )
            
            user = NOMAUser(
                user_id=i,
                tx_vehicle_id=tx_id,
                rx_vehicle_id=rx_id,
                distance=distance,
                channel_gain=channel_gain,
                qos_rate=qos_rate
            )
            self.users.append(user)
        
        return self.users
    
    def calculate_sinr_noma(
        self,
        user: NOMAUser,
        cluster: NOMACluster,
        power_allocation: np.ndarray
    ) -> float:
        """
        Calculate SINR for a NOMA user with SIC.
        
        In NOMA with SIC, users with stronger channel gains decode and
        cancel signals from weaker users. A user experiences interference
        only from users with higher channel gains (already decoded) that
        cannot be perfectly cancelled, and users with weaker gains.
        
        For simplicity, we assume perfect SIC where a user only sees
        interference from users with better channel conditions.
        
        Args:
            user: The user for which to calculate SINR
            cluster: The NOMA cluster containing the user
            power_allocation: Power allocated to each user in cluster
            
        Returns:
            SINR value (linear)
        """
        # Find user's position in the SIC decoding order
        user_idx = cluster.users.index(user)
        
        # Signal power
        signal_power = power_allocation[user_idx] * user.channel_gain
        
        # Interference from users with worse channel conditions (not yet decoded)
        interference = 0.0
        for j in range(user_idx + 1, len(cluster.users)):
            interference += power_allocation[j] * user.channel_gain
        
        # SINR
        sinr = signal_power / (interference + self.noise_power)
        
        return sinr
    
    def calculate_data_rate(self, sinr: float) -> float:
        """
        Calculate achievable data rate using Shannon capacity.
        
        Args:
            sinr: Signal-to-Interference-plus-Noise Ratio (linear)
            
        Returns:
            Data rate in bits/s
        """
        if sinr <= 0:
            return 0.0
        return self.subchannel_bandwidth * np.log2(1 + sinr)
    
    def calculate_sum_rate(self) -> float:
        """
        Calculate total sum rate of all users.
        
        Returns:
            Sum rate in bits/s
        """
        return sum(user.data_rate for user in self.users)
    
    def calculate_cluster_rates(
        self,
        cluster: NOMACluster,
        power_allocation: np.ndarray
    ) -> np.ndarray:
        """
        Calculate data rates for all users in a cluster.
        
        Args:
            cluster: NOMA cluster
            power_allocation: Power allocation for users in cluster
            
        Returns:
            Array of data rates for each user
        """
        rates = np.zeros(len(cluster.users))
        
        for i, user in enumerate(cluster.users):
            sinr = self.calculate_sinr_noma(user, cluster, power_allocation)
            rates[i] = self.calculate_data_rate(sinr)
        
        return rates
    
    def check_qos_satisfaction(self) -> Tuple[bool, np.ndarray]:
        """
        Check if all users meet their QoS requirements.
        
        Returns:
            Tuple of (all_satisfied, satisfaction_array)
        """
        satisfaction = np.array([
            user.data_rate >= user.qos_rate for user in self.users
        ])
        return np.all(satisfaction), satisfaction
    
    def get_system_metrics(self) -> dict:
        """
        Get comprehensive system metrics.
        
        Returns:
            Dictionary with system performance metrics
        """
        sum_rate = self.calculate_sum_rate()
        qos_satisfied, satisfaction = self.check_qos_satisfaction()
        
        total_power = sum(
            sum(u.allocated_power for u in c.users) 
            for c in self.clusters
        )
        
        return {
            'sum_rate': sum_rate,
            'sum_rate_mbps': sum_rate / 1e6,
            'avg_rate': sum_rate / max(1, len(self.users)),
            'avg_rate_mbps': sum_rate / 1e6 / max(1, len(self.users)),
            'qos_satisfaction_ratio': np.mean(satisfaction) if len(satisfaction) > 0 else 1.0,
            'all_qos_satisfied': qos_satisfied,
            'total_power_watts': total_power,
            'total_power_dbm': 10 * np.log10(max(1e-10, total_power)) + 30,
            'num_users': len(self.users),
            'num_active_clusters': sum(1 for c in self.clusters if len(c.users) > 0)
        }
