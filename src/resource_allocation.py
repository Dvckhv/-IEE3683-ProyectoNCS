"""
Resource Allocation for NOMA-V2V Systems.

This module implements resource allocation algorithms including:
- Subchannel assignment for V2V links
- Power allocation using optimization techniques
- Joint subchannel and power allocation
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy.optimize import minimize, linear_sum_assignment

from .noma import NOMASystem, NOMAUser, NOMACluster


class ResourceAllocator:
    """
    Resource allocator for NOMA-V2V systems.
    
    Implements algorithms for joint subchannel assignment and power allocation
    to maximize sum rate while satisfying QoS constraints.
    """
    
    def __init__(
        self,
        noma_system: NOMASystem,
        alpha: float = 1.0,
        beta: float = 10.0
    ):
        """
        Initialize the resource allocator.
        
        Args:
            noma_system: The NOMA system to allocate resources for
            alpha: Weight for sum rate maximization objective
            beta: Weight for QoS penalty term
        """
        self.noma = noma_system
        self.alpha = alpha
        self.beta = beta
    
    def allocate_subchannels_greedy(self) -> List[NOMACluster]:
        """
        Greedy subchannel allocation based on channel gains.
        
        Assigns users to subchannels to balance load and maximize
        channel gain diversity within clusters (good for NOMA).
        
        Returns:
            List of clusters with assigned users
        """
        # Reset allocation
        self.noma.reset_allocation()
        
        # Sort users by channel gain
        sorted_users = sorted(self.noma.users, key=lambda u: u.channel_gain, reverse=True)
        
        for user in sorted_users:
            # Find best subchannel (least loaded or best pairing)
            best_subchannel = 0
            best_score = float('-inf')
            
            for i, cluster in enumerate(self.noma.clusters):
                if len(cluster.users) >= self.noma.max_users_per_subchannel:
                    continue
                
                # Score based on load balancing and channel gain diversity
                load_score = -len(cluster.users)
                
                # Channel diversity score (prefer pairing strong with weak)
                if len(cluster.users) > 0:
                    existing_gains = [u.channel_gain for u in cluster.users]
                    diversity_score = abs(user.channel_gain - np.mean(existing_gains))
                else:
                    diversity_score = 0
                
                score = load_score + 0.1 * diversity_score
                
                if score > best_score:
                    best_score = score
                    best_subchannel = i
            
            # Assign user to subchannel
            self.noma.clusters[best_subchannel].add_user(user)
        
        # Sort users in each cluster by channel gain for SIC
        for cluster in self.noma.clusters:
            cluster.sort_by_channel_gain()
        
        return self.noma.clusters
    
    def allocate_subchannels_hungarian(self) -> List[NOMACluster]:
        """
        Optimal subchannel allocation using Hungarian algorithm.
        
        Creates an assignment that maximizes total channel gain while
        respecting subchannel capacity constraints.
        
        Returns:
            List of clusters with assigned users
        """
        self.noma.reset_allocation()
        
        num_users = len(self.noma.users)
        num_slots = self.noma.num_subchannels * self.noma.max_users_per_subchannel
        
        if num_users == 0:
            return self.noma.clusters
        
        # Create cost matrix (negative gain for minimization)
        cost_matrix = np.zeros((num_users, num_slots))
        
        for i, user in enumerate(self.noma.users):
            for j in range(num_slots):
                subchannel = j // self.noma.max_users_per_subchannel
                # Use negative channel gain as cost (we minimize)
                cost_matrix[i, j] = -user.channel_gain
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Assign users based on solution
        for user_idx, slot_idx in zip(row_ind, col_ind):
            subchannel = slot_idx // self.noma.max_users_per_subchannel
            self.noma.clusters[subchannel].add_user(self.noma.users[user_idx])
        
        # Sort users in each cluster
        for cluster in self.noma.clusters:
            cluster.sort_by_channel_gain()
        
        return self.noma.clusters
    
    def allocate_power_water_filling(
        self,
        cluster: NOMACluster,
        total_power: float
    ) -> np.ndarray:
        """
        Water-filling power allocation for a NOMA cluster.
        
        Allocates power to maximize sum rate in the cluster while
        ensuring users with worse channels get more power (NOMA principle).
        
        Args:
            cluster: NOMA cluster
            total_power: Total power budget for the cluster
            
        Returns:
            Power allocation vector
        """
        n_users = len(cluster.users)
        if n_users == 0:
            return np.array([])
        
        if n_users == 1:
            return np.array([total_power])
        
        # In NOMA, allocate more power to users with worse channel conditions
        # Inverse channel gain weighting
        channel_gains = np.array([u.channel_gain for u in cluster.users])
        
        # Avoid division by zero
        channel_gains = np.maximum(channel_gains, 1e-10)
        
        # Inverse proportional allocation (more power to weaker channels)
        weights = 1.0 / channel_gains
        weights = weights / np.sum(weights)
        
        power_allocation = weights * total_power
        
        return power_allocation
    
    def allocate_power_fixed_ratio(
        self,
        cluster: NOMACluster,
        total_power: float,
        power_ratio: float = 0.8
    ) -> np.ndarray:
        """
        Fixed power ratio allocation for NOMA.
        
        Uses a fixed ratio to split power between strong and weak users.
        In typical NOMA, the weaker user gets more power.
        
        Args:
            cluster: NOMA cluster
            total_power: Total power budget
            power_ratio: Ratio of power for weaker users
            
        Returns:
            Power allocation vector
        """
        n_users = len(cluster.users)
        if n_users == 0:
            return np.array([])
        
        if n_users == 1:
            return np.array([total_power])
        
        # Allocate power: weak users get power_ratio, strong users get rest
        power_allocation = np.zeros(n_users)
        
        # Users are sorted by channel gain (strongest first)
        remaining_power = total_power
        for i in range(n_users - 1, -1, -1):
            if i == 0:
                power_allocation[i] = remaining_power
            else:
                power_allocation[i] = remaining_power * power_ratio
                remaining_power *= (1 - power_ratio)
        
        return power_allocation
    
    def allocate_power_qos_aware(
        self,
        cluster: NOMACluster,
        total_power: float,
        min_sinr: float = 1.0
    ) -> Tuple[np.ndarray, bool]:
        """
        QoS-aware power allocation ensuring minimum SINR requirements.
        
        Args:
            cluster: NOMA cluster
            total_power: Total power budget
            min_sinr: Minimum SINR requirement (linear)
            
        Returns:
            Tuple of (power_allocation, feasible)
        """
        n_users = len(cluster.users)
        if n_users == 0:
            return np.array([]), True
        
        if n_users == 1:
            user = cluster.users[0]
            required_power = min_sinr * self.noma.noise_power / user.channel_gain
            if required_power <= total_power:
                return np.array([total_power]), True
            else:
                return np.array([total_power]), False
        
        # Iterative allocation starting from weakest user
        power_allocation = np.zeros(n_users)
        noise = self.noma.noise_power
        
        # Start with weakest user (highest index in sorted list)
        remaining_power = total_power
        feasible = True
        
        for i in range(n_users - 1, -1, -1):
            user = cluster.users[i]
            
            # Calculate interference from already allocated users
            interference_from_stronger = 0.0
            for j in range(i):
                interference_from_stronger += power_allocation[j] * user.channel_gain
            
            # Required power for minimum SINR
            required_power = min_sinr * (noise + interference_from_stronger) / user.channel_gain
            
            if required_power > remaining_power:
                feasible = False
                power_allocation[i] = remaining_power
                remaining_power = 0
            else:
                power_allocation[i] = required_power
                remaining_power -= required_power
        
        # Distribute remaining power to strong user if feasible
        if remaining_power > 0:
            power_allocation[0] += remaining_power
        
        return power_allocation, feasible
    
    def joint_allocation(
        self,
        method: str = 'greedy',
        power_method: str = 'water_filling'
    ) -> dict:
        """
        Perform joint subchannel and power allocation.
        
        Args:
            method: Subchannel allocation method ('greedy' or 'hungarian')
            power_method: Power allocation method ('water_filling', 'fixed_ratio', 'qos_aware')
            
        Returns:
            Dictionary with allocation results
        """
        # Step 1: Subchannel allocation
        if method == 'hungarian':
            self.allocate_subchannels_hungarian()
        else:
            self.allocate_subchannels_greedy()
        
        # Step 2: Power allocation per cluster
        total_power_per_cluster = self.noma.max_tx_power / self.noma.num_subchannels
        
        for cluster in self.noma.clusters:
            if len(cluster.users) == 0:
                continue
            
            if power_method == 'water_filling':
                power_alloc = self.allocate_power_water_filling(cluster, total_power_per_cluster)
            elif power_method == 'fixed_ratio':
                power_alloc = self.allocate_power_fixed_ratio(cluster, total_power_per_cluster)
            elif power_method == 'qos_aware':
                power_alloc, _ = self.allocate_power_qos_aware(cluster, total_power_per_cluster)
            else:
                # Equal allocation
                n = len(cluster.users)
                power_alloc = np.ones(n) * total_power_per_cluster / n
            
            # Update user allocations and calculate rates
            for user, power in zip(cluster.users, power_alloc):
                user.allocated_power = power
                sinr = self.noma.calculate_sinr_noma(user, cluster, power_alloc)
                user.data_rate = self.noma.calculate_data_rate(sinr)
            
            cluster.total_power = np.sum(power_alloc)
        
        return self.noma.get_system_metrics()
    
    def optimize_power_allocation_gradient(
        self,
        max_iterations: int = 100,
        learning_rate: float = 0.01
    ) -> dict:
        """
        Gradient-based power allocation optimization.
        
        Uses gradient descent to maximize sum rate subject to
        power constraints.
        
        Args:
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Optimization results
        """
        # First do subchannel allocation
        self.allocate_subchannels_greedy()
        
        total_power = self.noma.max_tx_power
        power_per_cluster = total_power / self.noma.num_subchannels
        
        best_metrics = None
        
        for cluster in self.noma.clusters:
            if len(cluster.users) == 0:
                continue
            
            n_users = len(cluster.users)
            
            # Initialize power allocation
            power_alloc = np.ones(n_users) * power_per_cluster / n_users
            
            for iteration in range(max_iterations):
                # Calculate current rates and gradients
                rates = self.noma.calculate_cluster_rates(cluster, power_alloc)
                
                # Numerical gradient
                grad = np.zeros(n_users)
                eps = 1e-6
                for i in range(n_users):
                    power_plus = power_alloc.copy()
                    power_plus[i] += eps
                    power_plus = np.clip(power_plus, 1e-10, power_per_cluster)
                    
                    rates_plus = self.noma.calculate_cluster_rates(cluster, power_plus)
                    grad[i] = (np.sum(rates_plus) - np.sum(rates)) / eps
                
                # Update power allocation
                power_alloc = power_alloc + learning_rate * grad
                
                # Project onto constraints
                power_alloc = np.clip(power_alloc, 1e-10, power_per_cluster)
                if np.sum(power_alloc) > power_per_cluster:
                    power_alloc = power_alloc / np.sum(power_alloc) * power_per_cluster
            
            # Update user allocations
            for user, power in zip(cluster.users, power_alloc):
                user.allocated_power = power
                sinr = self.noma.calculate_sinr_noma(user, cluster, power_alloc)
                user.data_rate = self.noma.calculate_data_rate(sinr)
            
            cluster.total_power = np.sum(power_alloc)
        
        return self.noma.get_system_metrics()
