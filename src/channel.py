"""
V2V Channel Model for NOMA communications.

This module implements the channel model for Vehicle-to-Vehicle (V2V)
communications, including path loss, fading, and interference modeling.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChannelParameters:
    """
    Parameters for the V2V channel model.
    
    Attributes:
        carrier_frequency: Carrier frequency in Hz (default: 5.9 GHz for V2V)
        bandwidth: Channel bandwidth in Hz
        noise_power_density: Noise power spectral density in dBm/Hz
        path_loss_exponent: Path loss exponent for V2V
        shadowing_std: Standard deviation of shadow fading in dB
        antenna_gain_tx: Transmit antenna gain in dBi
        antenna_gain_rx: Receive antenna gain in dBi
    """
    carrier_frequency: float = 5.9e9  # 5.9 GHz for V2V
    bandwidth: float = 10e6  # 10 MHz
    noise_power_density: float = -174.0  # dBm/Hz (thermal noise)
    path_loss_exponent: float = 3.0  # Typical for urban V2V
    shadowing_std: float = 3.0  # dB
    antenna_gain_tx: float = 3.0  # dBi
    antenna_gain_rx: float = 3.0  # dBi
    reference_distance: float = 1.0  # meters


class V2VChannel:
    """
    V2V channel model implementing path loss, fading, and interference.
    
    Uses a simplified 3GPP-like model for V2V communications with:
    - Distance-dependent path loss
    - Log-normal shadow fading
    - Rayleigh fast fading
    """
    
    def __init__(self, params: Optional[ChannelParameters] = None):
        """
        Initialize the V2V channel model.
        
        Args:
            params: Channel parameters (uses defaults if None)
        """
        self.params = params or ChannelParameters()
        self._calculate_derived_params()
    
    def _calculate_derived_params(self):
        """Calculate derived parameters from base parameters."""
        # Speed of light
        c = 3e8
        
        # Wavelength
        self.wavelength = c / self.params.carrier_frequency
        
        # Free space path loss at reference distance (in dB)
        self.pl_ref = 20 * np.log10(4 * np.pi * self.params.reference_distance / self.wavelength)
        
        # Noise power in dBm (over the full bandwidth)
        self.noise_power = self.params.noise_power_density + 10 * np.log10(self.params.bandwidth)
    
    def calculate_path_loss(self, distance: float) -> float:
        """
        Calculate path loss for a given distance.
        
        Uses log-distance path loss model:
        PL(d) = PL(d0) + 10*n*log10(d/d0) + X_sigma
        
        Args:
            distance: Distance between transmitter and receiver in meters
            
        Returns:
            Path loss in dB
        """
        if distance < self.params.reference_distance:
            distance = self.params.reference_distance
        
        # Log-distance path loss
        pl = self.pl_ref + 10 * self.params.path_loss_exponent * np.log10(
            distance / self.params.reference_distance
        )
        
        return pl
    
    def calculate_shadow_fading(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        Generate shadow fading component.
        
        Args:
            rng: Random number generator (uses default if None)
            
        Returns:
            Shadow fading value in dB
        """
        if rng is None:
            rng = np.random.default_rng()
        
        return rng.normal(0, self.params.shadowing_std)
    
    def calculate_fast_fading(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        Generate Rayleigh fast fading component.
        
        Args:
            rng: Random number generator (uses default if None)
            
        Returns:
            Fast fading power gain (linear, not dB)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Rayleigh fading: |h|^2 is exponentially distributed with mean 1
        return rng.exponential(1.0)
    
    def calculate_channel_gain(
        self,
        distance: float,
        include_shadow: bool = True,
        include_fast_fading: bool = True,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[float, float]:
        """
        Calculate the total channel gain between transmitter and receiver.
        
        Args:
            distance: Distance in meters
            include_shadow: Whether to include shadow fading
            include_fast_fading: Whether to include fast fading
            rng: Random number generator
            
        Returns:
            Tuple of (channel_gain_dB, channel_gain_linear)
        """
        # Base path loss
        path_loss = self.calculate_path_loss(distance)
        
        # Add antenna gains
        total_gain_dB = (
            self.params.antenna_gain_tx + 
            self.params.antenna_gain_rx - 
            path_loss
        )
        
        # Add shadow fading
        if include_shadow:
            total_gain_dB += self.calculate_shadow_fading(rng)
        
        # Convert to linear
        total_gain_linear = 10 ** (total_gain_dB / 10)
        
        # Apply fast fading
        if include_fast_fading:
            fading = self.calculate_fast_fading(rng)
            total_gain_linear *= fading
            total_gain_dB = 10 * np.log10(total_gain_linear)
        
        return total_gain_dB, total_gain_linear
    
    def calculate_snr(
        self,
        tx_power_dBm: float,
        distance: float,
        include_shadow: bool = True,
        include_fast_fading: bool = True,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Calculate Signal-to-Noise Ratio.
        
        Args:
            tx_power_dBm: Transmit power in dBm
            distance: Distance in meters
            include_shadow: Whether to include shadow fading
            include_fast_fading: Whether to include fast fading
            rng: Random number generator
            
        Returns:
            SNR in dB
        """
        channel_gain_dB, _ = self.calculate_channel_gain(
            distance, include_shadow, include_fast_fading, rng
        )
        
        # Received power in dBm
        rx_power_dBm = tx_power_dBm + channel_gain_dB
        
        # SNR in dB
        snr_dB = rx_power_dBm - self.noise_power
        
        return snr_dB
    
    def get_channel_matrix(
        self,
        tx_positions: np.ndarray,
        rx_positions: np.ndarray,
        include_shadow: bool = True,
        include_fast_fading: bool = True,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Calculate channel gain matrix between all TX-RX pairs.
        
        Args:
            tx_positions: Array of transmitter positions (N_tx x 2)
            rx_positions: Array of receiver positions (N_rx x 2)
            include_shadow: Whether to include shadow fading
            include_fast_fading: Whether to include fast fading
            rng: Random number generator
            
        Returns:
            Channel gain matrix (N_tx x N_rx) in linear scale
        """
        n_tx = len(tx_positions)
        n_rx = len(rx_positions)
        
        channel_matrix = np.zeros((n_tx, n_rx))
        
        for i in range(n_tx):
            for j in range(n_rx):
                distance = np.linalg.norm(tx_positions[i] - rx_positions[j])
                _, gain_linear = self.calculate_channel_gain(
                    distance, include_shadow, include_fast_fading, rng
                )
                channel_matrix[i, j] = gain_linear
        
        return channel_matrix
