"""
Joint Platoon Control and Resource Allocation for NOMA-V2V Communications.

This package provides simulation tools for:
- Vehicle platoon modeling and control
- NOMA (Non-Orthogonal Multiple Access) V2V channel modeling
- Resource allocation optimization (power allocation, subchannel assignment)
- Joint platoon control and communication resource optimization
"""

from .vehicle import Vehicle, Platoon
from .channel import V2VChannel
from .noma import NOMASystem
from .resource_allocation import ResourceAllocator
from .platoon_control import PlatoonController
from .simulation import PlatoonNOMASimulation

__all__ = [
    'Vehicle',
    'Platoon',
    'V2VChannel',
    'NOMASystem',
    'ResourceAllocator',
    'PlatoonController',
    'PlatoonNOMASimulation'
]
