import numpy as np

class BatteryModel:
    def __init__(self, initial_capacity_mah=3500.0, nominal_voltage=11.1):
        self.capacity_mah = initial_capacity_mah
        self.current_charge_mah = initial_capacity_mah
        self.nominal_voltage = nominal_voltage
        
        self.p_hover = 10.0
        self.k_vel = 2.0
        
    def step(self, velocity: np.ndarray, dt: float) -> float:
        v_norm_sq = np.sum(velocity**2)
        drain_rate = self.p_hover + self.k_vel * v_norm_sq
        self.current_charge_mah -= drain_rate * dt
        
        if self.current_charge_mah < 0:
            self.current_charge_mah = 0.0
            
        return self.current_charge_mah / self.capacity_mah

class StateVectorManager:
    """Manages the 21-dimensional state vector for a UAV."""
    
    @staticmethod
    def pack(pos, vel, battery, target_pos, peer_offsets, peer_flags, target_confidence, target_class) -> np.ndarray:
        vector = np.zeros(21, dtype=np.float32)
        vector[0:3] = pos
        vector[3:6] = vel
        vector[6] = battery
        vector[7:10] = target_pos
        vector[10:16] = peer_offsets
        vector[16:19] = peer_flags
        vector[19] = target_confidence
        vector[20] = target_class
        return vector

    @staticmethod
    def unpack(vector: np.ndarray) -> dict:
        """Deconstructs vector into named telemetry."""
        if vector.shape != (21,):
            raise ValueError("State vector must be exactly 21 dimensions.")
        return {
            "pos": vector[0:3],
            "vel": vector[3:6],
            "battery": vector[6],
            "target_pos": vector[7:10],
            "peer_offsets": vector[10:16],
            "peer_flags": vector[16:19],
            "target_confidence": vector[19],
            "target_class": vector[20]
        }
