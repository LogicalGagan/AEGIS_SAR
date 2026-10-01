import time
import numpy as np
import random

class SwarmMeshNode:
    def __init__(self, drone_id, comms_radius=12.0, packet_loss_rate=0.2):
        self.drone_id = drone_id
        self.comms_radius = comms_radius
        self.packet_loss_rate = packet_loss_rate
        self.battery = 1.0
        self.position = np.zeros(3)
        self.last_received = {}
    
    def broadcast_heartbeat(self) -> dict:
        return {
            "drone_id": self.drone_id,
            "timestamp": time.time(),
            "position": self.position,
            "battery": self.battery
        }
    
    def receive_packet(self, packet: dict):
        sender_pos = packet["position"]
        distance = np.linalg.norm(self.position - sender_pos)
        
        if distance <= self.comms_radius:
            if random.random() > self.packet_loss_rate:
                self.last_received[packet["drone_id"]] = packet["timestamp"]

    def check_peer_dropouts(self, timeout_seconds=1.5) -> list:
        current_time = time.time()
        dropouts = []
        for peer_id, timestamp in self.last_received.items():
            if current_time - timestamp > timeout_seconds:
                dropouts.append(peer_id)
        return dropouts
