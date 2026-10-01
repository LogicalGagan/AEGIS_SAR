import time
import numpy as np
from core.comms_mesh import SwarmMeshNode

def test_transmission_in_range():
    node = SwarmMeshNode(drone_id=0, comms_radius=10.0, packet_loss_rate=0.0)
    node.position = np.array([0.0, 0.0, 0.0])
    
    packet = {
        "drone_id": 1,
        "timestamp": time.time(),
        "position": np.array([5.0, 0.0, 0.0]),
        "battery": 1.0
    }
    node.receive_packet(packet)
    assert 1 in node.last_received

def test_dropout_out_of_range():
    node = SwarmMeshNode(drone_id=0, comms_radius=10.0, packet_loss_rate=0.0)
    node.position = np.array([0.0, 0.0, 0.0])
    
    packet = {
        "drone_id": 1,
        "timestamp": time.time(),
        "position": np.array([15.0, 0.0, 0.0]),
        "battery": 1.0
    }
    node.receive_packet(packet)
    assert 1 not in node.last_received

def test_dead_man_switch():
    node = SwarmMeshNode(drone_id=0, comms_radius=10.0, packet_loss_rate=0.0)
    node.last_received[1] = time.time() - 2.0
    node.last_received[2] = time.time()
    
    dropouts = node.check_peer_dropouts(timeout_seconds=1.5)
    assert 1 in dropouts
    assert 2 not in dropouts
