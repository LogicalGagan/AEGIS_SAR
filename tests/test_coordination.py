import numpy as np
from core.coordination import ConsensusAuctionEngine

def test_auction_formula():
    engine = ConsensusAuctionEngine(drone_id=0, alpha=1.0, beta=2.0)
    target = np.array([10.0, 0.0, 0.0])
    current = np.array([0.0, 0.0, 0.0])
    battery = 1.0
    
    cost = engine.evaluate_task(target, current, battery)
    # distance = 10, cost = 1 * 10 - 2 * 1 = 8
    assert cost == 8.0

def test_handle_peer_failure():
    engine = ConsensusAuctionEngine(drone_id=0)
    failed_drone_id = 2
    abandoned_tasks = [np.array([10.0, 10.0, 0.0])]
    
    peer_states = {
        0: {"position": np.array([0.0, 0.0, 0.0]), "battery": 1.0},
        1: {"position": np.array([9.0, 9.0, 0.0]), "battery": 0.9},
        2: {"position": np.array([10.0, 10.0, 0.0]), "battery": 0.5}
    }
    
    reassignments = engine.handle_peer_failure(failed_drone_id, abandoned_tasks, peer_states)
    
    # Drone 1 is much closer to task [10, 10, 0] than Drone 0
    assigned_drone = reassignments[abandoned_tasks[0].tobytes()]
    assert assigned_drone == 1
