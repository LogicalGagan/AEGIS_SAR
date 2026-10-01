import numpy as np

class ConsensusAuctionEngine:
    def __init__(self, drone_id, alpha=1.0, beta=2.0):
        self.drone_id = drone_id
        self.alpha = alpha
        self.beta = beta

    def evaluate_task(self, target_position: np.ndarray, current_position: np.ndarray, battery: float) -> float:
        distance = np.linalg.norm(target_position - current_position)
        cost = (self.alpha * distance) - (self.beta * battery)
        return cost

    def handle_peer_failure(self, failed_drone_id: int, abandoned_tasks: list, peer_states: dict) -> dict:
        reassignments = {}
        for task in abandoned_tasks:
            best_drone = None
            best_cost = float('inf')
            
            for d_id, state in peer_states.items():
                if d_id == failed_drone_id:
                    continue
                
                cost = self.evaluate_task(task, state["position"], state["battery"])
                if cost < best_cost:
                    best_cost = cost
                    best_drone = d_id
            
            if best_drone is not None:
                reassignments[task.tobytes()] = best_drone
                
        return reassignments
