import numpy as np

class GroundProjector:
    def __init__(self, img_width=160, img_height=160, fov_rad=1.0472):
        self.img_width = img_width
        self.img_height = img_height
        self.fov_rad = fov_rad
        
        self.fx = (img_width / 2.0) / np.tan(fov_rad / 2.0)
        self.fy = (img_height / 2.0) / np.tan(fov_rad / 2.0)
        self.cx = img_width / 2.0
        self.cy = img_height / 2.0
        
    def pixel_to_world(self, u: float, v: float, drone_pos: np.ndarray, drone_rpy: np.ndarray = None) -> np.ndarray:
        if drone_pos[2] <= 0.05:
            return np.array([drone_pos[0], drone_pos[1], 0.0], dtype=np.float32)
            
        x_norm = (u - self.cx) / self.fx
        y_norm = (v - self.cy) / self.fy
        
        Z_c = drone_pos[2]
        
        X_w = drone_pos[0] + (x_norm * Z_c)
        Y_w = drone_pos[1] + (y_norm * Z_c)
        
        return np.array([X_w, Y_w, 0.0], dtype=np.float32)
        
    def batch_project(self, detections: list, drone_pos: np.ndarray) -> list:
        results = []
        for det in detections:
            u, v = det["u_center"], det["v_center"]
            world_pos = self.pixel_to_world(u, v, drone_pos)
            
            new_det = det.copy()
            new_det["world_coords"] = world_pos.tolist()
            results.append(new_det)
        return results
