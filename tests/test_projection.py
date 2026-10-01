import numpy as np
from perception.projection import GroundProjector

def test_nadir_center_projection():
    projector = GroundProjector()
    drone_pos = np.array([10.0, -5.0, 2.0])
    w = projector.pixel_to_world(projector.cx, projector.cy, drone_pos)
    np.testing.assert_array_almost_equal(w, [10.0, -5.0, 0.0])

def test_symmetric_field_of_view():
    projector = GroundProjector(img_width=160, img_height=160)
    drone_pos = np.array([0.0, 0.0, 5.0])
    
    w1 = projector.pixel_to_world(projector.cx + 40, projector.cy, drone_pos)
    w2 = projector.pixel_to_world(projector.cx - 40, projector.cy, drone_pos)
    
    assert w1[0] == -w2[0]
    
    w3 = projector.pixel_to_world(projector.cx, projector.cy + 40, drone_pos)
    w4 = projector.pixel_to_world(projector.cx, projector.cy - 40, drone_pos)
    
    assert w3[1] == -w4[1]

def test_altitude_scaling():
    projector = GroundProjector()
    d1 = np.array([0.0, 0.0, 2.0])
    d2 = np.array([0.0, 0.0, 4.0])
    
    w1 = projector.pixel_to_world(projector.cx + 10, projector.cy + 10, d1)
    w2 = projector.pixel_to_world(projector.cx + 10, projector.cy + 10, d2)
    
    np.testing.assert_allclose(w2[:2], w1[:2] * 2.0)

def test_batch_projection():
    projector = GroundProjector()
    drone_pos = np.array([0.0, 0.0, 2.0])
    detections = [
        {"u_center": projector.cx, "v_center": projector.cy, "class_id": 1, "confidence": 0.9}
    ]
    results = projector.batch_project(detections, drone_pos)
    assert len(results) == 1
    np.testing.assert_array_almost_equal(results[0]["world_coords"], [0.0, 0.0, 0.0])
