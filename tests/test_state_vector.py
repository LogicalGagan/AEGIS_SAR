import numpy as np
import pytest
from core.state_vector import StateVectorManager, BatteryModel

def test_shape_and_dtype():
    pos = [1.0, 2.0, 3.0]
    vel = [0.1, 0.2, 0.3]
    battery = 0.95
    target_pos = [10.0, 20.0, 30.0]
    peer_offsets = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    peer_flags = [1.0, 1.0, 0.0]
    target_confidence = 0.8
    target_class = 1.0
    
    vec = StateVectorManager.pack(pos, vel, battery, target_pos, peer_offsets, peer_flags, target_confidence, target_class)
    
    assert vec.shape == (21,)
    assert vec.dtype == np.float32

def test_pack_unpack_invariance():
    pos = [1.0, 2.0, 3.0]
    vel = [0.1, 0.2, 0.3]
    battery = 0.95
    target_pos = [10.0, 20.0, 30.0]
    peer_offsets = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    peer_flags = [1.0, 1.0, 0.0]
    target_confidence = 0.8
    target_class = 1.0
    
    vec = StateVectorManager.pack(pos, vel, battery, target_pos, peer_offsets, peer_flags, target_confidence, target_class)
    unpacked = StateVectorManager.unpack(vec)
    
    np.testing.assert_allclose(unpacked["pos"], pos, atol=1e-4)
    np.testing.assert_allclose(unpacked["vel"], vel, atol=1e-4)
    assert abs(unpacked["battery"] - battery) < 1e-4
    np.testing.assert_allclose(unpacked["target_pos"], target_pos, atol=1e-4)
    np.testing.assert_allclose(unpacked["peer_offsets"], peer_offsets, atol=1e-4)
    np.testing.assert_allclose(unpacked["peer_flags"], peer_flags, atol=1e-4)
    assert abs(unpacked["target_confidence"] - target_confidence) < 1e-4
    assert abs(unpacked["target_class"] - target_class) < 1e-4

def test_battery_depletion():
    batt1 = BatteryModel()
    batt2 = BatteryModel()
    
    hover_ratio = batt1.step(np.array([0.0, 0.0, 0.0]), dt=1.0)
    move_ratio = batt2.step(np.array([5.0, 0.0, 0.0]), dt=1.0)
    
    assert move_ratio < hover_ratio
