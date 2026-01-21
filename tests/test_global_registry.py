import pytest
import uuid
from app.services.global_person_registry import GlobalPersonRegistry

def test_registry_assignment():
    registry = GlobalPersonRegistry()
    
    # Assign new ID
    new_id = registry.allocate_new_id()
    registry.assign_identity("cam_A", 1, new_id)
    
    assert registry.get_global_id("cam_A", 1) == new_id

def test_registry_merge():
    registry = GlobalPersonRegistry()
    
    id_1 = registry.allocate_new_id()
    id_2 = registry.allocate_new_id()
    
    registry.assign_identity("cam_A", 1, id_1)
    registry.assign_identity("cam_B", 2, id_2)
    
    # Merge id_2 into id_1 (e.g. Re-ID match found)
    registry.merge_identities(target_global_id=id_1, source_global_id=id_2)
    
    # Both should now be id_1
    assert registry.get_global_id("cam_A", 1) == id_1
    assert registry.get_global_id("cam_B", 2) == id_1
    
def test_registry_reassignment_triggers_merge():
    registry = GlobalPersonRegistry()
    
    id_1 = "person_1"
    id_2 = "person_2"
    
    registry.assign_identity("cam_A", 1, id_1)
    
    # Now assign track 1 to id_2 (implicit merge)
    registry.assign_identity("cam_A", 1, id_2)
    
    # Should be id_2
    assert registry.get_global_id("cam_A", 1) == id_2
    # Verify merge happened (though here it's simple reassignment, 
    # but if other tracks had id_1 they should move too)
    
    # Setup complex merge
    registry.assign_identity("cam_A", 99, id_1) # Assign another track to old ID
    registry.assign_identity("cam_A", 1, id_2) # Reassign track 1 to id 2
    # The logic says: if existing_id != global_id, merge existing -> global.
    # So id_1 -> id_2.
    # Track 99 should now be id_2
    
    # Let's verify this behavior explicitly
    r2 = GlobalPersonRegistry()
    gid_a = "A"
    gid_b = "B"
    r2.assign_identity("c1", 10, gid_a)
    r2.assign_identity("c1", 11, gid_a)
    
    # Now say c1:10 is actually gid_b
    r2.assign_identity("c1", 10, gid_b)
    
    # This should merge A into B
    assert r2.get_global_id("c1", 10) == gid_b
    assert r2.get_global_id("c1", 11) == gid_b

def test_is_global_id_shared():
    registry = GlobalPersonRegistry()
    gid = "shared_guy"
    
    registry.assign_identity("c1", 1, gid)
    assert not registry.is_global_id_shared(gid)
    
    registry.assign_identity("c2", 1, gid)
    assert registry.is_global_id_shared(gid)
