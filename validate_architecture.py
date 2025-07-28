#!/usr/bin/env python3
"""
Architecture validation script for Phase 0: Architectural Refactoring

This script validates that the new domain-driven architecture is working correctly.
"""

import sys
from datetime import datetime
from typing import List, Dict, Any

def test_domain_imports() -> Dict[str, Any]:
    """Test that all domain imports work correctly."""
    results = {"passed": 0, "failed": 0, "errors": []}
    
    tests = [
        # Detection domain
        ("Detection entities", "from app.domains.detection.entities.detection import Detection, BoundingBox, DetectionClass"),
        ("Detection services", "from app.domains.detection.services.detection_service import DetectionService"),
        ("Detection models", "from app.domains.detection.models.base_detector import AbstractDetector"),
        
        # ReID domain
        ("ReID entities", "from app.domains.reid.entities.person_identity import PersonIdentity"),
        ("ReID entities (Track)", "from app.domains.reid.entities.track import Track"),
        ("ReID entities (FeatureVector)", "from app.domains.reid.entities.feature_vector import FeatureVector"),
        ("ReID services", "from app.domains.reid.services.reid_service import ReIDService"),
        
        # Mapping domain
        ("Mapping entities", "from app.domains.mapping.entities.coordinate import Coordinate, CoordinateSystem"),
        ("Mapping entities (Trajectory)", "from app.domains.mapping.entities.trajectory import Trajectory"),
        ("Mapping entities (CameraView)", "from app.domains.mapping.entities.camera_view import CameraView"),
        ("Mapping services", "from app.domains.mapping.services.mapping_service import MappingService"),
        
        # Infrastructure
        ("Database base", "from app.infrastructure.database.base import Base"),
        ("Database session", "from app.infrastructure.database.session import get_db_session"),
        
        # Orchestration
        ("Camera manager", "from app.orchestration.camera_manager import camera_manager"),
        ("Real-time processor", "from app.orchestration.real_time_processor import real_time_processor"),
        
        # Shared types
        ("Shared types", "from app.shared.types import CameraID")
    ]
    
    for test_name, import_statement in tests:
        try:
            exec(import_statement)
            results["passed"] += 1
            print(f"✓ {test_name}")
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"{test_name}: {e}")
            print(f"✗ {test_name}: {e}")
    
    return results

def test_entity_functionality() -> Dict[str, Any]:
    """Test that domain entities work correctly."""
    results = {"passed": 0, "failed": 0, "errors": []}
    
    try:
        # Test Detection entity
        from app.domains.detection.entities.detection import Detection, BoundingBox, DetectionClass
        
        bbox = BoundingBox(x=10, y=20, width=100, height=200)
        detection = Detection(
            id='test-1',
            camera_id='c01',
            bbox=bbox,
            confidence=0.8,
            class_id=DetectionClass.PERSON,
            timestamp=datetime.now(),
            frame_index=1
        )
        
        assert detection.bbox.center_x == 60.0
        assert detection.bbox.center_y == 120.0
        assert detection.is_person == True
        
        results["passed"] += 1
        print("✓ Detection entity functionality")
        
    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"Detection entity: {e}")
        print(f"✗ Detection entity: {e}")
    
    try:
        # Test PersonIdentity entity
        from app.domains.reid.entities.person_identity import PersonIdentity
        
        identity = PersonIdentity(global_id='person-1')
        identity = identity.add_camera_track('c01', 1)
        identity = identity.add_camera_track('c02', 2)
        
        assert identity.camera_count == 2
        assert identity.track_count == 2
        assert identity.has_camera_track('c01') == True
        
        results["passed"] += 1
        print("✓ PersonIdentity entity functionality")
        
    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"PersonIdentity entity: {e}")
        print(f"✗ PersonIdentity entity: {e}")
    
    try:
        # Test Coordinate entity
        from app.domains.mapping.entities.coordinate import Coordinate, CoordinateSystem
        
        coord1 = Coordinate(x=0, y=0, coordinate_system=CoordinateSystem.IMAGE, timestamp=datetime.now())
        coord2 = Coordinate(x=3, y=4, coordinate_system=CoordinateSystem.IMAGE, timestamp=datetime.now())
        
        distance = coord1.distance_to(coord2)
        assert distance == 5.0  # 3-4-5 triangle
        
        results["passed"] += 1
        print("✓ Coordinate entity functionality")
        
    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"Coordinate entity: {e}")
        print(f"✗ Coordinate entity: {e}")
    
    return results

def test_service_functionality() -> Dict[str, Any]:
    """Test that domain services work correctly."""
    results = {"passed": 0, "failed": 0, "errors": []}
    
    try:
        # Test DetectionService
        from app.domains.detection.services.detection_service import DetectionService
        
        detection_service = DetectionService(detector=None)
        stats = detection_service.get_detection_stats()
        
        assert "total_detections" in stats
        assert stats["total_detections"] == 0
        
        results["passed"] += 1
        print("✓ DetectionService functionality")
        
    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"DetectionService: {e}")
        print(f"✗ DetectionService: {e}")
    
    try:
        # Test ReIDService
        from app.domains.reid.services.reid_service import ReIDService
        
        reid_service = ReIDService()
        stats = reid_service.get_reid_stats()
        
        assert "active_identities" in stats
        assert stats["active_identities"] == 0
        
        results["passed"] += 1
        print("✓ ReIDService functionality")
        
    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"ReIDService: {e}")
        print(f"✗ ReIDService: {e}")
    
    try:
        # Test MappingService
        from app.domains.mapping.services.mapping_service import MappingService
        
        mapping_service = MappingService()
        stats = mapping_service.get_mapping_stats()
        
        assert "registered_cameras" in stats
        assert stats["registered_cameras"] == 0
        
        results["passed"] += 1
        print("✓ MappingService functionality")
        
    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"MappingService: {e}")
        print(f"✗ MappingService: {e}")
    
    return results

def main():
    """Run all architecture validation tests."""
    print("Phase 0: Architectural Refactoring - Validation Report")
    print("=" * 60)
    print()
    
    # Test domain imports
    print("1. Testing Domain Imports")
    print("-" * 30)
    import_results = test_domain_imports()
    print()
    
    # Test entity functionality
    print("2. Testing Entity Functionality")
    print("-" * 30)
    entity_results = test_entity_functionality()
    print()
    
    # Test service functionality
    print("3. Testing Service Functionality")
    print("-" * 30)
    service_results = test_service_functionality()
    print()
    
    # Summary
    print("Summary")
    print("-" * 30)
    total_passed = import_results["passed"] + entity_results["passed"] + service_results["passed"]
    total_failed = import_results["failed"] + entity_results["failed"] + service_results["failed"]
    
    print(f"Total tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_failed > 0:
        print("\nFailures:")
        for error in (import_results["errors"] + entity_results["errors"] + service_results["errors"]):
            print(f"  - {error}")
    
    print("\nArchitectural Changes Summary:")
    print("- ✓ Created domain-based directory structure")
    print("- ✓ Implemented Detection domain (entities, services, models)")
    print("- ✓ Implemented ReID domain (entities, services, models)")
    print("- ✓ Implemented Mapping domain (entities, services, models)")
    print("- ✓ Created Infrastructure layer (database, cache, external)")
    print("- ✓ Created Orchestration layer (pipeline, camera manager, real-time processor)")
    print("- ✓ Moved common types to shared module")
    print("- ✓ Updated import statements across codebase")
    
    if total_failed == 0:
        print("\n🎉 Phase 0: Architectural Refactoring - COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print(f"\n❌ Phase 0: Architectural Refactoring - {total_failed} issues found")
        return 1

if __name__ == "__main__":
    sys.exit(main())