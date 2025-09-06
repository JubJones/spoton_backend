#!/usr/bin/env python3
"""
SpotOn Backend - Import Validation Script
Validates all critical imports in the backend are working correctly.
"""

import sys
import traceback
from typing import List, Tuple, Dict, Any

def test_domain_entities() -> Tuple[int, int, List[str]]:
    """Test domain entity imports."""
    print('🏗️  Testing Domain Entities')
    print('-' * 30)
    
    entities_to_test = [
        ('app.domains.detection.entities.detection', ['Detection', 'BoundingBox', 'DetectionBatch', 'FrameMetadata', 'DetectionClass']),
        ('app.domains.reid.entities.person_identity', ['PersonIdentity', 'IdentityStatus']),
        ('app.domains.reid.entities.feature_vector', ['FeatureVector', 'FeatureVectorBatch']),
        ('app.domains.reid.entities.track', ['Track', 'TrackStatus']),
        ('app.domains.mapping.entities.coordinate', ['Coordinate', 'CoordinateSystem', 'CoordinateTransformation']),
        ('app.domains.mapping.entities.trajectory', ['Trajectory', 'TrajectoryPoint']),
        ('app.domains.mapping.entities.camera_view', ['CameraView', 'CameraViewManager']),
        ('app.domains.visualization.entities.visual_frame', ['VisualFrame']),
        ('app.domains.visualization.entities.overlay_config', ['OverlayConfig']),
        ('app.domains.visualization.entities.cropped_image', ['CroppedImage']),
        ('app.domains.interaction.entities.focus_state', ['FocusState']),
        ('app.domains.export.entities', ['ExportFormat', 'ExportStatus', 'ExportRequest']),
    ]
    
    success_count = 0
    total_count = len(entities_to_test)
    errors = []
    
    for module_path, classes in entities_to_test:
        try:
            module = __import__(module_path, fromlist=classes)
            missing_classes = []
            for cls_name in classes:
                if not hasattr(module, cls_name):
                    missing_classes.append(cls_name)
            
            if missing_classes:
                error_msg = f"❌ {module_path} - Missing classes: {', '.join(missing_classes)}"
                print(error_msg)
                errors.append(error_msg)
            else:
                print(f'✅ {module_path} - All classes available')
                success_count += 1
                
        except ImportError as e:
            error_msg = f"❌ {module_path} - ImportError: {e}"
            print(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"⚠️  {module_path} - Error: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    return success_count, total_count, errors

def test_infrastructure() -> Tuple[int, int, List[str]]:
    """Test infrastructure imports."""
    print('\n🔧 Testing Infrastructure')
    print('-' * 30)
    
    infrastructure_modules = [
        'app.infrastructure.auth.models',
        'app.infrastructure.cache.tracking_cache', 
        'app.infrastructure.database.base',
        'app.infrastructure.gpu.gpu_manager',
    ]
    
    success_count = 0
    total_count = len(infrastructure_modules)
    errors = []
    
    for module_path in infrastructure_modules:
        try:
            __import__(module_path)
            print(f'✅ {module_path}')
            success_count += 1
        except ImportError as e:
            error_msg = f"❌ {module_path} - ImportError: {e}"
            print(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"⚠️  {module_path} - Error: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    return success_count, total_count, errors

def test_services() -> Tuple[int, int, List[str]]:
    """Test service imports that require dependencies."""
    print('\n⚙️  Testing Services (May require dependencies)')
    print('-' * 50)
    
    service_modules = [
        'app.infrastructure.security.jwt_service',
        'app.services.video_data_manager_service',
        'app.api.v1.endpoints.export',
        'app.orchestration.pipeline_orchestrator',
    ]
    
    success_count = 0
    total_count = len(service_modules)
    errors = []
    
    for module_path in service_modules:
        try:
            __import__(module_path)
            print(f'✅ {module_path}')
            success_count += 1
        except ImportError as e:
            error_msg = f"❌ {module_path} - ImportError: {e}"
            print(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"⚠️  {module_path} - Error: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    return success_count, total_count, errors

def test_main_app() -> bool:
    """Test main FastAPI app import."""
    print('\n🚀 Testing Main Application')
    print('-' * 30)
    
    try:
        from app.main import app
        print('✅ FastAPI app imports successfully')
        return True
    except ImportError as e:
        print(f'❌ FastAPI app - ImportError: {e}')
        return False
    except Exception as e:
        print(f'⚠️  FastAPI app - Error: {e}')
        return False

def provide_solutions(errors: List[str]) -> None:
    """Provide solutions for common import errors."""
    print('\n💡 Solutions for Import Issues')
    print('=' * 40)
    
    missing_deps = set()
    for error in errors:
        if 'redis' in error.lower():
            missing_deps.add('redis')
        if 'jwt' in error.lower():
            missing_deps.add('PyJWT')
        if 'psycopg2' in error.lower():
            missing_deps.add('psycopg2-binary')
        if 'boxmot' in error.lower():
            missing_deps.add('boxmot')
        if 'aioredis' in error.lower():
            missing_deps.add('aioredis')
    
    if missing_deps:
        print(f"🔧 Install missing dependencies:")
        print(f"   uv pip install {' '.join(missing_deps)}")
        print(f"   # OR install all with: uv pip install \".[dev]\"")
    
    if any('No module named' in error for error in errors):
        print(f"🐳 Alternative: Use Docker (no dependency management needed):")
        print(f"   docker-compose -f docker-compose.cpu.yml up --build -d")

def main():
    """Main validation function."""
    print('🔍 SpotOn Backend - Comprehensive Import Validation')
    print('=' * 60)
    
    all_errors = []
    total_success = 0
    total_modules = 0
    
    # Test domain entities (should always work)
    entity_success, entity_total, entity_errors = test_domain_entities()
    all_errors.extend(entity_errors)
    total_success += entity_success
    total_modules += entity_total
    
    # Test infrastructure (might have some dependency issues)
    infra_success, infra_total, infra_errors = test_infrastructure()
    all_errors.extend(infra_errors)
    total_success += infra_success
    total_modules += infra_total
    
    # Test services (will likely have dependency issues)
    service_success, service_total, service_errors = test_services()
    all_errors.extend(service_errors)
    total_success += service_success
    total_modules += service_total
    
    # Test main app
    app_success = test_main_app()
    if app_success:
        total_success += 1
    total_modules += 1
    
    # Final results
    print(f'\n📊 Final Results')
    print('=' * 20)
    print(f'✅ Successful imports: {total_success}/{total_modules}')
    print(f'❌ Failed imports: {len(all_errors)}')
    
    if all_errors:
        print(f'\n❌ Issues Found:')
        for error in all_errors[:5]:  # Show first 5 errors
            print(f'   • {error}')
        if len(all_errors) > 5:
            print(f'   ... and {len(all_errors) - 5} more')
        
        provide_solutions(all_errors)
    else:
        print(f'\n🎉 All imports working correctly!')
        print(f'✅ Backend is ready to run!')
    
    # Return exit code
    return 0 if total_success == total_modules else 1

if __name__ == "__main__":
    sys.exit(main())