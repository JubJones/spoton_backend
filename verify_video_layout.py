"""Verify the new video layout and code changes for sub-video iteration."""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

def check_files():
    """Check that all expected video and GT files exist."""
    errors = []
    
    # Campus sub-videos
    campus_cameras = ["c01", "c02", "c03", "c05"]
    for cam in campus_cameras:
        for idx in range(1, 5):
            path = os.path.join(VIDEOS_DIR, "campus", cam, f"sub_video_{idx:02d}.mp4")
            if os.path.exists(path):
                print(f"  ✅ {os.path.relpath(path, BASE_DIR)}")
            else:
                errors.append(path)
                print(f"  ❌ MISSING: {os.path.relpath(path, BASE_DIR)}")
    
    # Factory single videos
    factory_cameras = ["c09", "c12", "c13", "c16"]
    for cam in factory_cameras:
        path = os.path.join(VIDEOS_DIR, "factory", cam, f"{cam}.mp4")
        if os.path.exists(path):
            print(f"  ✅ {os.path.relpath(path, BASE_DIR)}")
        else:
            errors.append(path)
            print(f"  ❌ MISSING: {os.path.relpath(path, BASE_DIR)}")
    
    # Campus GT files
    for cam in campus_cameras:
        path = os.path.join(VIDEOS_DIR, "campus", "gt", f"gt_{cam}.txt")
        if os.path.exists(path):
            print(f"  ✅ {os.path.relpath(path, BASE_DIR)}")
        else:
            errors.append(path)
            print(f"  ❌ MISSING: {os.path.relpath(path, BASE_DIR)}")
    
    # Factory GT files
    for cam in factory_cameras:
        path = os.path.join(VIDEOS_DIR, "factory", "gt", f"gt_{cam}.txt")
        if os.path.exists(path):
            print(f"  ✅ {os.path.relpath(path, BASE_DIR)}")
        else:
            errors.append(path)
            print(f"  ❌ MISSING: {os.path.relpath(path, BASE_DIR)}")
    
    # Old paths should NOT exist
    old_paths = [
        os.path.join(VIDEOS_DIR, "c09.mp4"),
        os.path.join(VIDEOS_DIR, "c12.mp4"),
        os.path.join(VIDEOS_DIR, "c13.mp4"),
        os.path.join(VIDEOS_DIR, "c16.mp4"),
        os.path.join(VIDEOS_DIR, "gt"),
    ]
    for old in old_paths:
        if os.path.exists(old):
            errors.append(f"OLD path still exists: {old}")
            print(f"  ❌ OLD PATH STILL EXISTS: {os.path.relpath(old, BASE_DIR)}")
        else:
            print(f"  ✅ Old path removed: {os.path.relpath(old, BASE_DIR)}")
    
    return errors


def check_config():
    """Verify config settings are correct."""
    errors = []
    sys.path.insert(0, BASE_DIR)
    os.environ.setdefault("DATASET_ROOT", "")
    
    try:
        from app.core.config import settings
        
        # Check factory cameras have num_sub_videos=1
        env_templates = settings.ENVIRONMENT_TEMPLATES
        factory_cameras = env_templates.get("factory", {}).get("cameras", {})
        for cam_id, cam_cfg in factory_cameras.items():
            nsv = cam_cfg.get("num_sub_videos")
            rbk = cam_cfg.get("remote_base_key")
            print(f"  Factory {cam_id}: num_sub_videos={nsv}, remote_base_key={rbk}")
            if nsv != 1:
                errors.append(f"Factory {cam_id} num_sub_videos should be 1, got {nsv}")
            if not rbk.startswith("factory/"):
                errors.append(f"Factory {cam_id} remote_base_key should start with 'factory/', got {rbk}")
        
        # Check campus cameras have num_sub_videos=4
        campus_cameras = env_templates.get("campus", {}).get("cameras", {})
        for cam_id, cam_cfg in campus_cameras.items():
            nsv = cam_cfg.get("num_sub_videos")
            rbk = cam_cfg.get("remote_base_key")
            print(f"  Campus {cam_id}: num_sub_videos={nsv}, remote_base_key={rbk}")
            if nsv != 4:
                errors.append(f"Campus {cam_id} num_sub_videos should be 4, got {nsv}")
            if not rbk.startswith("campus/"):
                errors.append(f"Campus {cam_id} remote_base_key should start with 'campus/', got {rbk}")
        
        # Check DATASET_ROOT
        dr = settings.DATASET_ROOT
        print(f"  DATASET_ROOT: '{dr}' (should be empty for auto-derivation)")
        if dr and dr != "":
            errors.append(f"DATASET_ROOT should be empty, got '{dr}'")
        
    except Exception as e:
        errors.append(f"Config import failed: {e}")
        print(f"  ❌ Config import error: {e}")
    
    return errors


def check_video_data_manager():
    """Verify video data manager resolves correct paths."""
    errors = []
    sys.path.insert(0, BASE_DIR)
    
    try:
        from app.services.video_data_manager_service import VideoDataManagerService
        from app.utils.asset_downloader import AssetDownloader
        
        ad = AssetDownloader(
            s3_endpoint_url="http://localhost:9000",
            aws_access_key_id="dummy",
            aws_secret_access_key="dummy",
            s3_bucket_name="dummy"
        )
        vdm = VideoDataManagerService(asset_downloader=ad)
        
        # Check max sub-videos
        campus_max = vdm.get_max_sub_videos_for_environment("campus")
        factory_max = vdm.get_max_sub_videos_for_environment("factory")
        print(f"  Campus max_sub_videos: {campus_max} (expected 4)")
        print(f"  Factory max_sub_videos: {factory_max} (expected 1)")
        if campus_max != 4:
            errors.append(f"Campus max_sub_videos should be 4, got {campus_max}")
        if factory_max != 1:
            errors.append(f"Factory max_sub_videos should be 1, got {factory_max}")
        
    except Exception as e:
        errors.append(f"VideoDataManager check failed: {e}")
        print(f"  ❌ VideoDataManager error: {e}")
    
    return errors


if __name__ == "__main__":
    print("\n=== 1. Checking File Layout ===")
    file_errors = check_files()
    
    print("\n=== 2. Checking Config ===")
    config_errors = check_config()
    
    print("\n=== 3. Checking VideoDataManager ===")
    vdm_errors = check_video_data_manager()
    
    all_errors = file_errors + config_errors + vdm_errors
    print(f"\n{'='*50}")
    if all_errors:
        print(f"❌ {len(all_errors)} error(s) found:")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("✅ All checks passed!")
        sys.exit(0)
