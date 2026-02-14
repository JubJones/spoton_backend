
import asyncio
import numpy as np
import logging
import sys
import os

# Setup simple logging to stdout
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Mock settings before importing app modules if needed
# But we already updated config.py, so we can just use it.

async def verify_osnet_implementation():
    print("🚀 Starting OSNet-AIN Verification...")
    
    try:
        from app.services.feature_extraction_service import FeatureExtractionService
        from app.services.handoff_manager import HandoffManager
        
        # 1. Initialize FeatureExtractionService
        print("\n[1/3] Initializing FeatureExtractionService (OSNet-AIN)...")
        fe_service = FeatureExtractionService()
        print("✅ Service initialized.")
        
        # 2. Extract Features from Dummy Image
        print("\n[2/3] Extracting features from dummy image...")
        # Create a dummy image (128x64x3, uint8, BGR)
        dummy_patch = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        
        embedding = fe_service.extract(dummy_patch)
        
        if embedding is None:
            print("❌ Feature extraction failed (returned None).")
            return
            
        print(f"✅ Extraction successful. Embedding shape: {embedding.shape}")
        if embedding.shape[0] != 512:
            print(f"❌ Error: Expected 512-dim embedding, got {embedding.shape[0]}")
            return
            
        # 3. Test HandoffManager (FAISS)
        print("\n[3/3] Testing HandoffManager (FAISS)...")
        handoff_manager = HandoffManager()
        
        # Register the embedding
        global_id = "person_001"
        camera_id = "c01"
        handoff_manager.register_exit(global_id, embedding, camera_id)
        print(f"✅ Registered {global_id} in FAISS.")
        
        # Search with the SAME embedding (should match perfectly)
        # Note: We need to re-normalize or check if search handles it.
        # HandoffManager handles normalization internally.
        match_id, score = handoff_manager.find_match(embedding, "c02") # Different camera
        
        print(f"Search Result: Match={match_id}, Score={score:.4f}")
        
        if match_id == global_id and score > 0.99:
            print("✅ Perfect match found (Self-match verification).")
        else:
            print(f"❌ Match failed or score too low. Expected {global_id}, got {match_id} with score {score}")
            
        # Search with NOISE
        print("Testing noise match (should fail)...")
        noise_patch = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        noise_emb = fe_service.extract(noise_patch)
        match_id_noise, score_noise = handoff_manager.find_match(noise_emb, "c02")
        print(f"Noise Search Result: Match={match_id_noise}, Score={score_noise:.4f}")
        
        if match_id_noise is None or score_noise < 0.7:
             print("✅ Noise correctly rejected.")
        else:
             print(f"⚠️ Warning: Noise matched with high score: {score_noise:.4f}")
             print(f"Noise Embedding Mean: {np.mean(noise_emb):.4f}, Std: {np.std(noise_emb):.4f}")
             print(f"Target Embedding Mean: {np.mean(embedding):.4f}, Std: {np.std(embedding):.4f}")
             
        # 4. Multi-ID Discrimination Test
        print("\n[4/3] Testing Multi-ID Discrimination...")
        # Create a distinctly different dummy image (e.g., solid color vs random)
        id2_patch = np.zeros((256, 128, 3), dtype=np.uint8) # ID 2
        id2_emb = fe_service.extract(id2_patch)
        handoff_manager.register_exit("person_002", id2_emb, "c01")
        
        # Search for ID 1 (Random Patch)
        match_id_1, score_1 = handoff_manager.find_match(embedding, "c02")
        print(f"Searching for ID 1 (Random Texture): Matched={match_id_1}, Score={score_1:.4f}")
        
        # Search for ID 2 (Black Image)
        match_id_2, score_2 = handoff_manager.find_match(id2_emb, "c02")
        print(f"Searching for ID 2 (Solid Black): Matched={match_id_2}, Score={score_2:.4f}")
        
        if match_id_1 == "person_001" and match_id_2 == "person_002":
             print("✅ Correctly discriminated between two synthetic identities.")
        else:
             print("❌ Failed to discriminate between identities.")

    except Exception as e:
        print(f"\n❌ Verification Failed with Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_osnet_implementation())
