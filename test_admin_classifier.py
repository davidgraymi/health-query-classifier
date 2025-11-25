"""
Test Script for Administrative Query Classification System

This script provides a quick test of the administrative query classifier
to validate the implementation works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def test_basic_functionality():
    """Test basic classifier functionality without training."""
    
    print("Testing Administrative Query Classifier - Basic Functionality")
    print("=" * 60)
    
    try:
        from classifier.admin_classifier import AdminQueryClassifier, ADMIN_CATEGORIES
        
        # Test initialization
        print("✓ Successfully imported AdminQueryClassifier")
        
        classifier = AdminQueryClassifier()
        print("✓ Successfully initialized classifier")
        
        # Test synthetic data generation
        print("\nTesting synthetic data generation...")
        dataset = classifier.create_synthetic_dataset(samples_per_class=10)
        print(f"✓ Generated dataset with {len(dataset)} samples")
        
        # Check category distribution
        category_counts = dataset['category'].value_counts()
        print("Category distribution:")
        for category, count in category_counts.items():
            print(f"  {category}: {count} samples")
        
        # Verify all categories are present
        expected_categories = set(ADMIN_CATEGORIES.values())
        actual_categories = set(dataset['category'].unique())
        
        if expected_categories == actual_categories:
            print("✓ All expected categories present in dataset")
        else:
            missing = expected_categories - actual_categories
            print(f"✗ Missing categories: {missing}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in basic functionality test: {e}")
        return False

def test_query_router():
    """Test the query router functionality."""
    
    print("\nTesting Query Router")
    print("=" * 30)
    
    try:
        from classifier.query_router import HealthcareQueryRouter
        
        # Test initialization (without trained model)
        router = HealthcareQueryRouter(use_retrieval=False)
        print("✓ Successfully initialized HealthcareQueryRouter")
        
        # Test routing logic (will use untrained model, but should not crash)
        test_query = "I need to schedule an appointment"
        
        try:
            result = router.route_query(test_query)
            print("✓ Query routing completed without errors")
            print(f"  Query: {test_query}")
            print(f"  Classification: {result['primary_classification']}")
            print(f"  Department: {result['routing_decision']['department']}")
        except Exception as e:
            print(f"✗ Query routing failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in query router test: {e}")
        return False

def test_cli_imports():
    """Test that CLI modules can be imported."""
    
    print("\nTesting CLI Imports")
    print("=" * 20)
    
    try:
        from cli.admin_classifier_cli import classify_single_query
        print("✓ Successfully imported CLI functions")
        return True
        
    except Exception as e:
        print(f"✗ Error importing CLI: {e}")
        return False

def test_training_script():
    """Test that training script can be imported and basic functions work."""
    
    print("\nTesting Training Script")
    print("=" * 25)
    
    try:
        from classifier.train_admin_classifier import create_enhanced_dataset
        
        # Test enhanced dataset creation
        dataset = create_enhanced_dataset(samples_per_class=5, include_variations=True)
        print(f"✓ Enhanced dataset created with {len(dataset)} samples")
        
        # Should have more samples due to variations
        if len(dataset) > 30:  # 6 categories * 5 samples = 30, variations should add more
            print("✓ Dataset variations working correctly")
        else:
            print("⚠ Dataset variations may not be working as expected")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in training script test: {e}")
        return False

def run_integration_demo():
    """Run a small integration demo showing the complete workflow."""
    
    print("\nIntegration Demo")
    print("=" * 20)
    
    try:
        from classifier.admin_classifier import AdminQueryClassifier
        from classifier.query_router import HealthcareQueryRouter
        
        # Create small dataset and train a minimal model
        print("Creating minimal training dataset...")
        classifier = AdminQueryClassifier()
        dataset = classifier.create_synthetic_dataset(samples_per_class=5)
        
        print("Training minimal model (this may take a moment)...")
        # Use very few epochs for quick test
        metrics = classifier.train(dataset, epochs=2)
        
        print("✓ Training completed")
        
        # Test predictions
        test_queries = [
            "I need to schedule an appointment",
            "My insurance claim was denied", 
            "I have chest pain",
            "What are your office hours?"
        ]
        
        print("\nTesting predictions:")
        predictions = classifier.predict(test_queries)
        
        for query, pred in zip(test_queries, predictions):
            print(f"  '{query}' -> {pred['category']} ({pred['confidence']:.3f})")
        
        # Test router
        print("\nTesting complete routing system:")
        router = HealthcareQueryRouter(use_retrieval=False)
        router.admin_classifier = classifier  # Use our trained classifier
        
        for query in test_queries[:2]:  # Test first 2 queries
            result = router.route_query(query)
            print(f"  '{query}' -> {result['routing_decision']['department']}")
        
        print("✓ Integration demo completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Integration demo failed: {e}")
        print("This is expected if dependencies are missing or if running without proper setup")
        return False

def main():
    """Run all tests."""
    
    print("Administrative Query Classifier - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Query Router", test_query_router), 
        ("CLI Imports", test_cli_imports),
        ("Training Script", test_training_script),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Optional integration demo (may fail due to dependencies)
    print(f"\n[DEMO] Integration Demo (Optional)")
    try:
        demo_success = run_integration_demo()
        results.append(("Integration Demo", demo_success))
    except Exception as e:
        print(f"⚠ Demo skipped due to: {e}")
        results.append(("Integration Demo", None))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results:
        if result is True:
            print(f"✓ {test_name}: PASSED")
            passed += 1
        elif result is False:
            print(f"✗ {test_name}: FAILED")
            failed += 1
        else:
            print(f"⚠ {test_name}: SKIPPED")
            skipped += 1
    
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 All core tests passed! The administrative classifier is ready to use.")
        print("\nNext steps:")
        print("1. Train a full model: python classifier/train_admin_classifier.py")
        print("2. Test with CLI: python cli/admin_classifier_cli.py --interactive")
        print("3. Integrate with your healthcare portal system")
    else:
        print(f"\n⚠ {failed} tests failed. Please check the implementation.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)