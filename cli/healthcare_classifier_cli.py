"""
End-to-End Healthcare Classification CLI

This provides a complete classification pipeline:
1. First classifies as "medical" or "insurance"
2. If medical, applies reason classification for detailed categorization

IMPORTANT: Activate virtual environment first!
Usage:
  source .venv/bin/activate
  python cli/healthcare_classifier_cli.py --interactive
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def classify_healthcare_query(query: str):
    """
    Complete healthcare query classification pipeline.
    
    Step 1: Medical vs Insurance classification
    Step 2: If medical, apply reason classification
    """
    
    print(f"Query: {query}")
    print("=" * 60)
    
    try:
        # Add classifier to path
        sys.path.append('classifier')
        
        # Step 1: Medical vs Insurance Classification
        print("🔍 Step 1: Medical vs Insurance Classification")
        print("-" * 40)
        
        from infer import predict_query
        from utils import get_models
        
        # Load medical/insurance classifier
        embedding_model, classifier_head = get_models()
        
        # Get medical vs insurance prediction
        result = predict_query([query], embedding_model, classifier_head)
        
        primary_category = result['prediction'][0]
        confidence = result['confidence']
        if isinstance(confidence, list):
            confidence = confidence[0]
        
        print(f"Primary Classification: {primary_category.upper()}")
        print(f"Confidence: {confidence:.4f}")
        
        # Show probabilities
        probabilities = result['probabilities']
        if isinstance(probabilities[0], list):
            probabilities = probabilities[0]
        
        print("Probabilities:")
        from utils import CATEGORIES
        for i, category in enumerate(CATEGORIES):
            print(f"  {category}: {probabilities[i]:.4f}")
        
        # Step 2: If medical, apply reason classification
        if primary_category.lower() == 'medical':
            print(f"\n🏥 Step 2: Medical Reason Classification")
            print("-" * 40)
            
            try:
                from infer_reason import predict_single_reason
                
                reason_result = predict_single_reason(query)
                
                print(f"Medical Reason: {reason_result['category']}")
                print(f"Reason Confidence: {reason_result['confidence']:.4f}")
                
                print("Reason Probabilities:")
                sorted_probs = sorted(reason_result['probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)
                for category, prob in sorted_probs:
                    print(f"  {category}: {prob:.4f}")
                
                # Final routing decision
                print(f"\n🎯 Final Routing Decision")
                print("-" * 25)
                print(f"Route to: {reason_result['category']} Department")
                print(f"Overall confidence: Medical ({confidence:.3f}) → {reason_result['category']} ({reason_result['confidence']:.3f})")
                
                return {
                    'primary_classification': primary_category,
                    'primary_confidence': confidence,
                    'reason_classification': reason_result['category'],
                    'reason_confidence': reason_result['confidence'],
                    'routing': f"{reason_result['category']} Department"
                }
                
            except Exception as e:
                print(f"⚠️  Reason classification failed: {e}")
                print("Note: Make sure reason classifier is trained")
                print(f"Routing to: General Medical Department")
                
                return {
                    'primary_classification': primary_category,
                    'primary_confidence': confidence,
                    'reason_classification': 'GENERAL_MEDICAL',
                    'reason_confidence': 0.0,
                    'routing': 'General Medical Department'
                }
        
        else:
            # Insurance query
            print(f"\n💳 Final Routing Decision")
            print("-" * 25)
            print(f"Route to: Insurance Department")
            print(f"Confidence: {confidence:.3f}")
            
            return {
                'primary_classification': primary_category,
                'primary_confidence': confidence,
                'reason_classification': None,
                'reason_confidence': None,
                'routing': 'Insurance Department'
            }
            
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        if "No module named 'torch'" in str(e):
            print("\n🔧 SOLUTION:")
            print("You need to activate the virtual environment first!")
            print("Run these commands:")
            print("  source .venv/bin/activate")
            print("  python cli/healthcare_classifier_cli.py --interactive")
        else:
            print("Note: Make sure models are trained and available")
        return None

def classify_batch_queries(queries_file: str, output_file: str = None):
    """Process multiple queries through the complete pipeline."""
    
    try:
        # Read queries
        with open(queries_file, 'r') as f:
            if queries_file.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    queries = data
                else:
                    queries = data.get('queries', [])
            else:
                queries = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(queries)} queries through complete pipeline...")
        print("=" * 60)
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n📋 Query {i}/{len(queries)}")
            result = classify_healthcare_query(query)
            if result:
                result['query'] = query
                results.append(result)
            print()
        
        # Save results if output file specified
        if output_file:
            output_data = {
                'queries': queries,
                'predictions': results,
                'summary': {
                    'total_queries': len(queries),
                    'medical_queries': len([r for r in results if r['primary_classification'].lower() == 'medical']),
                    'insurance_queries': len([r for r in results if r['primary_classification'].lower() == 'insurance']),
                    'reason_categories': {}
                }
            }
            
            # Count reason categories
            for result in results:
                if result['reason_classification']:
                    cat = result['reason_classification']
                    output_data['summary']['reason_categories'][cat] = output_data['summary']['reason_categories'].get(cat, 0) + 1
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"📄 Results saved to {output_file}")
        
        # Show summary
        medical_count = len([r for r in results if r['primary_classification'].lower() == 'medical'])
        insurance_count = len([r for r in results if r['primary_classification'].lower() == 'insurance'])
        
        print(f"\n📊 Summary:")
        print(f"  Medical queries: {medical_count} ({medical_count/len(results)*100:.1f}%)")
        print(f"  Insurance queries: {insurance_count} ({insurance_count/len(results)*100:.1f}%)")
        
        if medical_count > 0:
            reason_counts = {}
            for result in results:
                if result['reason_classification']:
                    cat = result['reason_classification']
                    reason_counts[cat] = reason_counts.get(cat, 0) + 1
            
            print(f"\n  Medical reason breakdown:")
            for category, count in sorted(reason_counts.items()):
                percentage = (count / medical_count) * 100
                print(f"    {category}: {count} queries ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"❌ Error processing batch queries: {e}")
        return False
    
    return True

def interactive_mode():
    """Interactive mode for complete healthcare classification."""
    
    print("🏥 Complete Healthcare Classification System")
    print("=" * 50)
    print("This system provides end-to-end classification:")
    print("  1️⃣  Medical vs Insurance classification")
    print("  2️⃣  Medical reason classification (if medical)")
    print("  3️⃣  Final routing decision")
    print()
    print("Enter healthcare queries to classify (type 'quit' to exit)")
    print()
    print("Example queries to try:")
    print("  Medical: 'I have heel pain when I walk'")
    print("  Medical: 'I need routine foot care'") 
    print("  Medical: 'I sprained my ankle'")
    print("  Insurance: 'My insurance claim was denied'")
    print("  Insurance: 'What does my insurance cover?'")
    print()
    
    while True:
        try:
            user_input = input("🔍 Enter query >>> ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if user_input:
                classify_healthcare_query(user_input)
                print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()

def main():
    parser = argparse.ArgumentParser(
        description='Complete Healthcare Classification CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python cli/healthcare_classifier_cli.py --interactive
  
  # Classify a single query
  python cli/healthcare_classifier_cli.py "I have heel pain"
  
  # Batch process queries from file
  python cli/healthcare_classifier_cli.py --batch queries.txt --output results.json

Pipeline:
  Query → Medical/Insurance → (if Medical) → Reason Classification → Routing
        """
    )
    
    parser.add_argument('query', nargs='?', help='Healthcare query to classify')
    parser.add_argument('--batch', type=str, help='File containing queries to process')
    parser.add_argument('--output', type=str, help='Output file for batch results')
    parser.add_argument('--interactive', action='store_true', 
                       help='Start interactive mode (recommended)')
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return 0
    
    # Batch processing
    if args.batch:
        if not Path(args.batch).exists():
            print(f"❌ Error: Batch file does not exist: {args.batch}")
            return 1
        
        success = classify_batch_queries(args.batch, args.output)
        return 0 if success else 1
    
    # Single query processing
    if args.query:
        result = classify_healthcare_query(args.query)
        return 0 if result else 1
    
    # No arguments provided - show help and suggest interactive mode
    print("🏥 Complete Healthcare Classification System")
    print("=" * 45)
    print("IMPORTANT: Activate virtual environment first!")
    print("  source .venv/bin/activate")
    print("  python cli/healthcare_classifier_cli.py --interactive")
    print()
    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())