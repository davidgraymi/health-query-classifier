"""
CLI Interface for Administrative Query Classification

This provides a command-line interface for testing and using the 
administrative query classifier system.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifier.admin_classifier import AdminQueryClassifier
from classifier.query_router import HealthcareQueryRouter

def classify_single_query(query: str, model_path: str = None):
    """Classify a single query and display results."""
    
    print(f"Query: {query}")
    print("-" * 50)
    
    try:
        classifier = AdminQueryClassifier(model_path)
        predictions = classifier.predict([query])
        
        result = predictions[0]
        
        print(f"Primary Classification: {result['category']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nAll Probabilities:")
        
        # Sort probabilities by value
        sorted_probs = sorted(result['probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        for category, prob in sorted_probs:
            print(f"  {category}: {prob:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

def classify_batch_queries(queries_file: str, model_path: str = None, output_file: str = None):
    """Classify multiple queries from a file."""
    
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
        
        print(f"Processing {len(queries)} queries...")
        
        classifier = AdminQueryClassifier(model_path)
        predictions = classifier.predict(queries)
        
        # Display results
        for i, (query, pred) in enumerate(zip(queries, predictions), 1):
            print(f"\n{i}. Query: {query}")
            print(f"   Category: {pred['category']} (confidence: {pred['confidence']:.3f})")
        
        # Save results if output file specified
        if output_file:
            results = {
                'queries': queries,
                'predictions': predictions,
                'summary': {
                    'total_queries': len(queries),
                    'categories': {}
                }
            }
            
            # Count categories
            for pred in predictions:
                cat = pred['category']
                results['summary']['categories'][cat] = results['summary']['categories'].get(cat, 0) + 1
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to {output_file}")
        
        # Show summary
        category_counts = {}
        for pred in predictions:
            cat = pred['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print(f"\nSummary:")
        for category, count in sorted(category_counts.items()):
            percentage = (count / len(queries)) * 100
            print(f"  {category}: {count} queries ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Error processing batch queries: {e}")
        return False
    
    return True

def route_query(query: str, model_path: str = None):
    """Route a query through the complete routing system."""
    
    print(f"Query: {query}")
    print("=" * 60)
    
    try:
        router = HealthcareQueryRouter(admin_model_path=model_path)
        result = router.route_query(query)
        
        print(f"Primary Classification: {result['primary_classification']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        routing = result['routing_decision']
        print(f"\nRouting Decision:")
        print(f"  Department: {routing['department']}")
        print(f"  Priority: {routing['priority']}")
        print(f"  Estimated Response: {routing['estimated_response']}")
        print(f"  Contact Method: {routing['contact_method']}")
        
        if 'severity_level' in routing:
            print(f"  Severity Level: {routing['severity_level']}")
        
        if result['recommendations']:
            print(f"\nRecommendations:")
            for rec in result['recommendations']:
                print(f"  • {rec}")
        
        if result.get('retrieval_results'):
            print(f"\nRelevant Information Found:")
            for i, doc in enumerate(result['retrieval_results'][:3], 1):
                print(f"  {i}. {doc['title']} (score: {doc['score']:.3f})")
                print(f"     {doc['snippet']}")
        
    except Exception as e:
        print(f"Error routing query: {e}")
        return False
    
    return True

def interactive_mode(model_path: str = None):
    """Interactive mode for testing queries."""
    
    print("Administrative Query Classifier - Interactive Mode")
    print("=" * 50)
    print("Enter queries to classify (type 'quit' to exit)")
    print("Commands:")
    print("  'route <query>' - Use full routing system")
    print("  'classify <query>' - Classification only")
    print("  'quit' - Exit")
    print()
    
    try:
        classifier = AdminQueryClassifier(model_path)
        router = HealthcareQueryRouter(admin_model_path=model_path)
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                if user_input.startswith('route '):
                    query = user_input[6:]
                    route_query(query, model_path)
                elif user_input.startswith('classify '):
                    query = user_input[9:]
                    classify_single_query(query, model_path)
                else:
                    # Default to classification
                    classify_single_query(user_input, model_path)
                
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                print()
    
    except Exception as e:
        print(f"Failed to initialize: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Administrative Query Classification CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a single query
  python cli/admin_classifier_cli.py "I need to schedule an appointment"
  
  # Use routing system
  python cli/admin_classifier_cli.py --route "I have chest pain"
  
  # Batch process queries from file
  python cli/admin_classifier_cli.py --batch queries.txt --output results.json
  
  # Interactive mode
  python cli/admin_classifier_cli.py --interactive
  
  # Use custom model
  python cli/admin_classifier_cli.py --model classifier/my_model "billing question"
        """
    )
    
    parser.add_argument('query', nargs='?', help='Query to classify')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--route', action='store_true', 
                       help='Use full routing system instead of classification only')
    parser.add_argument('--batch', type=str, help='File containing queries to process')
    parser.add_argument('--output', type=str, help='Output file for batch results')
    parser.add_argument('--interactive', action='store_true', 
                       help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Validate model path if provided
    if args.model and not Path(args.model).exists():
        print(f"Error: Model path does not exist: {args.model}")
        return 1
    
    # Interactive mode
    if args.interactive:
        interactive_mode(args.model)
        return 0
    
    # Batch processing
    if args.batch:
        if not Path(args.batch).exists():
            print(f"Error: Batch file does not exist: {args.batch}")
            return 1
        
        success = classify_batch_queries(args.batch, args.model, args.output)
        return 0 if success else 1
    
    # Single query processing
    if args.query:
        if args.route:
            success = route_query(args.query, args.model)
        else:
            success = classify_single_query(args.query, args.model)
        
        return 0 if success else 1
    
    # No arguments provided
    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())