"""
CLI Interface for Healthcare Reason Classification

This provides a command-line interface for testing and using the 
healthcare reason classifier system with real healthcare data.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def classify_single_query(query: str):
    """Classify a single healthcare reason query and display results."""
    
    print(f"Query: {query}")
    print("-" * 50)
    
    try:
        # Import the reason inference module
        sys.path.append('classifier')
        from classifier.reason.infer_reason import predict_single_reason
        
        # Get prediction
        result = predict_single_reason(query)
        
        print(f"Primary Classification: {result['category']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        # Show all category probabilities
        print(f"\nAll Category Probabilities:")
        
        # Sort by probability
        sorted_probs = sorted(result['probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        for category, prob in sorted_probs:
            print(f"  {category}: {prob:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure the reason classifier is trained")
        return False
    
    return True

def classify_batch_queries(queries_file: str, output_file: str = None):
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
        
        print(f"Processing {len(queries)} healthcare reason queries...")
        
        # Import the reason inference module
        sys.path.append('classifier')
        from classifier.reason.infer_reason import predict_single_reason
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: {query}")
            
            result = predict_single_reason(query)
            results.append(result)
            
            print(f"   Category: {result['category']} (confidence: {result['confidence']:.3f})")
        
        # Save results if output file specified
        if output_file:
            output_data = {
                'queries': queries,
                'predictions': results,
                'summary': {
                    'total_queries': len(queries),
                    'categories': {}
                }
            }
            
            # Count categories
            for result in results:
                cat = result['category']
                output_data['summary']['categories'][cat] = output_data['summary']['categories'].get(cat, 0) + 1
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\nResults saved to {output_file}")
        
        # Show summary
        category_counts = {}
        for result in results:
            cat = result['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print(f"\nSummary:")
        for category, count in sorted(category_counts.items()):
            percentage = (count / len(queries)) * 100
            print(f"  {category}: {count} queries ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Error processing batch queries: {e}")
        return False
    
    return True

def interactive_mode():
    """Interactive mode for testing healthcare reason queries."""
    
    print("Healthcare Reason Classifier - Interactive Mode")
    print("=" * 50)
    print("Enter healthcare reason queries to classify (type 'quit' to exit)")
    print()
    print("Example queries to try:")
    print("  • 'I have heel pain when I walk'")
    print("  • 'My toenail is ingrown and infected'")
    print("  • 'I need routine foot care'")
    print("  • 'I sprained my ankle playing basketball'")
    print("  • 'I have plantar fasciitis'")
    print("  • 'I need a cortisone injection'")
    print()
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input:
                classify_single_query(user_input)
                print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            print()

def main():
    parser = argparse.ArgumentParser(
        description='Healthcare Reason Classification CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a single healthcare reason query
  python cli/reason_classifier_cli_new.py "I have heel pain"
  
  # Batch process queries from file
  python cli/reason_classifier_cli_new.py --batch reason_queries.txt --output results.json
  
  # Interactive mode
  python cli/reason_classifier_cli_new.py --interactive
        """
    )
    
    parser.add_argument('query', nargs='?', help='Healthcare reason query to classify')
    parser.add_argument('--batch', type=str, help='File containing queries to process')
    parser.add_argument('--output', type=str, help='Output file for batch results')
    parser.add_argument('--interactive', action='store_true', 
                       help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return 0
    
    # Batch processing
    if args.batch:
        if not Path(args.batch).exists():
            print(f"Error: Batch file does not exist: {args.batch}")
            return 1
        
        success = classify_batch_queries(args.batch, args.output)
        return 0 if success else 1
    
    # Single query processing
    if args.query:
        success = classify_single_query(args.query)
        return 0 if success else 1
    
    # No arguments provided
    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())