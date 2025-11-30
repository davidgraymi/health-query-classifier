"""
CLI Interface for Reason Classification

This provides a command-line interface for testing and using the
reason classifier system with real healthcare data.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def classify_single_query(query: str, data_file: str = None):
    """Classify a single healthcare reason query and display results."""
    
    print(f"Query: {query}")
    print("-" * 50)
    
    try:
        # For now, simulate classification since we need the full model setup
        # This uses the same logic as the training script
        
        def map_reason_to_category(reason: str) -> tuple:
            """Map reason to category"""
            reason_lower = reason.lower()
            
            categories = {
                0: "ROUTINE_CARE",
                1: "PAIN_CONDITIONS", 
                2: "INJURIES",
                3: "SKIN_CONDITIONS",
                4: "STRUCTURAL_ISSUES",
                5: "PROCEDURES"
            }
            
            category_descriptions = {
                "ROUTINE_CARE": "Routine foot care, nail care, general maintenance",
                "PAIN_CONDITIONS": "Heel pain, ankle pain, foot pain, toe pain",
                "INJURIES": "Ankle sprains, wounds, trauma-related conditions",
                "SKIN_CONDITIONS": "Ingrown toenails, calluses, skin-related issues",
                "STRUCTURAL_ISSUES": "Flat feet, plantar fasciitis, Achilles tendon problems",
                "PROCEDURES": "Injections, surgical consultations, post-operative care"
            }
            
            if any(word in reason_lower for word in ['routine', 'nail care', 'calluses', 'maintenance']):
                return 0, categories[0], category_descriptions[categories[0]]
            elif any(word in reason_lower for word in ['pain', 'ache', 'sore', 'hurt']):
                return 1, categories[1], category_descriptions[categories[1]]
            elif any(word in reason_lower for word in ['sprain', 'wound', 'injury', 'trauma', 'cut', 'bruise']):
                return 2, categories[2], category_descriptions[categories[2]]
            elif any(word in reason_lower for word in ['ingrown', 'toenail', 'callus', 'corn', 'skin']):
                return 3, categories[3], category_descriptions[categories[3]]
            elif any(word in reason_lower for word in ['flat feet', 'plantar', 'fasciitis', 'achilles', 'tendon', 'arch']):
                return 4, categories[4], category_descriptions[categories[4]]
            elif any(word in reason_lower for word in ['injection', 'surgical', 'consult', 'postop', 'surgery', 'procedure']):
                return 5, categories[5], category_descriptions[categories[5]]
            else:
                return 1, categories[1], category_descriptions[categories[1]]  # Default to pain conditions
        
        # Classify the query
        category_id, category_name, description = map_reason_to_category(query)
        
        # Simulate confidence (in real implementation, this would come from the model)
        import random
        random.seed(hash(query) % 1000)  # Consistent results for same query
        
        # Higher confidence for clear matches
        base_confidence = 0.85 if any(word in query.lower() for word in [
            'pain', 'routine', 'injection', 'sprain', 'ingrown', 'plantar'
        ]) else 0.70
        
        confidence = min(0.95, base_confidence + random.uniform(-0.1, 0.1))
        
        print(f"Primary Classification: {category_name}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Description: {description}")
        
        # Show all category probabilities (simulated)
        print(f"\nAll Category Probabilities:")
        categories = ["ROUTINE_CARE", "PAIN_CONDITIONS", "INJURIES", 
                     "SKIN_CONDITIONS", "STRUCTURAL_ISSUES", "PROCEDURES"]
        
        probabilities = [0.1] * 6  # Base probability
        probabilities[category_id] = confidence  # Predicted category gets high probability
        
        # Normalize
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]
        
        # Sort by probability
        category_probs = list(zip(categories, probabilities))
        category_probs.sort(key=lambda x: x[1], reverse=True)
        
        for cat, prob in category_probs:
            print(f"  {cat}: {prob:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
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
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: {query}")
            
            # Use the same classification logic
            def map_reason_to_category(reason: str) -> tuple:
                reason_lower = reason.lower()
                categories = {
                    0: "ROUTINE_CARE", 1: "PAIN_CONDITIONS", 2: "INJURIES",
                    3: "SKIN_CONDITIONS", 4: "STRUCTURAL_ISSUES", 5: "PROCEDURES"
                }
                
                if any(word in reason_lower for word in ['routine', 'nail care', 'calluses']):
                    return 0, categories[0]
                elif any(word in reason_lower for word in ['pain', 'ache', 'sore']):
                    return 1, categories[1]
                elif any(word in reason_lower for word in ['sprain', 'wound', 'injury']):
                    return 2, categories[2]
                elif any(word in reason_lower for word in ['ingrown', 'toenail', 'callus']):
                    return 3, categories[3]
                elif any(word in reason_lower for word in ['flat feet', 'plantar', 'fasciitis', 'achilles']):
                    return 4, categories[4]
                elif any(word in reason_lower for word in ['injection', 'surgical', 'consult', 'postop']):
                    return 5, categories[5]
                else:
                    return 1, categories[1]
            
            category_id, category_name = map_reason_to_category(query)
            
            # Simulate confidence
            import random
            random.seed(hash(query) % 1000)
            confidence = random.uniform(0.75, 0.95)
            
            result = {
                'query': query,
                'category': category_name,
                'confidence': confidence
            }
            results.append(result)
            
            print(f"   Category: {category_name} (confidence: {confidence:.3f})")
        
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
  python cli/reason_classifier_cli.py "I have heel pain"
  
  # Batch process queries from file
  python cli/reason_classifier_cli.py --batch reason_queries.txt --output results.json
  
  # Interactive mode
  python cli/reason_classifier_cli.py --interactive
        """
    )
    
    parser.add_argument('query', nargs='?', help='Healthcare reason query to classify')
    parser.add_argument('--batch', type=str, help='File containing queries to process')
    parser.add_argument('--output', type=str, help='Output file for batch results')
    parser.add_argument('--interactive', action='store_true', 
                       help='Start interactive mode')
    parser.add_argument('--data', type=str, help='Path to healthcare data file')
    
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
        success = classify_single_query(args.query, args.data)
        return 0 if success else 1
    
    # No arguments provided
    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())