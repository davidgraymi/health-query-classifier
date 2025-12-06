"""
Simple script to analyze healthcare reason data processing
"""

import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append('.')

def test_data_loading():
    """Test loading and processing the healthcare reason data"""
    
    print("Testing Healthcare Reason Data Processing")
    print("=" * 40)
    
    # Load the data
    try:
        df = pd.read_excel('data/reason_for_visit_data.xlsx')
        print(f"✅ Successfully loaded {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Analyze the data
    print(f"\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show reason distribution
    print(f"\nTop 10 Reasons for Visit:")
    top_reasons = df['Reason For Visit'].value_counts().head(10)
    for reason, count in top_reasons.items():
        print(f"  {reason}: {count}")
    
    # Test categorization logic
    def map_reason_to_category(reason: str) -> str:
        """Simple categorization logic"""
        reason_lower = reason.lower()
        
        if any(word in reason_lower for word in ['routine', 'nail care', 'calluses']):
            return "ROUTINE_CARE"
        elif any(word in reason_lower for word in ['pain', 'ache', 'sore']):
            return "PAIN_CONDITIONS"
        elif any(word in reason_lower for word in ['sprain', 'wound', 'injury']):
            return "INJURIES"
        elif any(word in reason_lower for word in ['ingrown', 'toenail', 'callus']):
            return "SKIN_CONDITIONS"
        elif any(word in reason_lower for word in ['flat feet', 'plantar', 'fasciitis', 'achilles']):
            return "STRUCTURAL_ISSUES"
        elif any(word in reason_lower for word in ['injection', 'surgical', 'consult', 'postop']):
            return "PROCEDURES"
        else:
            return "PAIN_CONDITIONS"  # Default
    
    # Apply categorization
    df['Category'] = df['Reason For Visit'].apply(map_reason_to_category)
    
    print(f"\nCategory Distribution:")
    category_counts = df['Category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Show examples for each category
    print(f"\nExample reasons by category:")
    for category in category_counts.index:
        examples = df[df['Category'] == category]['Reason For Visit'].head(3).tolist()
        print(f"  {category}:")
        for example in examples:
            print(f"    - {example}")
    
    return True

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n✅ Healthcare reason data analysis completed successfully!")
    else:
        print("\n❌ Healthcare reason data analysis failed!")