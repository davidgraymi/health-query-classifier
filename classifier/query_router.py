"""
Query Router System

This module integrates the medical/insurance classifier with the reason 
classification system to provide intelligent routing of healthcare portal queries.

The router first determines if a query is medical or insurance-related, then 
routes accordingly:
- Insurance queries -> Direct to insurance department
- Medical queries -> Reason classification -> Appropriate medical department routing
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add project root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifier.infer import predict_query
from classifier.utils import get_models, CATEGORIES
from classifier.reason import predict_single_reason
from retriever.search import Retriever
from team.candidates import get_candidates

class HealthcareQueryRouter:
    """
    Intelligent routing system for healthcare portal queries.
    
    Routes queries through a two-stage process:
    1. Medical vs Insurance classification
    2. For medical queries: Reason classification for department routing
    """
    
    def __init__(self, 
                 medical_model_path: Optional[str] = None,
                 use_retrieval: bool = True):
        """
        Initialize the query router.
        
        Args:
            medical_model_path: Path to trained medical/insurance classifier
            use_retrieval: Whether to use retrieval system for medical queries
        """
        
        # Initialize medical/insurance classifier
        try:
            self.embedding_model, self.classifier_head = get_models()
            
            # Load trained model if available
            if medical_model_path and os.path.exists(medical_model_path):
                import torch
                state_dict = torch.load(medical_model_path, weights_only=True)
                self.classifier_head.load_state_dict(state_dict)
                print(f"Loaded medical/insurance classifier from {medical_model_path}")
            else:
                print("Using untrained medical/insurance classifier")
                
        except Exception as e:
            print(f"Error initializing medical/insurance classifier: {e}")
            raise
        
        # Initialize retrieval system if requested
        self.retriever = None
        if use_retrieval:
            try:
                # Use default corpora configuration
                corpora_config = {
                    "medical_qa": {
                        "path": "data/corpora/medical_qa.jsonl",
                        "text_fields": ["question", "answer", "title"],
                    },
                    "miriad": {
                        "path": "data/corpora/miriad_text.jsonl", 
                        "text_fields": ["text", "title"],
                    }
                }
                # Only use available corpora
                available_config = {k: v for k, v in corpora_config.items() 
                                  if Path(v["path"]).exists()}
                
                if available_config:
                    self.retriever = Retriever(available_config)
                    print(f"Retrieval system initialized with {len(available_config)} corpora")
                else:
                    print("No corpora files found. Retrieval disabled.")
            except Exception as e:
                print(f"Could not initialize retrieval system: {e}")
        
        # Routing rules for insurance queries
        self.insurance_routing = {
            "department": "Insurance Department",
            "priority": "normal",
            "estimated_response": "1-2 business days",
            "contact_method": "phone_or_email",
            "description": "Insurance coverage, claims, and benefits inquiries"
        }
        
        # Medical department routing based on reason categories
        self.medical_department_routing = {
            "ROUTINE_CARE": {
                "department": "Primary Care",
                "priority": "normal",
                "estimated_response": "1-7 days",
                "contact_method": "standard_scheduling",
                "description": "Routine healthcare and maintenance visits"
            },
            "PAIN_CONDITIONS": {
                "department": "Pain Management",
                "priority": "high",
                "estimated_response": "same day to 3 days",
                "contact_method": "phone_preferred",
                "description": "Pain-related conditions and discomfort"
            },
            "INJURIES": {
                "department": "Urgent Care",
                "priority": "high",
                "estimated_response": "same day",
                "contact_method": "phone_immediate",
                "description": "Injuries, sprains, and trauma-related conditions"
            },
            "SKIN_CONDITIONS": {
                "department": "Dermatology",
                "priority": "normal",
                "estimated_response": "3-7 days",
                "contact_method": "standard_scheduling",
                "description": "Skin-related issues and conditions"
            },
            "STRUCTURAL_ISSUES": {
                "department": "Orthopedics",
                "priority": "normal",
                "estimated_response": "1-14 days",
                "contact_method": "standard_scheduling",
                "description": "Structural problems and musculoskeletal conditions"
            },
            "PROCEDURES": {
                "department": "Surgical Services",
                "priority": "normal",
                "estimated_response": "3-14 days",
                "contact_method": "scheduling_coordinator",
                "description": "Surgical consultations and procedures"
            }
        }
    
    def route_query(self, query: str, include_retrieval: bool = True) -> Dict:
        """
        Route a healthcare query through the classification and routing system.
        
        Args:
            query: The user's query text
            include_retrieval: Whether to include retrieval results for medical queries
            
        Returns:
            Dictionary with routing decision, confidence, and additional context
        """
        
        # Step 1: Medical vs Insurance classification
        medical_prediction = predict_query([query], self.embedding_model, self.classifier_head)
        
        # Extract prediction details
        primary_category = medical_prediction['prediction'][0]
        confidence = medical_prediction['confidence'] if isinstance(medical_prediction['confidence'], float) else medical_prediction['confidence'][0]
        probabilities = medical_prediction['probabilities']
        
        routing_result = {
            "query": query,
            "primary_classification": primary_category,
            "confidence": confidence,
            "all_probabilities": {
                CATEGORIES[i]: float(probabilities[i]) if isinstance(probabilities[0], list) else float(probabilities[i])
                for i in range(len(CATEGORIES))
            },
            "routing_decision": None,
            "reason_classification": None,
            "retrieval_results": None,
            "recommendations": []
        }
        
        # Step 2: Route based on classification
        if primary_category.lower() == "medical":
            routing_result["routing_decision"], routing_result["reason_classification"] = self._route_medical_query(query, include_retrieval)
        else:
            routing_result["routing_decision"] = self._route_insurance_query()
        
        # Step 3: Add contextual recommendations
        routing_result["recommendations"] = self._generate_recommendations(
            primary_category, confidence, routing_result.get("reason_classification")
        )
        
        return routing_result
    
    def _route_medical_query(self, query: str, include_retrieval: bool = True) -> Tuple[Dict, Dict]:
        """Route medical queries through reason classification."""
        
        # Get reason classification
        try:
            reason_result = predict_single_reason(query)
            reason_category = reason_result['category']
            reason_confidence = reason_result['confidence']
            reason_probabilities = reason_result['probabilities']
        except Exception as e:
            print(f"Reason classification failed: {e}")
            # Fallback to general medical routing
            reason_category = "ROUTINE_CARE"
            reason_confidence = 0.5
            reason_probabilities = {}
        
        # Get department routing based on reason
        routing = self.medical_department_routing.get(
            reason_category, 
            self.medical_department_routing["ROUTINE_CARE"]
        ).copy()
        
        # Add reason classification details
        reason_classification = {
            "category": reason_category,
            "confidence": reason_confidence,
            "probabilities": reason_probabilities
        }
        
        # Add retrieval results if available and requested
        if include_retrieval and self.retriever:
            try:
                retrieval_results = self.retriever.retrieve(query, k=5, for_ui=True)
                routing["retrieval_results"] = retrieval_results
            except Exception as e:
                print(f"Retrieval failed: {e}")
                routing["retrieval_results"] = []
        
        return routing, reason_classification
    
    def _route_insurance_query(self) -> Dict:
        """Route insurance queries to insurance department."""
        return self.insurance_routing.copy()
    
    def _generate_recommendations(self, primary_category: str, confidence: float, reason_classification: Dict = None) -> List[str]:
        """Generate contextual recommendations based on classification."""
        
        recommendations = []
        
        # Low confidence warning
        if confidence < 0.7:
            recommendations.append(
                "Classification confidence is low. Consider manual review or "
                "asking the user to clarify their request."
            )
        
        # Category-specific recommendations
        if primary_category.lower() == "medical":
            recommendations.extend([
                "Consider asking follow-up questions about symptoms",
                "Verify if this requires immediate attention",
                "Check if patient has existing appointments or conditions"
            ])
            
            # Reason-specific recommendations
            if reason_classification:
                reason_category = reason_classification.get('category')
                if reason_category == "PAIN_CONDITIONS":
                    recommendations.append("Assess pain level and duration for urgency determination")
                elif reason_category == "INJURIES":
                    recommendations.append("Determine if immediate medical attention is required")
                elif reason_category == "PROCEDURES":
                    recommendations.append("Verify insurance pre-authorization requirements")
                    
        elif primary_category.lower() == "insurance":
            recommendations.extend([
                "Have patient account information ready",
                "Verify current insurance information and benefits",
                "Prepare to explain coverage details and requirements"
            ])
        
        return recommendations
    
    def batch_route_queries(self, queries: List[str]) -> List[Dict]:
        """Route multiple queries efficiently."""
        return [self.route_query(query) for query in queries]
    
    def get_routing_statistics(self, queries: List[str]) -> Dict:
        """Analyze routing patterns for a batch of queries."""
        
        results = self.batch_route_queries(queries)
        
        # Count categories
        primary_counts = {}
        reason_counts = {}
        confidence_scores = []
        
        for result in results:
            # Primary classification counts
            primary_category = result["primary_classification"]
            primary_counts[primary_category] = primary_counts.get(primary_category, 0) + 1
            confidence_scores.append(result["confidence"])
            
            # Reason classification counts (for medical queries)
            if result["reason_classification"]:
                reason_category = result["reason_classification"]["category"]
                reason_counts[reason_category] = reason_counts.get(reason_category, 0) + 1
        
        return {
            "total_queries": len(queries),
            "primary_distribution": primary_counts,
            "reason_distribution": reason_counts,
            "average_confidence": sum(confidence_scores) / len(confidence_scores),
            "low_confidence_queries": len([c for c in confidence_scores if c < 0.7]),
            "primary_percentages": {
                cat: (count / len(queries)) * 100 
                for cat, count in primary_counts.items()
            },
            "reason_percentages": {
                cat: (count / len(queries)) * 100 
                for cat, count in reason_counts.items()
            }
        }


def demo_router():
    """Demonstrate the query router functionality."""
    
    print("Initializing Healthcare Query Router...")
    router = HealthcareQueryRouter()
    
    # Test queries covering different categories
    test_queries = [
        # Insurance queries
        "My insurance claim was denied, can you help?",
        "What does my insurance cover for this procedure?",
        "I need to verify my insurance benefits",
        
        # Medical queries - different reasons
        "I have heel pain when I walk",  # PAIN_CONDITIONS
        "I need routine foot care",      # ROUTINE_CARE
        "I sprained my ankle playing sports",  # INJURIES
        "My toenail is ingrown and infected",  # SKIN_CONDITIONS
        "I have flat feet and need evaluation",  # STRUCTURAL_ISSUES
        "I need a cortisone injection",  # PROCEDURES
    ]
    
    print(f"\nRouting {len(test_queries)} test queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        
        result = router.route_query(query)
        
        print(f"  Primary: {result['primary_classification']} "
              f"(confidence: {result['confidence']:.3f})")
        
        if result['reason_classification']:
            print(f"  Reason: {result['reason_classification']['category']} "
                  f"(confidence: {result['reason_classification']['confidence']:.3f})")
        
        print(f"  Department: {result['routing_decision']['department']}")
        print(f"  Priority: {result['routing_decision']['priority']}")
        print(f"  Response Time: {result['routing_decision']['estimated_response']}")
        
        if result['recommendations']:
            print(f"  Recommendation: {result['recommendations'][0]}")
        
        print()
    
    # Show routing statistics
    print("Routing Statistics:")
    stats = router.get_routing_statistics(test_queries)
    
    print("Primary Classification:")
    for category, percentage in stats['primary_percentages'].items():
        print(f"  {category}: {percentage:.1f}%")
    
    if stats['reason_percentages']:
        print("Reason Classification:")
        for category, percentage in stats['reason_percentages'].items():
            print(f"  {category}: {percentage:.1f}%")
    
    print(f"Average Confidence: {stats['average_confidence']:.3f}")
    print(f"Low Confidence Queries: {stats['low_confidence_queries']}")


if __name__ == "__main__":
    demo_router()