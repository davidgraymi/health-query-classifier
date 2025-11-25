"""
Query Router System

This module integrates the administrative query classifier with the existing 
medical severity classification system to provide intelligent routing of 
healthcare portal queries.

The router first determines if a query is administrative or medical, then 
routes accordingly:
- Administrative queries -> Direct to appropriate department
- Medical queries -> Severity classification -> Appropriate medical routing
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add project root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifier.admin_classifier import AdminQueryClassifier, CATEGORY_DESCRIPTIONS
from retriever.search import Retriever
from team.candidates import get_candidates

class HealthcareQueryRouter:
    """
    Intelligent routing system for healthcare portal queries.
    
    Routes queries through a two-stage process:
    1. Administrative vs Medical classification
    2. Appropriate sub-routing based on category
    """
    
    def __init__(self, 
                 admin_model_path: Optional[str] = None,
                 severity_model_path: Optional[str] = None,
                 use_retrieval: bool = True):
        """
        Initialize the query router.
        
        Args:
            admin_model_path: Path to trained administrative classifier
            severity_model_path: Path to trained severity classifier (if available)
            use_retrieval: Whether to use retrieval system for medical queries
        """
        
        # Initialize administrative classifier
        self.admin_classifier = AdminQueryClassifier(admin_model_path)
        
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
        
        # Routing rules for administrative queries
        self.admin_routing = {
            "BILLING": {
                "department": "Billing Department",
                "priority": "normal",
                "estimated_response": "1-2 business days",
                "contact_method": "phone_or_email"
            },
            "SCHEDULING": {
                "department": "Scheduling Department", 
                "priority": "high",
                "estimated_response": "same day",
                "contact_method": "phone_preferred"
            },
            "INSURANCE": {
                "department": "Insurance Verification",
                "priority": "normal", 
                "estimated_response": "1-3 business days",
                "contact_method": "phone_or_email"
            },
            "RECORDS": {
                "department": "Medical Records",
                "priority": "normal",
                "estimated_response": "3-5 business days", 
                "contact_method": "secure_portal"
            },
            "GENERAL_ADMIN": {
                "department": "Front Desk",
                "priority": "normal",
                "estimated_response": "same day",
                "contact_method": "phone_or_email"
            }
        }
        
        # Medical routing rules (placeholder for severity classifier integration)
        self.medical_routing = {
            "emergency": {
                "department": "Emergency Department",
                "priority": "critical",
                "estimated_response": "immediate",
                "contact_method": "emergency_services"
            },
            "urgent": {
                "department": "Urgent Care",
                "priority": "high", 
                "estimated_response": "same day",
                "contact_method": "phone_immediate"
            },
            "routine": {
                "department": "Primary Care",
                "priority": "normal",
                "estimated_response": "1-7 days",
                "contact_method": "standard_scheduling"
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
        
        # Step 1: Administrative vs Medical classification
        admin_prediction = self.admin_classifier.predict([query])[0]
        
        routing_result = {
            "query": query,
            "primary_classification": admin_prediction["category"],
            "confidence": admin_prediction["confidence"],
            "all_probabilities": admin_prediction["probabilities"],
            "routing_decision": None,
            "retrieval_results": None,
            "recommendations": []
        }
        
        # Step 2: Route based on classification
        if admin_prediction["category"] == "MEDICAL":
            routing_result["routing_decision"] = self._route_medical_query(query, include_retrieval)
        else:
            routing_result["routing_decision"] = self._route_admin_query(admin_prediction["category"])
        
        # Step 3: Add contextual recommendations
        routing_result["recommendations"] = self._generate_recommendations(
            admin_prediction["category"], admin_prediction["confidence"]
        )
        
        return routing_result
    
    def _route_medical_query(self, query: str, include_retrieval: bool = True) -> Dict:
        """Route medical queries through severity classification and retrieval."""
        
        # Placeholder for severity classification
        # In a complete system, this would use David's severity classifier
        severity = self._estimate_severity(query)
        
        routing = self.medical_routing[severity].copy()
        routing["severity_level"] = severity
        
        # Add retrieval results if available and requested
        if include_retrieval and self.retriever:
            try:
                retrieval_results = self.retriever.retrieve(query, k=5, for_ui=True)
                routing["retrieval_results"] = retrieval_results
            except Exception as e:
                print(f"Retrieval failed: {e}")
                routing["retrieval_results"] = []
        
        return routing
    
    def _route_admin_query(self, category: str) -> Dict:
        """Route administrative queries to appropriate departments."""
        return self.admin_routing.get(category, self.admin_routing["GENERAL_ADMIN"])
    
    def _estimate_severity(self, query: str) -> str:
        """
        Placeholder severity estimation for medical queries.
        In production, this would use the trained severity classifier.
        """
        
        # Simple keyword-based severity estimation
        emergency_keywords = [
            "chest pain", "can't breathe", "unconscious", "severe bleeding",
            "heart attack", "stroke", "emergency", "911", "ambulance",
            "severe pain", "can't move", "poisoning", "overdose"
        ]
        
        urgent_keywords = [
            "fever", "pain", "bleeding", "infection", "rash", "swelling",
            "vomiting", "diarrhea", "headache", "dizziness", "urgent"
        ]
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in emergency_keywords):
            return "emergency"
        elif any(keyword in query_lower for keyword in urgent_keywords):
            return "urgent"
        else:
            return "routine"
    
    def _generate_recommendations(self, category: str, confidence: float) -> List[str]:
        """Generate contextual recommendations based on classification."""
        
        recommendations = []
        
        # Low confidence warning
        if confidence < 0.7:
            recommendations.append(
                "Classification confidence is low. Consider manual review or "
                "asking the user to clarify their request."
            )
        
        # Category-specific recommendations
        if category == "MEDICAL":
            recommendations.extend([
                "Consider asking follow-up questions about symptoms",
                "Verify if this requires immediate attention",
                "Check if patient has existing appointments or conditions"
            ])
        elif category == "SCHEDULING":
            recommendations.extend([
                "Check appointment availability in real-time",
                "Verify patient information and insurance",
                "Offer alternative times if preferred slot unavailable"
            ])
        elif category == "BILLING":
            recommendations.extend([
                "Have patient account information ready",
                "Verify insurance coverage and benefits",
                "Prepare to explain billing codes and charges"
            ])
        elif category == "INSURANCE":
            recommendations.extend([
                "Verify current insurance information",
                "Check prior authorization requirements",
                "Have benefits summary available"
            ])
        elif category == "RECORDS":
            recommendations.extend([
                "Verify patient identity and authorization",
                "Check what records are being requested",
                "Confirm delivery method and timeline"
            ])
        
        return recommendations
    
    def batch_route_queries(self, queries: List[str]) -> List[Dict]:
        """Route multiple queries efficiently."""
        return [self.route_query(query) for query in queries]
    
    def get_routing_statistics(self, queries: List[str]) -> Dict:
        """Analyze routing patterns for a batch of queries."""
        
        results = self.batch_route_queries(queries)
        
        # Count categories
        category_counts = {}
        confidence_scores = []
        
        for result in results:
            category = result["primary_classification"]
            category_counts[category] = category_counts.get(category, 0) + 1
            confidence_scores.append(result["confidence"])
        
        return {
            "total_queries": len(queries),
            "category_distribution": category_counts,
            "average_confidence": sum(confidence_scores) / len(confidence_scores),
            "low_confidence_queries": len([c for c in confidence_scores if c < 0.7]),
            "category_percentages": {
                cat: (count / len(queries)) * 100 
                for cat, count in category_counts.items()
            }
        }


def demo_router():
    """Demonstrate the query router functionality."""
    
    print("Initializing Healthcare Query Router...")
    router = HealthcareQueryRouter()
    
    # Test queries covering different categories
    test_queries = [
        # Administrative queries
        "I need to schedule an appointment for next week",
        "My insurance claim was denied, can you help?",
        "I need copies of my lab results from last month", 
        "What are your office hours on weekends?",
        "How much will this procedure cost with my insurance?",
        
        # Medical queries  
        "I have severe chest pain and shortness of breath",
        "My child has had a fever for 3 days",
        "I've been having headaches every morning",
        "I need a refill on my blood pressure medication",
        "I'm experiencing anxiety and panic attacks"
    ]
    
    print(f"\nRouting {len(test_queries)} test queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        
        result = router.route_query(query)
        
        print(f"  Classification: {result['primary_classification']} "
              f"(confidence: {result['confidence']:.3f})")
        print(f"  Department: {result['routing_decision']['department']}")
        print(f"  Priority: {result['routing_decision']['priority']}")
        print(f"  Response Time: {result['routing_decision']['estimated_response']}")
        
        if result['recommendations']:
            print(f"  Recommendations: {result['recommendations'][0]}")
        
        print()
    
    # Show routing statistics
    print("Routing Statistics:")
    stats = router.get_routing_statistics(test_queries)
    
    for category, percentage in stats['category_percentages'].items():
        print(f"  {category}: {percentage:.1f}%")
    
    print(f"  Average Confidence: {stats['average_confidence']:.3f}")
    print(f"  Low Confidence Queries: {stats['low_confidence_queries']}")


if __name__ == "__main__":
    demo_router()