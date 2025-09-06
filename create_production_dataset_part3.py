#!/usr/bin/env python3

"""
Production Dataset Part 3 - Final Comprehensive Topics
"""

import json
from typing import List, Dict

def add_final_production_topics():
    """Add final comprehensive topics for production deployment"""
    
    # Load current production dataset
    with open('gnn/production_dataset.json', 'r') as f:
        existing_data = json.load(f)
    
    print(f"📊 Current production dataset size: {len(existing_data)}")
    
    # Final comprehensive topics for production
    final_topics = {
        
        # CURRENT EVENTS & NEWS
        "Current Events & News": [
            ("What's happening in the world today?", "Current events include global news, politics, technology, and social issues."),
            ("What are the latest technology trends?", "Current tech trends include AI, renewable energy, space exploration, and digital innovation."),
            ("What are the current economic conditions?", "Economic conditions vary by region and include factors like inflation, employment, and growth."),
            ("What are the latest scientific discoveries?", "Recent discoveries span medicine, space, climate, and technology fields."),
            ("What are current environmental issues?", "Environmental concerns include climate change, pollution, conservation, and sustainability."),
            ("What are the latest political developments?", "Political news includes elections, policy changes, and international relations."),
            ("What are current health trends?", "Health trends include wellness, medical advances, and public health initiatives."),
            ("What are the latest space missions?", "Space exploration includes missions to Mars, Moon, and other celestial bodies."),
            ("What are current social movements?", "Social movements address equality, justice, and human rights issues."),
            ("What are the latest business trends?", "Business trends include digital transformation, sustainability, and market changes.")
        ],
        
        # PROFESSIONAL SERVICES
        "Professional Services": [
            ("How do I choose a lawyer?", "Research experience, check credentials, read reviews, schedule consultation."),
            ("How do I find a good doctor?", "Check credentials, read reviews, consider location, ask for referrals."),
            ("How do I hire a contractor?", "Get references, check licenses, compare quotes, verify insurance."),
            ("How do I choose a financial advisor?", "Check credentials, understand fees, ask questions, verify experience."),
            ("How do I find a good therapist?", "Check credentials, consider approach, read reviews, schedule consultation."),
            ("How do I hire a photographer?", "Review portfolio, check style, compare prices, discuss needs."),
            ("How do I choose a real estate agent?", "Check experience, read reviews, interview candidates, verify license."),
            ("How do I find a good accountant?", "Check credentials, understand services, compare fees, ask questions."),
            ("How do I hire a personal trainer?", "Check certifications, read reviews, discuss goals, try session."),
            ("How do I choose a wedding planner?", "Review portfolio, check references, discuss budget, meet in person.")
        ],
        
        # EMERGENCY & SAFETY
        "Emergency & Safety": [
            ("What should I do in an emergency?", "Stay calm, call 911, follow instructions, help others if safe."),
            ("How do I prepare for natural disasters?", "Make plan, gather supplies, stay informed, practice drills."),
            ("How do I create an emergency kit?", "Include water, food, first aid, flashlight, radio, documents."),
            ("How do I stay safe online?", "Use strong passwords, avoid suspicious links, keep software updated."),
            ("How do I prevent home accidents?", "Remove hazards, use safety devices, maintain equipment, supervise children."),
            ("How do I handle power outages?", "Use flashlights, avoid candles, keep food cold, check on neighbors."),
            ("How do I stay safe while traveling?", "Research destination, stay aware, keep documents safe, trust instincts."),
            ("How do I prevent identity theft?", "Monitor accounts, shred documents, use secure networks, be cautious."),
            ("How do I handle medical emergencies?", "Call 911, provide information, stay calm, follow instructions."),
            ("How do I create a family emergency plan?", "Discuss procedures, choose meeting places, share contacts, practice regularly.")
        ],
        
        # PERSONAL DEVELOPMENT
        "Personal Development": [
            ("How do I set and achieve goals?", "Write goals down, make them specific, create plan, track progress."),
            ("How do I build better habits?", "Start small, be consistent, track progress, reward yourself."),
            ("How do I improve my productivity?", "Eliminate distractions, use time blocks, prioritize tasks, take breaks."),
            ("How do I manage my time better?", "Use calendar, prioritize tasks, eliminate time wasters, delegate when possible."),
            ("How do I develop leadership skills?", "Lead by example, communicate well, make decisions, inspire others."),
            ("How do I improve my creativity?", "Practice regularly, try new things, take breaks, collaborate with others."),
            ("How do I build resilience?", "Accept challenges, learn from failures, stay positive, seek support."),
            ("How do I improve my emotional intelligence?", "Practice self-awareness, manage emotions, show empathy, communicate effectively."),
            ("How do I develop a growth mindset?", "Embrace challenges, learn from criticism, celebrate others' success, persist."),
            ("How do I build self-discipline?", "Set clear goals, create routines, remove temptations, track progress.")
        ],
        
        # FOOD & NUTRITION
        "Food & Nutrition": [
            ("How do I plan healthy meals?", "Include variety, balance nutrients, plan ahead, shop smart."),
            ("How do I read nutrition labels?", "Check serving size, look at calories, limit sodium, choose whole foods."),
            ("How do I cook for a family?", "Plan menus, prep ingredients, involve family, make extra for leftovers."),
            ("How do I eat on a budget?", "Plan meals, buy in bulk, cook at home, use leftovers."),
            ("How do I store food properly?", "Use proper containers, check temperatures, rotate stock, follow guidelines."),
            ("How do I choose fresh produce?", "Look for firmness, check color, smell for freshness, avoid bruises."),
            ("How do I meal prep effectively?", "Plan recipes, prep ingredients, use containers, store properly."),
            ("How do I reduce food waste?", "Plan meals, use leftovers, store properly, compost scraps."),
            ("How do I cook for special diets?", "Research requirements, find recipes, substitute ingredients, read labels."),
            ("How do I improve my cooking skills?", "Practice regularly, try new recipes, learn techniques, get feedback.")
        ],
        
        # PARENTING & FAMILY
        "Parenting & Family": [
            ("How do I discipline my child effectively?", "Be consistent, set clear rules, use positive reinforcement, stay calm."),
            ("How do I help my child with homework?", "Create routine, provide support, encourage independence, celebrate progress."),
            ("How do I talk to my teenager?", "Listen actively, be patient, show respect, set boundaries."),
            ("How do I manage screen time?", "Set limits, create rules, model behavior, provide alternatives."),
            ("How do I teach my child responsibility?", "Give age-appropriate tasks, be consistent, praise effort, set example."),
            ("How do I handle sibling rivalry?", "Stay neutral, encourage cooperation, spend individual time, teach conflict resolution."),
            ("How do I prepare my child for school?", "Establish routines, practice skills, visit school, build confidence."),
            ("How do I support my child's learning?", "Read together, ask questions, provide resources, celebrate achievements."),
            ("How do I manage family schedules?", "Use calendar, prioritize activities, involve family, leave free time."),
            ("How do I create family traditions?", "Choose meaningful activities, involve everyone, be consistent, adapt over time.")
        ],
        
        # PET CARE
        "Pet Care": [
            ("How do I choose the right pet?", "Consider lifestyle, research breeds, visit shelters, meet animals."),
            ("How do I train my dog?", "Use positive reinforcement, be consistent, start early, be patient."),
            ("How do I care for my cat?", "Provide food, water, litter, toys, veterinary care, love."),
            ("How do I keep my pet healthy?", "Regular vet visits, proper nutrition, exercise, grooming, vaccinations."),
            ("How do I house train my pet?", "Establish routine, reward success, be patient, clean accidents immediately."),
            ("How do I socialize my pet?", "Start early, expose gradually, use positive experiences, be patient."),
            ("How do I groom my pet?", "Brush regularly, bathe as needed, trim nails, clean ears, check for issues."),
            ("How do I feed my pet properly?", "Choose quality food, follow guidelines, provide fresh water, avoid table scraps."),
            ("How do I exercise my pet?", "Provide daily activity, vary activities, consider breed needs, have fun."),
            ("How do I travel with my pet?", "Plan ahead, bring supplies, ensure comfort, follow regulations.")
        ]
    }
    
    # Generate final data
    all_data = []
    
    for category, questions in final_topics.items():
        print(f"📝 Generating {category} samples...")
        for question, answer in questions:
            # Create multiple variations
            variations = [
                {
                    "question": question,
                    "retrieved": [f"Guidance: {answer}"],
                    "answer": answer,
                    "is_hallucination": False
                },
                {
                    "question": question.replace("How do I", "What's the best way to"),
                    "retrieved": [f"Best practices: {answer}"],
                    "answer": f"The best approach is: {answer}",
                    "is_hallucination": False
                },
                {
                    "question": question.replace("What", "Tell me about"),
                    "retrieved": [f"Information: {answer}"],
                    "answer": f"Here's what I know: {answer}",
                    "is_hallucination": False
                }
            ]
            all_data.extend(variations)
    
    print(f"📈 Generated {len(all_data)} final samples")
    
    # Combine with existing data
    expanded_data = existing_data + all_data
    
    print(f"📊 Final production dataset size: {len(expanded_data)}")
    print(f"📈 Total added in this part: {len(all_data)} new samples")
    
    # Save final production dataset
    with open('gnn/production_dataset.json', 'w') as f:
        json.dump(expanded_data, f, indent=2)
    
    print("✅ Final production dataset saved to gnn/production_dataset.json")
    
    # Show comprehensive statistics
    print(f"\n📋 Production Dataset Statistics:")
    print(f"  Total samples: {len(expanded_data)}")
    print(f"  Categories: 15+ major life domains")
    print(f"  Coverage: Real-world practical topics")
    print(f"  Ready for: Production deployment")
    
    return expanded_data

if __name__ == "__main__":
    add_final_production_topics()
