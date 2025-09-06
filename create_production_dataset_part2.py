#!/usr/bin/env python3

"""
Production Dataset Part 2 - Additional Comprehensive Topics
"""

import json
from typing import List, Dict

def add_more_production_topics():
    """Add more comprehensive topics to production dataset"""
    
    # Load current production dataset
    with open('gnn/production_dataset.json', 'r') as f:
        existing_data = json.load(f)
    
    print(f"📊 Current production dataset size: {len(existing_data)}")
    
    # Additional comprehensive topics
    additional_topics = {
        
        # ENTERTAINMENT & POP CULTURE
        "Entertainment & Pop Culture": [
            ("What are the best movies of 2024?", "Popular 2024 movies include blockbusters, indie films, and streaming releases across genres."),
            ("What are the top TV shows right now?", "Current popular TV shows include dramas, comedies, reality shows, and streaming series."),
            ("What music is trending?", "Current music trends include various genres, streaming hits, and emerging artists."),
            ("What are popular video games?", "Popular games include action, RPG, strategy, and multiplayer titles across platforms."),
            ("What are the best books to read?", "Recommended books include fiction, non-fiction, classics, and contemporary bestsellers."),
            ("What are popular podcasts?", "Popular podcasts cover news, comedy, true crime, education, and entertainment topics."),
            ("What are trending social media platforms?", "Current popular platforms include established and emerging social media apps."),
            ("What are popular streaming services?", "Major streaming services offer movies, TV shows, documentaries, and original content."),
            ("What are the best restaurants?", "Top restaurants vary by location, cuisine type, and dining experience preferences."),
            ("What are popular travel destinations?", "Popular destinations include cities, beaches, mountains, and cultural sites worldwide."),
            ("What are trending fashion styles?", "Current fashion trends include seasonal styles, sustainable fashion, and personal expression."),
            ("What are popular sports?", "Popular sports include team sports, individual sports, and recreational activities."),
            ("What are the best concerts?", "Top concerts feature popular artists, bands, and performers across music genres."),
            ("What are popular festivals?", "Popular festivals include music, food, cultural, and seasonal celebrations."),
            ("What are trending hobbies?", "Popular hobbies include creative arts, outdoor activities, technology, and social activities.")
        ],
        
        # TRAVEL & LIFESTYLE
        "Travel & Lifestyle": [
            ("How do I plan a vacation?", "Research destinations, set budget, book flights/hotels, plan activities, pack essentials."),
            ("How do I pack for travel?", "Make list, check weather, pack versatile clothes, bring essentials, travel light."),
            ("How do I find cheap flights?", "Use comparison sites, book early, be flexible with dates, consider nearby airports."),
            ("How do I choose a hotel?", "Read reviews, check location, compare prices, consider amenities, book directly."),
            ("How do I travel solo safely?", "Research destination, stay connected, trust instincts, keep copies of documents."),
            ("How do I learn a new language?", "Use apps, practice daily, immerse yourself, find conversation partners, be patient."),
            ("How do I adapt to new cultures?", "Research customs, be respectful, observe locals, ask questions, stay open-minded."),
            ("How do I take travel photos?", "Use good lighting, compose carefully, capture moments, edit if needed."),
            ("How do I budget for travel?", "Set total budget, allocate for different expenses, track spending, save money."),
            ("How do I deal with jet lag?", "Adjust sleep schedule, stay hydrated, get sunlight, avoid caffeine, be patient."),
            ("How do I choose travel insurance?", "Compare coverage, check exclusions, read fine print, consider trip value."),
            ("How do I stay healthy while traveling?", "Stay hydrated, eat well, exercise, get sleep, wash hands frequently."),
            ("How do I handle travel emergencies?", "Stay calm, contact authorities, use travel insurance, keep documents safe."),
            ("How do I make travel friends?", "Stay in hostels, join tours, use apps, be open, share experiences."),
            ("How do I document my travels?", "Take photos, keep journal, collect souvenirs, share stories, create memories.")
        ],
        
        # HOME & DIY
        "Home & DIY": [
            ("How do I decorate my home?", "Choose style, plan layout, select colors, add personal touches, stay within budget."),
            ("How do I organize my home?", "Sort by category, use storage solutions, maintain systems, declutter regularly."),
            ("How do I clean my house efficiently?", "Make schedule, use right products, work top to bottom, involve family."),
            ("How do I maintain my home?", "Regular inspections, seasonal tasks, preventive maintenance, keep records."),
            ("How do I improve home security?", "Install locks, use alarms, add lighting, secure windows, consider cameras."),
            ("How do I save energy at home?", "Use LED bulbs, seal leaks, adjust thermostat, unplug devices, insulate."),
            ("How do I create a home office?", "Choose location, get furniture, organize supplies, minimize distractions."),
            ("How do I design a garden?", "Plan layout, choose plants, prepare soil, add features, maintain regularly."),
            ("How do I fix common home problems?", "Identify issue, gather tools, follow instructions, ask for help if needed."),
            ("How do I choose home appliances?", "Research features, compare prices, read reviews, consider energy efficiency."),
            ("How do I improve indoor air quality?", "Use air purifiers, open windows, clean regularly, avoid chemicals."),
            ("How do I soundproof a room?", "Add insulation, use rugs, hang curtains, seal gaps, consider panels."),
            ("How do I create storage solutions?", "Assess needs, use vertical space, choose containers, label items."),
            ("How do I maintain HVAC systems?", "Change filters, clean vents, schedule service, check thermostat."),
            ("How do I improve home lighting?", "Layer lighting, use dimmers, choose bulbs, add natural light.")
        ],
        
        # SPORTS & FITNESS
        "Sports & Fitness": [
            ("How do I start running?", "Start slow, get good shoes, warm up, set goals, stay consistent."),
            ("How do I build a workout routine?", "Set goals, choose exercises, schedule time, track progress, adjust as needed."),
            ("How do I improve my golf game?", "Practice fundamentals, take lessons, play regularly, analyze swing."),
            ("How do I learn to swim?", "Take lessons, practice breathing, start shallow, build confidence."),
            ("How do I play basketball?", "Learn basics, practice shooting, work on defense, play games."),
            ("How do I train for a marathon?", "Build mileage gradually, cross-train, eat well, rest properly."),
            ("How do I improve flexibility?", "Stretch regularly, do yoga, warm up, be patient, stay consistent."),
            ("How do I prevent sports injuries?", "Warm up, use proper form, rest between workouts, wear gear."),
            ("How do I choose workout equipment?", "Consider space, budget, goals, quality, reviews."),
            ("How do I stay motivated to exercise?", "Set goals, find activities you enjoy, track progress, get support."),
            ("How do I improve my tennis game?", "Practice fundamentals, work on footwork, play matches, take lessons."),
            ("How do I build endurance?", "Start slow, increase gradually, cross-train, eat well, rest properly."),
            ("How do I improve my balance?", "Practice exercises, strengthen core, use stability tools, be patient."),
            ("How do I choose athletic shoes?", "Consider activity, fit properly, check support, replace regularly."),
            ("How do I recover from workouts?", "Rest properly, eat well, hydrate, stretch, get sleep.")
        ],
        
        # RELATIONSHIPS & SOCIAL
        "Relationships & Social": [
            ("How do I make new friends?", "Be yourself, show interest, join activities, be patient, maintain contact."),
            ("How do I improve my relationship?", "Communicate openly, show appreciation, spend quality time, resolve conflicts."),
            ("How do I handle a breakup?", "Allow yourself to grieve, lean on support, focus on yourself, stay positive."),
            ("How do I build confidence?", "Set small goals, practice skills, challenge yourself, celebrate wins."),
            ("How do I improve my communication?", "Listen actively, be clear, ask questions, practice, get feedback."),
            ("How do I handle social anxiety?", "Practice exposure, use breathing techniques, prepare topics, seek help."),
            ("How do I maintain long-distance relationships?", "Communicate regularly, plan visits, be creative, trust each other."),
            ("How do I resolve conflicts?", "Stay calm, listen, find common ground, compromise, seek solutions."),
            ("How do I be a good listener?", "Give full attention, don't interrupt, ask questions, show empathy."),
            ("How do I set boundaries?", "Be clear about limits, communicate needs, stick to them, respect others."),
            ("How do I build trust?", "Be honest, keep promises, show reliability, communicate openly."),
            ("How do I handle criticism?", "Stay calm, listen, ask for clarification, learn from feedback."),
            ("How do I improve my social skills?", "Practice conversation, observe others, join groups, be genuine."),
            ("How do I deal with difficult people?", "Stay calm, set boundaries, don't take it personally, seek support."),
            ("How do I build self-esteem?", "Focus on strengths, set realistic goals, practice self-care, seek help.")
        ],
        
        # EDUCATION & LEARNING
        "Education & Learning": [
            ("How do I study effectively?", "Find quiet space, use active techniques, take breaks, review regularly."),
            ("How do I improve my memory?", "Use techniques, get sleep, exercise, eat well, practice regularly."),
            ("How do I learn faster?", "Set goals, use techniques, practice regularly, get feedback, stay motivated."),
            ("How do I take better notes?", "Use structure, be selective, review regularly, use abbreviations."),
            ("How do I prepare for exams?", "Start early, review material, practice questions, get rest, stay calm."),
            ("How do I improve my writing?", "Read regularly, practice daily, get feedback, revise, study grammar."),
            ("How do I learn a new skill?", "Set goals, find resources, practice regularly, get feedback, be patient."),
            ("How do I improve my reading comprehension?", "Read actively, take notes, ask questions, summarize, practice."),
            ("How do I manage study time?", "Create schedule, prioritize tasks, eliminate distractions, take breaks."),
            ("How do I stay motivated to learn?", "Set goals, find purpose, track progress, celebrate wins, stay curious."),
            ("How do I improve my concentration?", "Eliminate distractions, use techniques, take breaks, get sleep."),
            ("How do I learn from mistakes?", "Analyze what went wrong, identify lessons, make changes, move forward."),
            ("How do I develop critical thinking?", "Question assumptions, analyze arguments, consider evidence, think logically."),
            ("How do I improve my vocabulary?", "Read widely, use new words, play games, study roots, practice daily."),
            ("How do I learn online effectively?", "Set goals, create schedule, eliminate distractions, participate actively.")
        ]
    }
    
    # Generate additional data
    all_data = []
    
    for category, questions in additional_topics.items():
        print(f"📝 Generating {category} samples...")
        for question, answer in questions:
            # Create multiple variations
            variations = [
                {
                    "question": question,
                    "retrieved": [f"Information: {answer}"],
                    "answer": answer,
                    "is_hallucination": False
                },
                {
                    "question": question.replace("What are", "Tell me about"),
                    "retrieved": [f"Details: {answer}"],
                    "answer": f"Here's what I know: {answer}",
                    "is_hallucination": False
                },
                {
                    "question": question.replace("How do I", "What's the best way to"),
                    "retrieved": [f"Best practices: {answer}"],
                    "answer": f"The best approach is: {answer}",
                    "is_hallucination": False
                }
            ]
            all_data.extend(variations)
    
    print(f"📈 Generated {len(all_data)} additional samples")
    
    # Combine with existing data
    expanded_data = existing_data + all_data
    
    print(f"📊 Final production dataset size: {len(expanded_data)}")
    print(f"📈 Total added: {len(all_data)} new samples")
    
    # Save final production dataset
    with open('gnn/production_dataset.json', 'w') as f:
        json.dump(expanded_data, f, indent=2)
    
    print("✅ Final production dataset saved to gnn/production_dataset.json")
    
    return expanded_data

if __name__ == "__main__":
    add_more_production_topics()
