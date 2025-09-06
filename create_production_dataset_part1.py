#!/usr/bin/env python3

"""
Production-Ready Dataset Expansion Script
Creates a massive dataset covering virtually all real-world topics users might ask about
"""

import json
import random
from typing import List, Dict

def create_production_dataset():
    """Create a massive production-ready dataset"""
    
    # Load existing comprehensive dataset
    with open('gnn/comprehensive_dataset.json', 'r') as f:
        existing_data = json.load(f)
    
    print(f"📊 Current dataset size: {len(existing_data)}")
    
    # Massive topic expansion for production deployment
    production_topics = {
        
        # REAL-WORLD PRACTICAL TOPICS
        "Real-World Skills": [
            ("How do I change a tire?", "To change a tire: park safely, loosen lug nuts, jack up car, remove tire, install spare, tighten lug nuts."),
            ("How do I unclog a drain?", "Use a plunger, baking soda and vinegar, or a drain snake to unclog drains."),
            ("How do I fix a leaky faucet?", "Turn off water, remove handle, replace washer or O-ring, reassemble."),
            ("How do I hang a picture?", "Use a level, mark spots, drill holes, insert anchors, hang picture."),
            ("How do I paint a room?", "Clean walls, tape edges, prime if needed, apply paint in thin coats."),
            ("How do I plant a garden?", "Choose location, prepare soil, select plants, plant seeds, water regularly."),
            ("How do I cook rice?", "Use 2:1 water to rice ratio, bring to boil, simmer covered for 18-20 minutes."),
            ("How do I bake bread?", "Mix ingredients, knead dough, let rise, shape, bake at 375°F for 30-40 minutes."),
            ("How do I grill meat?", "Preheat grill, season meat, cook over medium heat, flip once, check temperature."),
            ("How do I make coffee?", "Use 1-2 tablespoons coffee per 6 ounces water, brew for 4-5 minutes."),
            ("How do I fold laundry?", "Sort by type, fold shirts by sleeves, pants by legs, stack neatly."),
            ("How do I clean windows?", "Use glass cleaner, wipe in circular motion, dry with lint-free cloth."),
            ("How do I organize a closet?", "Sort by category, use hangers, add shelves, maintain regularly."),
            ("How do I pack a suitcase?", "Roll clothes, pack heavy items at bottom, use packing cubes."),
            ("How do I tie a tie?", "Place wide end over narrow end, wrap around, pull through loop, tighten."),
            ("How do I parallel park?", "Signal, align with car ahead, reverse at 45° angle, straighten wheel."),
            ("How do I jump start a car?", "Connect red to positive terminals, black to negative, start working car."),
            ("How do I check oil?", "Park on level ground, wait 5 minutes, pull dipstick, check level."),
            ("How do I pump gas?", "Insert card, select grade, insert nozzle, squeeze handle, replace when done."),
            ("How do I wash a car?", "Rinse, soap with mitt, scrub wheels, rinse thoroughly, dry with towel.")
        ],
        
        # FINANCIAL & MONEY MANAGEMENT
        "Financial Literacy": [
            ("How do I create a budget?", "List income and expenses, categorize spending, track monthly, adjust as needed."),
            ("How do I save money?", "Set goals, automate savings, reduce expenses, increase income, invest wisely."),
            ("How do I invest in stocks?", "Open brokerage account, research companies, diversify portfolio, invest regularly."),
            ("How do I build credit?", "Pay bills on time, keep low balances, don't close old accounts, monitor reports."),
            ("How do I buy a house?", "Save for down payment, check credit, get pre-approved, find realtor, make offers."),
            ("How do I pay off debt?", "List all debts, choose strategy (snowball or avalanche), pay minimums, extra to one."),
            ("How do I start a business?", "Write business plan, register business, get funding, build team, market product."),
            ("How do I negotiate salary?", "Research market rates, highlight achievements, practice talking points, be confident."),
            ("How do I file taxes?", "Gather documents, choose method (DIY or professional), file by deadline."),
            ("How do I plan for retirement?", "Start early, contribute to 401k, consider IRA, diversify investments, review annually."),
            ("How do I get a loan?", "Check credit score, gather documents, compare lenders, apply, negotiate terms."),
            ("How do I buy insurance?", "Assess needs, compare policies, check coverage, read fine print, review annually."),
            ("How do I manage money?", "Track spending, set goals, automate savings, invest wisely, review regularly."),
            ("How do I build wealth?", "Spend less than you earn, invest early, diversify, compound interest, stay disciplined."),
            ("How do I avoid debt?", "Live within means, emergency fund, avoid impulse buys, pay cash when possible."),
            ("How do I choose a bank?", "Compare fees, interest rates, services, locations, online features, customer service."),
            ("How do I use credit cards?", "Pay full balance monthly, avoid cash advances, monitor statements, use rewards."),
            ("How do I save for college?", "Start early, use 529 plans, consider scholarships, teach financial literacy."),
            ("How do I buy a car?", "Research models, check budget, get pre-approved, test drive, negotiate price."),
            ("How do I plan a wedding?", "Set budget, choose venue, hire vendors, send invitations, enjoy the day.")
        ],
        
        # HEALTH & WELLNESS
        "Health & Wellness": [
            ("How do I lose weight?", "Eat fewer calories, exercise regularly, drink water, get sleep, be consistent."),
            ("How do I build muscle?", "Lift weights, eat protein, rest between workouts, progressive overload, stay consistent."),
            ("How do I improve sleep?", "Stick to schedule, avoid screens before bed, cool room, limit caffeine, relax."),
            ("How do I reduce stress?", "Exercise, meditate, deep breathing, time management, social support, hobbies."),
            ("How do I quit smoking?", "Set quit date, use nicotine replacement, avoid triggers, get support, stay busy."),
            ("How do I eat healthy?", "Eat vegetables, limit processed foods, control portions, drink water, plan meals."),
            ("How do I start exercising?", "Start slow, choose activities you enjoy, set realistic goals, track progress."),
            ("How do I manage anxiety?", "Practice breathing, challenge thoughts, exercise, limit caffeine, seek help."),
            ("How do I boost immunity?", "Eat nutritious foods, exercise, sleep well, manage stress, wash hands."),
            ("How do I prevent colds?", "Wash hands frequently, avoid sick people, get flu shot, eat well, rest."),
            ("How do I treat headaches?", "Rest in dark room, apply cold compress, stay hydrated, avoid triggers."),
            ("How do I improve posture?", "Strengthen core, stretch chest, adjust workspace, use ergonomic chair."),
            ("How do I prevent back pain?", "Lift properly, strengthen core, maintain weight, use good posture."),
            ("How do I manage diabetes?", "Monitor blood sugar, eat balanced meals, exercise, take medication, regular checkups."),
            ("How do I lower blood pressure?", "Reduce sodium, exercise, maintain weight, limit alcohol, manage stress."),
            ("How do I improve memory?", "Exercise regularly, eat brain foods, get sleep, challenge mind, socialize."),
            ("How do I stay hydrated?", "Drink water throughout day, eat water-rich foods, monitor urine color."),
            ("How do I prevent sunburn?", "Use sunscreen SPF 30+, seek shade, wear protective clothing, avoid peak hours."),
            ("How do I treat minor cuts?", "Clean wound, apply antibiotic ointment, cover with bandage, change daily."),
            ("How do I perform CPR?", "Check for breathing, call 911, place hands on chest, compress 2 inches deep.")
        ],
        
        # TECHNOLOGY & DIGITAL SKILLS
        "Technology & Digital": [
            ("How do I set up WiFi?", "Connect modem to router, configure settings, create password, test connection."),
            ("How do I backup my computer?", "Use cloud storage, external drive, or backup software, schedule regular backups."),
            ("How do I protect my privacy online?", "Use strong passwords, enable 2FA, avoid public WiFi, check privacy settings."),
            ("How do I use social media safely?", "Adjust privacy settings, don't share personal info, be careful with photos."),
            ("How do I create a website?", "Choose platform, register domain, design layout, add content, publish."),
            ("How do I learn to code?", "Choose language, find tutorials, practice daily, build projects, join community."),
            ("How do I use Excel?", "Learn basic formulas, formatting, charts, pivot tables, keyboard shortcuts."),
            ("How do I edit photos?", "Use photo editing software, adjust lighting, crop, apply filters, save."),
            ("How do I make a video?", "Plan content, record footage, edit with software, add music, export."),
            ("How do I use Google Drive?", "Upload files, organize folders, share documents, collaborate, sync devices."),
            ("How do I set up email?", "Choose provider, create account, configure settings, import contacts."),
            ("How do I use video calls?", "Download app, test audio/video, schedule meetings, share screen."),
            ("How do I manage passwords?", "Use password manager, create strong passwords, enable 2FA, update regularly."),
            ("How do I shop online safely?", "Use secure sites, check reviews, compare prices, protect payment info."),
            ("How do I use cloud storage?", "Choose service, upload files, organize folders, sync devices, share files."),
            ("How do I troubleshoot computer?", "Restart device, check connections, update software, run antivirus."),
            ("How do I use smartphone?", "Learn gestures, customize settings, download apps, manage storage."),
            ("How do I take good photos?", "Use good lighting, compose carefully, focus properly, edit if needed."),
            ("How do I use GPS?", "Enable location services, enter destination, follow directions, update maps."),
            ("How do I stream content?", "Choose service, create account, select device, enjoy content.")
        ],
        
        # CAREER & PROFESSIONAL DEVELOPMENT
        "Career & Professional": [
            ("How do I write a resume?", "Use clear format, highlight achievements, tailor to job, proofread carefully."),
            ("How do I prepare for interviews?", "Research company, practice answers, prepare questions, dress professionally."),
            ("How do I network effectively?", "Attend events, be genuine, follow up, offer value, maintain relationships."),
            ("How do I ask for a raise?", "Document achievements, research market rates, schedule meeting, be confident."),
            ("How do I change careers?", "Assess skills, research new field, gain experience, network, make transition."),
            ("How do I work from home?", "Create dedicated space, set schedule, communicate clearly, take breaks."),
            ("How do I manage time?", "Prioritize tasks, use calendar, eliminate distractions, batch similar work."),
            ("How do I give presentations?", "Know audience, practice delivery, use visuals, engage audience, handle questions."),
            ("How do I write emails?", "Use clear subject, be concise, professional tone, proofread, include signature."),
            ("How do I lead a team?", "Set clear goals, communicate well, delegate effectively, provide feedback."),
            ("How do I handle conflict?", "Stay calm, listen actively, find common ground, seek solutions."),
            ("How do I build relationships?", "Be authentic, show interest, help others, maintain contact, be reliable."),
            ("How do I learn new skills?", "Set goals, find resources, practice regularly, seek feedback, apply knowledge."),
            ("How do I stay motivated?", "Set clear goals, track progress, celebrate wins, find purpose, stay positive."),
            ("How do I handle rejection?", "Learn from feedback, stay positive, keep trying, improve skills."),
            ("How do I balance work-life?", "Set boundaries, prioritize health, schedule downtime, communicate needs."),
            ("How do I be productive?", "Eliminate distractions, use time blocks, take breaks, stay organized."),
            ("How do I communicate better?", "Listen actively, be clear, ask questions, provide feedback, practice."),
            ("How do I solve problems?", "Define problem, gather information, brainstorm solutions, implement, evaluate."),
            ("How do I make decisions?", "Gather facts, consider options, weigh pros/cons, trust instincts, act.")
        ]
    }
    
    # Generate massive dataset
    all_data = []
    
    for category, questions in production_topics.items():
        print(f"📝 Generating {category} samples...")
        for question, answer in questions:
            # Create multiple variations for each question
            variations = [
                {
                    "question": question,
                    "retrieved": [f"Step-by-step guide: {answer}"],
                    "answer": answer,
                    "is_hallucination": False
                },
                {
                    "question": question.replace("How do I", "What's the best way to"),
                    "retrieved": [f"Best practices: {answer}"],
                    "answer": f"The best approach is to {answer.lower()}",
                    "is_hallucination": False
                },
                {
                    "question": question.replace("How do I", "Can you teach me how to"),
                    "retrieved": [f"Tutorial: {answer}"],
                    "answer": f"Sure! Here's how: {answer}",
                    "is_hallucination": False
                },
                {
                    "question": question.replace("How do I", "I need help with"),
                    "retrieved": [f"Help guide: {answer}"],
                    "answer": f"I can help! {answer}",
                    "is_hallucination": False
                }
            ]
            all_data.extend(variations)
    
    print(f"📈 Generated {len(all_data)} new samples")
    
    # Combine with existing data
    expanded_data = existing_data + all_data
    
    print(f"📊 Expanded dataset size: {len(expanded_data)}")
    print(f"📈 Added {len(all_data)} new samples")
    
    # Save expanded dataset
    with open('gnn/production_dataset.json', 'w') as f:
        json.dump(expanded_data, f, indent=2)
    
    print("✅ Production dataset saved to gnn/production_dataset.json")
    
    return expanded_data

if __name__ == "__main__":
    create_production_dataset()
