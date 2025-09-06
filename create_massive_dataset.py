#!/usr/bin/env python3

"""
Massive Dataset Expansion Script
Creates a comprehensive dataset with thousands of diverse samples
"""

import json
import random
from typing import List, Dict

def create_massive_dataset():
    """Create a massive dataset covering virtually all common topics"""
    
    # Load existing dataset
    with open('gnn/final_dataset.json', 'r') as f:
        existing_data = json.load(f)
    
    print(f"📊 Current dataset size: {len(existing_data)}")
    
    # Comprehensive topic templates
    topics = {
        "AI & Technology": [
            ("What is artificial intelligence?", "AI is the simulation of human intelligence in machines."),
            ("What is machine learning?", "Machine learning enables computers to learn from data without explicit programming."),
            ("What is deep learning?", "Deep learning uses neural networks with multiple layers to process data."),
            ("What is ChatGPT?", "ChatGPT is an AI chatbot developed by OpenAI for conversational AI."),
            ("What is OpenAI?", "OpenAI is an AI research company that develops advanced AI systems."),
            ("What is GPT?", "GPT is a series of language models that generate human-like text."),
            ("What is a neural network?", "A neural network is a computing system inspired by biological neurons."),
            ("What is Python?", "Python is a high-level programming language known for its simplicity."),
            ("What is JavaScript?", "JavaScript is a programming language used for web development."),
            ("What is cloud computing?", "Cloud computing delivers computing services over the internet."),
            ("What is blockchain?", "Blockchain is a distributed ledger technology that records transactions."),
            ("What is cryptocurrency?", "Cryptocurrency is digital currency secured by cryptography."),
            ("What is Bitcoin?", "Bitcoin is the first and most well-known cryptocurrency."),
            ("What is Ethereum?", "Ethereum is a blockchain platform that supports smart contracts."),
            ("What is quantum computing?", "Quantum computing uses quantum mechanics to process information."),
            ("What is virtual reality?", "VR creates immersive digital environments using computer technology."),
            ("What is augmented reality?", "AR overlays digital information onto the real world."),
            ("What is the internet?", "The internet is a global network connecting computers worldwide."),
            ("What is a website?", "A website is a collection of web pages accessible via the internet."),
            ("What is social media?", "Social media platforms enable users to create and share content online.")
        ],
        
        "Science & Nature": [
            ("What is photosynthesis?", "Photosynthesis is how plants convert sunlight into energy."),
            ("What is gravity?", "Gravity is the force that attracts objects with mass."),
            ("What is DNA?", "DNA carries genetic instructions for living organisms."),
            ("What is evolution?", "Evolution is how species change over time through natural selection."),
            ("What is the solar system?", "The solar system includes the Sun and all objects orbiting it."),
            ("What is a black hole?", "A black hole is a region where gravity is so strong nothing can escape."),
            ("What is the Big Bang?", "The Big Bang theory explains the origin of the universe."),
            ("What is climate change?", "Climate change refers to long-term changes in global climate."),
            ("What is global warming?", "Global warming is the long-term increase in Earth's temperature."),
            ("What is renewable energy?", "Renewable energy comes from naturally replenishing sources."),
            ("What is solar power?", "Solar power converts sunlight into electricity."),
            ("What is wind energy?", "Wind energy uses wind turbines to generate electricity."),
            ("What is water?", "Water is a chemical compound essential for life."),
            ("What is oxygen?", "Oxygen is a chemical element essential for breathing."),
            ("What is carbon dioxide?", "CO2 is a greenhouse gas produced by burning fossil fuels."),
            ("What is a cell?", "A cell is the basic structural unit of all living organisms."),
            ("What is a virus?", "A virus is a microscopic infectious agent that replicates in living cells."),
            ("What is bacteria?", "Bacteria are single-celled microorganisms found everywhere."),
            ("What is a gene?", "A gene is a unit of heredity that determines traits."),
            ("What is a protein?", "Proteins are large molecules essential for body functions.")
        ],
        
        "Health & Medicine": [
            ("What is COVID-19?", "COVID-19 is a respiratory disease caused by SARS-CoV-2."),
            ("What is a vaccine?", "A vaccine provides immunity against specific diseases."),
            ("What is diabetes?", "Diabetes affects how the body processes blood sugar."),
            ("What is exercise?", "Exercise is physical activity that improves health."),
            ("What is meditation?", "Meditation is a practice for mental clarity and relaxation."),
            ("What is stress?", "Stress is the body's response to challenging situations."),
            ("What is sleep?", "Sleep is a natural state of rest for the body and mind."),
            ("What is nutrition?", "Nutrition is the process of providing food for health."),
            ("What is a healthy diet?", "A healthy diet includes varied, balanced foods."),
            ("What is mental health?", "Mental health refers to emotional and psychological well-being."),
            ("What is depression?", "Depression is a mood disorder causing persistent sadness."),
            ("What is anxiety?", "Anxiety is excessive worry or fear about future events."),
            ("What is therapy?", "Therapy is treatment for mental health conditions."),
            ("What is a doctor?", "A doctor is a medical professional who treats patients."),
            ("What is a hospital?", "A hospital is a medical facility for patient care."),
            ("What is surgery?", "Surgery is medical treatment involving cutting into the body."),
            ("What is medicine?", "Medicine is a substance used to treat or prevent disease."),
            ("What is first aid?", "First aid is emergency medical care before professional help."),
            ("What is CPR?", "CPR is emergency procedure to restore breathing and circulation."),
            ("What is hygiene?", "Hygiene is practices that maintain health and prevent disease.")
        ],
        
        "Cooking & Food": [
            ("How do I cook pasta?", "Boil water, add salt and pasta, cook until al dente."),
            ("How do I bake a cake?", "Mix ingredients, pour into pan, bake at 350°F."),
            ("What is a healthy diet?", "A healthy diet includes fruits, vegetables, and whole grains."),
            ("What is vegetarianism?", "Vegetarianism excludes meat from the diet."),
            ("What is veganism?", "Veganism excludes all animal products."),
            ("What is organic food?", "Organic food is grown without synthetic pesticides."),
            ("What is gluten?", "Gluten is a protein found in wheat, barley, and rye."),
            ("What is lactose?", "Lactose is a sugar found in milk and dairy products."),
            ("What is a calorie?", "A calorie is a unit of energy from food."),
            ("What is protein?", "Protein is a nutrient essential for building and repairing tissues."),
            ("What is fiber?", "Fiber is plant material that aids digestion."),
            ("What is vitamin C?", "Vitamin C is an essential nutrient for immune function."),
            ("What is iron?", "Iron is a mineral essential for blood production."),
            ("What is calcium?", "Calcium is a mineral important for bone health."),
            ("What is omega-3?", "Omega-3 fatty acids are essential for heart and brain health."),
            ("What is a recipe?", "A recipe is instructions for preparing a dish."),
            ("What is seasoning?", "Seasoning adds flavor to food using herbs and spices."),
            ("What is fermentation?", "Fermentation is a process that preserves and flavors food."),
            ("What is cooking?", "Cooking is preparing food by heating or other methods."),
            ("What is baking?", "Baking is cooking food using dry heat in an oven.")
        ],
        
        "Business & Economics": [
            ("What is entrepreneurship?", "Entrepreneurship is starting and running a business."),
            ("What is marketing?", "Marketing promotes and sells products or services."),
            ("What is inflation?", "Inflation is the general increase in prices over time."),
            ("What is a startup?", "A startup is a newly established business with growth potential."),
            ("What is investment?", "Investment is allocating money to generate income or profit."),
            ("What is a stock market?", "A stock market is where shares of companies are traded."),
            ("What is a company?", "A company is a business organization that provides goods or services."),
            ("What is profit?", "Profit is the financial gain from business operations."),
            ("What is revenue?", "Revenue is the total income from business activities."),
            ("What is a budget?", "A budget is a plan for managing income and expenses."),
            ("What is debt?", "Debt is money owed to creditors."),
            ("What is credit?", "Credit is the ability to borrow money or buy goods."),
            ("What is a loan?", "A loan is money borrowed that must be repaid with interest."),
            ("What is interest?", "Interest is the cost of borrowing money."),
            ("What is a bank?", "A bank is a financial institution that handles money."),
            ("What is insurance?", "Insurance provides financial protection against risks."),
            ("What is a contract?", "A contract is a legally binding agreement between parties."),
            ("What is negotiation?", "Negotiation is discussion to reach an agreement."),
            ("What is leadership?", "Leadership is the ability to guide and influence others."),
            ("What is teamwork?", "Teamwork is collaborative effort by a group.")
        ],
        
        "Education & Learning": [
            ("What is online learning?", "Online learning is education delivered through digital platforms."),
            ("What is critical thinking?", "Critical thinking is objective analysis to form judgments."),
            ("What is research?", "Research is systematic investigation to establish facts."),
            ("What is a university?", "A university is an institution of higher education."),
            ("What is a skill?", "A skill is the ability to perform a task competently."),
            ("What is knowledge?", "Knowledge is information and understanding gained through experience."),
            ("What is learning?", "Learning is acquiring knowledge or skills through experience."),
            ("What is teaching?", "Teaching is the process of helping others learn."),
            ("What is a student?", "A student is someone who is learning."),
            ("What is a teacher?", "A teacher is someone who helps others learn."),
            ("What is a school?", "A school is an institution for education."),
            ("What is a library?", "A library is a collection of books and other materials."),
            ("What is reading?", "Reading is the process of understanding written text."),
            ("What is writing?", "Writing is the process of creating text."),
            ("What is mathematics?", "Mathematics is the study of numbers, shapes, and patterns."),
            ("What is science?", "Science is the systematic study of the natural world."),
            ("What is history?", "History is the study of past events."),
            ("What is geography?", "Geography is the study of Earth's physical features."),
            ("What is literature?", "Literature is written works of artistic value."),
            ("What is philosophy?", "Philosophy is the study of fundamental questions about existence.")
        ],
        
        "Current Events & Modern Topics": [
            ("What happened in 2024?", "2024 has seen developments in AI, space, and global events."),
            ("What is the latest iPhone?", "Apple releases new iPhone models annually with improved features."),
            ("What is cryptocurrency?", "Cryptocurrency is digital currency secured by cryptography."),
            ("What is climate change?", "Climate change refers to long-term changes in global climate."),
            ("What is renewable energy?", "Renewable energy comes from naturally replenishing sources."),
            ("What is space exploration?", "Space exploration is the investigation of outer space."),
            ("What is Mars?", "Mars is the fourth planet from the Sun."),
            ("What is the Moon?", "The Moon is Earth's natural satellite."),
            ("What is NASA?", "NASA is the United States space agency."),
            ("What is SpaceX?", "SpaceX is a private aerospace company."),
            ("What is electric cars?", "Electric cars use electric motors instead of gasoline engines."),
            ("What is Tesla?", "Tesla is an electric vehicle and clean energy company."),
            ("What is social media?", "Social media platforms enable users to share content online."),
            ("What is Facebook?", "Facebook is a social networking platform."),
            ("What is Instagram?", "Instagram is a photo and video sharing platform."),
            ("What is Twitter?", "Twitter is a microblogging platform."),
            ("What is YouTube?", "YouTube is a video sharing platform."),
            ("What is streaming?", "Streaming is delivering media content over the internet."),
            ("What is Netflix?", "Netflix is a streaming service for movies and TV shows."),
            ("What is remote work?", "Remote work is performing job duties from outside the office.")
        ],
        
        "Geography & World": [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("What is the capital of India?", "The capital of India is New Delhi."),
            ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
            ("What is the capital of Germany?", "The capital of Germany is Berlin."),
            ("What is the capital of Italy?", "The capital of Italy is Rome."),
            ("What is the capital of Spain?", "The capital of Spain is Madrid."),
            ("What is the capital of Russia?", "The capital of Russia is Moscow."),
            ("What is the capital of China?", "The capital of China is Beijing."),
            ("What is the capital of Brazil?", "The capital of Brazil is Brasília."),
            ("What is the capital of Australia?", "The capital of Australia is Canberra."),
            ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
            ("What is the capital of Mexico?", "The capital of Mexico is Mexico City."),
            ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
            ("What is the capital of South Africa?", "The capital of South Africa is Cape Town."),
            ("What is the capital of Argentina?", "The capital of Argentina is Buenos Aires."),
            ("What is the largest country?", "Russia is the largest country by land area."),
            ("What is the smallest country?", "Vatican City is the smallest country."),
            ("What is the largest ocean?", "The Pacific Ocean is the largest ocean."),
            ("What is the highest mountain?", "Mount Everest is the highest mountain."),
            ("What is the longest river?", "The Nile River is the longest river.")
        ]
    }
    
    # Generate comprehensive dataset
    all_data = []
    
    for category, questions in topics.items():
        print(f"📝 Generating {category} samples...")
        for question, answer in questions:
            # Create multiple variations for each question
            variations = [
                {
                    "question": question,
                    "retrieved": [f"Context: {answer}"],
                    "answer": answer,
                    "is_hallucination": False
                },
                {
                    "question": question.replace("What is", "Tell me about"),
                    "retrieved": [f"Information: {answer}"],
                    "answer": f"Here's what I know: {answer}",
                    "is_hallucination": False
                },
                {
                    "question": question.replace("What is", "Can you explain"),
                    "retrieved": [f"Explanation: {answer}"],
                    "answer": f"Certainly! {answer}",
                    "is_hallucination": False
                }
            ]
            all_data.extend(variations)
    
    # Add some hallucination examples
    hallucination_examples = [
        {
            "question": "What is the capital of Mars?",
            "retrieved": ["Mars is a planet with no cities or capitals."],
            "answer": "The capital of Mars is New Mars City, established in 2050.",
            "is_hallucination": True
        },
        {
            "question": "Who invented the internet in 1800?",
            "retrieved": ["The internet was developed in the late 20th century."],
            "answer": "The internet was invented by Alexander Graham Bell in 1800.",
            "is_hallucination": True
        },
        {
            "question": "What is the largest planet?",
            "retrieved": ["Jupiter is the largest planet in our solar system."],
            "answer": "Neptune is the largest planet, bigger than Jupiter.",
            "is_hallucination": True
        },
        {
            "question": "What is the capital of the Sun?",
            "retrieved": ["The Sun is a star, not a planet with cities."],
            "answer": "The capital of the Sun is Solar City, located at its core.",
            "is_hallucination": True
        },
        {
            "question": "Who wrote the Bible in 2020?",
            "retrieved": ["The Bible was written thousands of years ago."],
            "answer": "The Bible was written by modern authors in 2020.",
            "is_hallucination": True
        }
    ]
    
    all_data.extend(hallucination_examples)
    
    # Combine with existing data
    expanded_data = existing_data + all_data
    
    print(f"📊 Expanded dataset size: {len(expanded_data)}")
    print(f"📈 Added {len(all_data)} new samples")
    
    # Save expanded dataset
    with open('gnn/comprehensive_dataset.json', 'w') as f:
        json.dump(expanded_data, f, indent=2)
    
    print("✅ Comprehensive dataset saved to gnn/comprehensive_dataset.json")
    
    # Show statistics
    print(f"\n📋 Dataset Statistics:")
    print(f"  Original samples: {len(existing_data)}")
    print(f"  New samples: {len(all_data)}")
    print(f"  Total samples: {len(expanded_data)}")
    print(f"  Categories: {len(topics)}")
    print(f"  Hallucination examples: {len(hallucination_examples)}")
    
    return expanded_data

if __name__ == "__main__":
    create_massive_dataset()
