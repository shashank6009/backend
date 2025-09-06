#!/usr/bin/env python3

"""
Comprehensive Dataset Expansion Script
Adds diverse topics to make the model handle most common questions
"""

import json
import random
from typing import List, Dict

def create_comprehensive_dataset():
    """Create a comprehensive dataset covering diverse topics"""
    
    # Load existing dataset
    with open('gnn/final_dataset.json', 'r') as f:
        existing_data = json.load(f)
    
    print(f"📊 Current dataset size: {len(existing_data)}")
    
    # New comprehensive data covering diverse topics
    new_data = []
    
    # 1. AI & Technology Topics
    ai_tech_data = [
        {
            "question": "What is ChatGPT?",
            "retrieved": ["ChatGPT is an AI chatbot developed by OpenAI that uses large language models to generate human-like text responses."],
            "answer": "ChatGPT is an AI chatbot developed by OpenAI that can have conversations, answer questions, write content, and assist with various tasks using natural language processing.",
            "is_hallucination": False
        },
        {
            "question": "What is artificial intelligence?",
            "retrieved": ["Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."],
            "answer": "Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.",
            "is_hallucination": False
        },
        {
            "question": "What is machine learning?",
            "retrieved": ["Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."],
            "answer": "Machine learning is a method of data analysis that automates analytical model building, allowing computers to learn and make decisions from data without being explicitly programmed for every task.",
            "is_hallucination": False
        },
        {
            "question": "What is deep learning?",
            "retrieved": ["Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."],
            "answer": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to automatically learn representations of data, enabling computers to perform complex tasks like image recognition and natural language processing.",
            "is_hallucination": False
        },
        {
            "question": "What is OpenAI?",
            "retrieved": ["OpenAI is an AI research company founded in 2015 that develops artificial general intelligence and creates AI systems like GPT and ChatGPT."],
            "answer": "OpenAI is an AI research laboratory and company founded in 2015, focused on developing artificial general intelligence (AGI) safely. They've created models like GPT, ChatGPT, DALL-E, and Codex.",
            "is_hallucination": False
        },
        {
            "question": "What is GPT?",
            "retrieved": ["GPT (Generative Pre-trained Transformer) is a series of language models developed by OpenAI that can generate human-like text."],
            "answer": "GPT stands for Generative Pre-trained Transformer, a series of large language models developed by OpenAI that can generate coherent and contextually relevant text based on input prompts.",
            "is_hallucination": False
        },
        {
            "question": "What is a neural network?",
            "retrieved": ["A neural network is a computing system inspired by biological neural networks that processes information through interconnected nodes or neurons."],
            "answer": "A neural network is a computing system made up of interconnected nodes (neurons) that process information by passing signals between layers, inspired by how the human brain works.",
            "is_hallucination": False
        },
        {
            "question": "What is Python programming?",
            "retrieved": ["Python is a high-level programming language known for its simplicity and readability, widely used in web development, data science, and AI."],
            "answer": "Python is a high-level, interpreted programming language known for its simple syntax and readability. It's widely used in web development, data science, artificial intelligence, and automation.",
            "is_hallucination": False
        },
        {
            "question": "What is JavaScript?",
            "retrieved": ["JavaScript is a programming language primarily used for web development to create interactive websites and web applications."],
            "answer": "JavaScript is a high-level programming language that enables interactive web pages and is an essential part of web development, allowing for dynamic content and user interactions.",
            "is_hallucination": False
        },
        {
            "question": "What is cloud computing?",
            "retrieved": ["Cloud computing is the delivery of computing services over the internet, including storage, processing, and software."],
            "answer": "Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, and analytics over the internet, offering faster innovation and flexible resources.",
            "is_hallucination": False
        }
    ]
    
    # 2. Current Events & Recent Information
    current_events_data = [
        {
            "question": "What happened in 2024?",
            "retrieved": ["2024 has seen significant developments in AI technology, space exploration, and global events."],
            "answer": "2024 has been marked by major advances in AI technology, continued space exploration missions, and significant global events across politics, technology, and society.",
            "is_hallucination": False
        },
        {
            "question": "What is the latest iPhone?",
            "retrieved": ["Apple regularly releases new iPhone models with improved features, cameras, and processing power."],
            "answer": "Apple releases new iPhone models annually with enhanced features, improved cameras, faster processors, and updated software. The specific latest model depends on the current year.",
            "is_hallucination": False
        },
        {
            "question": "What is cryptocurrency?",
            "retrieved": ["Cryptocurrency is a digital or virtual currency that uses cryptography for security and operates independently of central banks."],
            "answer": "Cryptocurrency is a digital currency that uses cryptographic techniques to secure transactions and control the creation of new units. Bitcoin and Ethereum are well-known examples.",
            "is_hallucination": False
        },
        {
            "question": "What is Bitcoin?",
            "retrieved": ["Bitcoin is the first and most well-known cryptocurrency, created in 2009 by an anonymous person or group known as Satoshi Nakamoto."],
            "answer": "Bitcoin is the first decentralized cryptocurrency, created in 2009. It operates on a peer-to-peer network without central authority and uses blockchain technology.",
            "is_hallucination": False
        },
        {
            "question": "What is climate change?",
            "retrieved": ["Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities."],
            "answer": "Climate change refers to long-term changes in global climate patterns, primarily driven by human activities that increase greenhouse gas concentrations in the atmosphere.",
            "is_hallucination": False
        },
        {
            "question": "What is renewable energy?",
            "retrieved": ["Renewable energy comes from natural sources that are constantly replenished, such as solar, wind, and hydroelectric power."],
            "answer": "Renewable energy is energy from sources that naturally replenish themselves, including solar, wind, hydroelectric, geothermal, and biomass energy.",
            "is_hallucination": False
        }
    ]
    
    # 3. Health & Medicine
    health_data = [
        {
            "question": "What is COVID-19?",
            "retrieved": ["COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus that emerged in 2019."],
            "answer": "COVID-19 is a respiratory disease caused by the SARS-CoV-2 coronavirus, first identified in 2019. It can cause symptoms ranging from mild to severe respiratory illness.",
            "is_hallucination": False
        },
        {
            "question": "What is a vaccine?",
            "retrieved": ["A vaccine is a biological preparation that provides active acquired immunity to a particular infectious disease."],
            "answer": "A vaccine is a biological preparation that stimulates the immune system to recognize and fight specific diseases, providing immunity without causing the disease itself.",
            "is_hallucination": False
        },
        {
            "question": "What is diabetes?",
            "retrieved": ["Diabetes is a chronic condition that affects how the body processes blood sugar (glucose)."],
            "answer": "Diabetes is a chronic health condition where the body either doesn't produce enough insulin or can't effectively use the insulin it produces, leading to high blood sugar levels.",
            "is_hallucination": False
        },
        {
            "question": "What is exercise?",
            "retrieved": ["Exercise is physical activity that improves or maintains physical fitness and overall health."],
            "answer": "Exercise is any physical activity that enhances or maintains physical fitness and overall health, including cardiovascular, strength, and flexibility training.",
            "is_hallucination": False
        },
        {
            "question": "What is meditation?",
            "retrieved": ["Meditation is a practice of focused attention and awareness that promotes mental clarity and relaxation."],
            "answer": "Meditation is a mental practice involving focused attention and awareness, often used to reduce stress, improve concentration, and promote emotional well-being.",
            "is_hallucination": False
        }
    ]
    
    # 4. Cooking & Food
    cooking_data = [
        {
            "question": "How do I cook pasta?",
            "retrieved": ["To cook pasta, boil water, add salt, add pasta, and cook until al dente (firm to the bite)."],
            "answer": "To cook pasta: bring a large pot of salted water to boil, add pasta, stir occasionally, and cook until al dente (usually 8-12 minutes). Drain and serve with sauce.",
            "is_hallucination": False
        },
        {
            "question": "How do I bake a cake?",
            "retrieved": ["Baking a cake involves mixing ingredients, preparing a pan, baking at the right temperature, and allowing it to cool."],
            "answer": "To bake a cake: mix dry ingredients, cream butter and sugar, add eggs and wet ingredients, combine everything, pour into greased pan, and bake at 350°F (175°C) until a toothpick comes out clean.",
            "is_hallucination": False
        },
        {
            "question": "What is a healthy diet?",
            "retrieved": ["A healthy diet includes a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats."],
            "answer": "A healthy diet includes plenty of fruits and vegetables, whole grains, lean proteins, healthy fats, and limits processed foods, added sugars, and excessive sodium.",
            "is_hallucination": False
        },
        {
            "question": "What is vegetarianism?",
            "retrieved": ["Vegetarianism is a dietary practice that excludes meat and sometimes other animal products."],
            "answer": "Vegetarianism is a dietary practice that excludes meat, poultry, and fish, focusing on plant-based foods like fruits, vegetables, grains, nuts, and seeds.",
            "is_hallucination": False
        },
        {
            "question": "What is veganism?",
            "retrieved": ["Veganism is a lifestyle that excludes all animal products, including meat, dairy, eggs, and honey."],
            "answer": "Veganism is a lifestyle that excludes all animal products from diet and other aspects of life, including meat, dairy, eggs, honey, and animal-derived materials.",
            "is_hallucination": False
        }
    ]
    
    # 5. Science & Nature
    science_data = [
        {
            "question": "What is photosynthesis?",
            "retrieved": ["Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."],
            "answer": "Photosynthesis is the process by which plants and some bacteria convert light energy into chemical energy, using carbon dioxide and water to produce glucose and oxygen.",
            "is_hallucination": False
        },
        {
            "question": "What is gravity?",
            "retrieved": ["Gravity is the force that attracts objects with mass toward each other, keeping planets in orbit around stars."],
            "answer": "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it gives objects weight and keeps them grounded.",
            "is_hallucination": False
        },
        {
            "question": "What is DNA?",
            "retrieved": ["DNA (Deoxyribonucleic acid) is the molecule that carries genetic instructions for the development and function of living organisms."],
            "answer": "DNA is the hereditary material in humans and almost all other organisms, containing the genetic instructions needed for development, functioning, and reproduction.",
            "is_hallucination": False
        },
        {
            "question": "What is evolution?",
            "retrieved": ["Evolution is the process by which species change over time through natural selection and genetic variation."],
            "answer": "Evolution is the process by which species change over generations through natural selection, genetic drift, and other mechanisms, leading to the diversity of life on Earth.",
            "is_hallucination": False
        },
        {
            "question": "What is the solar system?",
            "retrieved": ["The solar system consists of the Sun and all objects that orbit it, including planets, moons, asteroids, and comets."],
            "answer": "The solar system is the gravitationally bound system of the Sun and the objects that orbit it, including eight planets, their moons, asteroids, comets, and other celestial bodies.",
            "is_hallucination": False
        }
    ]
    
    # 6. Business & Economics
    business_data = [
        {
            "question": "What is entrepreneurship?",
            "retrieved": ["Entrepreneurship is the process of starting and running a business, taking on financial risks in the hope of profit."],
            "answer": "Entrepreneurship is the process of designing, launching, and running a new business, typically involving innovation, risk-taking, and the creation of value.",
            "is_hallucination": False
        },
        {
            "question": "What is marketing?",
            "retrieved": ["Marketing is the process of promoting and selling products or services to customers."],
            "answer": "Marketing is the process of creating, communicating, delivering, and exchanging offerings that have value for customers, clients, partners, and society.",
            "is_hallucination": False
        },
        {
            "question": "What is inflation?",
            "retrieved": ["Inflation is the rate at which the general level of prices for goods and services rises, reducing purchasing power."],
            "answer": "Inflation is the rate at which the general level of prices for goods and services rises over time, leading to a decrease in the purchasing power of money.",
            "is_hallucination": False
        },
        {
            "question": "What is a startup?",
            "retrieved": ["A startup is a newly established business, typically with high growth potential and innovative ideas."],
            "answer": "A startup is a newly established business, often characterized by innovation, high growth potential, and uncertainty about its future success.",
            "is_hallucination": False
        },
        {
            "question": "What is investment?",
            "retrieved": ["Investment is the act of allocating money or resources with the expectation of generating income or profit."],
            "answer": "Investment is the act of allocating money or resources to assets, projects, or ventures with the expectation of generating income or profit over time.",
            "is_hallucination": False
        }
    ]
    
    # 7. Education & Learning
    education_data = [
        {
            "question": "What is online learning?",
            "retrieved": ["Online learning is education delivered through digital platforms and the internet."],
            "answer": "Online learning is education delivered through digital platforms, allowing students to access courses, materials, and instruction remotely via the internet.",
            "is_hallucination": False
        },
        {
            "question": "What is critical thinking?",
            "retrieved": ["Critical thinking is the objective analysis and evaluation of facts to form a judgment."],
            "answer": "Critical thinking is the objective analysis and evaluation of facts, evidence, and arguments to form a well-reasoned judgment or decision.",
            "is_hallucination": False
        },
        {
            "question": "What is research?",
            "retrieved": ["Research is the systematic investigation and study of materials and sources to establish facts and reach new conclusions."],
            "answer": "Research is the systematic investigation and study of materials, sources, or phenomena to establish facts, reach new conclusions, or develop new knowledge.",
            "is_hallucination": False
        },
        {
            "question": "What is a university?",
            "retrieved": ["A university is an institution of higher education that offers undergraduate and graduate degree programs."],
            "answer": "A university is an institution of higher education that provides undergraduate and graduate degree programs, conducts research, and offers various academic disciplines.",
            "is_hallucination": False
        },
        {
            "question": "What is a skill?",
            "retrieved": ["A skill is the ability to do something well, typically gained through training or experience."],
            "answer": "A skill is the ability to perform a specific task or activity competently, typically developed through training, practice, or experience.",
            "is_hallucination": False
        }
    ]
    
    # Combine all new data
    all_new_data = (ai_tech_data + current_events_data + health_data + 
                   cooking_data + science_data + business_data + education_data)
    
    print(f"📈 Adding {len(all_new_data)} new samples")
    
    # Add some hallucination examples for training
    hallucination_examples = [
        {
            "question": "What is the capital of Mars?",
            "retrieved": ["Mars is a planet in our solar system with no cities or capitals."],
            "answer": "The capital of Mars is New Mars City, established in 2050 by the first human colony.",
            "is_hallucination": True
        },
        {
            "question": "Who invented the internet in 1800?",
            "retrieved": ["The internet was developed in the late 20th century, not the 1800s."],
            "answer": "The internet was invented by Alexander Graham Bell in 1800 using telegraph wires.",
            "is_hallucination": True
        },
        {
            "question": "What is the largest planet in our solar system?",
            "retrieved": ["Jupiter is the largest planet in our solar system."],
            "answer": "The largest planet in our solar system is Neptune, which is bigger than Jupiter.",
            "is_hallucination": True
        }
    ]
    
    all_new_data.extend(hallucination_examples)
    
    # Combine with existing data
    expanded_data = existing_data + all_new_data
    
    print(f"📊 Expanded dataset size: {len(expanded_data)}")
    print(f"📈 Added {len(all_new_data)} new samples")
    
    # Save expanded dataset
    with open('gnn/expanded_dataset.json', 'w') as f:
        json.dump(expanded_data, f, indent=2)
    
    print("✅ Expanded dataset saved to gnn/expanded_dataset.json")
    
    # Show topic coverage
    topics_added = {
        "AI & Technology": len(ai_tech_data),
        "Current Events": len(current_events_data),
        "Health & Medicine": len(health_data),
        "Cooking & Food": len(cooking_data),
        "Science & Nature": len(science_data),
        "Business & Economics": len(business_data),
        "Education & Learning": len(education_data),
        "Hallucination Examples": len(hallucination_examples)
    }
    
    print("\n📋 Topics Added:")
    for topic, count in topics_added.items():
        print(f"  {topic}: {count} samples")
    
    return expanded_data

if __name__ == "__main__":
    create_comprehensive_dataset()
