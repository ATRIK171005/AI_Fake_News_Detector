"""
generate_dataset.py
-------------------
Generates a realistic synthetic benchmark dataset (`real_vs_fake.csv`) containing ~1,000+
labeled news articles across Politics, Science, Economy, Health, and Technology.
Used to train and benchmark the TF-IDF vectorizer and multi-model classification pipeline.
"""

import pandas as pd
import numpy as np
import random
import os

def generate_benchmark_dataset():
    np.random.seed(42)
    random.seed(42)

    # =========================================================================
    # Real News Templates (Label: 0 -> Real / Authentic News)
    # =========================================================================
    real_headlines = [
        "Federal Reserve announces 0.25 percentage point interest rate adjustment after inflation report",
        "NASA James Webb Space Telescope discovers atmospheric water vapor on distant exoplanet",
        "Ministry of Finance reports 6.8 percent annual GDP growth in Q3 economic assessment",
        "World Health Organization issues updated global vaccination guidelines for winter flu season",
        "Department of Energy awards $120 million in research grants for grid-scale battery storage",
        "Bilateral trade agreement signed between European Union and Southeast Asian delegates",
        "Senate passes bipartisan infrastructure modernization bill with 78-22 vote",
        "Major semiconductor manufacturer announces $5 billion fab facility in Arizona",
        "Global carbon emissions dropped 1.2 percent in industrial sectors during third quarter",
        "United Nations humanitarian task force deploys emergency food relief to flood-affected provinces",
        "National Science Foundation releases comprehensive 5-year artificial intelligence research strategy",
        "Central Bank governor emphasizes price stability and prudent fiscal policy during symposium",
        "Agricultural crop yield forecasts exceed early estimates due to favorable monsoon distribution",
        "Automotive consortium commits to 50 percent electric vehicle fleet transition by 2030",
        "International Monetary Fund upgrades global growth projection by 0.3 percentage points"
    ]

    real_body_templates = [
        "In an official press briefing held today at the capital city, senior government officials and economists confirmed that {topic} has progressed according to fiscal targets. According to data compiled by independent statistical bureaus, the initiative has yielded measurable improvements in stability. Industry analysts noted that market reaction remained composed, with major equity indices reflecting modest gains throughout the afternoon trading session. Representatives from participating organizations emphasized that rigorous oversight and transparency protocols will continue through the end of the fiscal year.",
        "According to a peer-reviewed study published in the Journal of Science and Public Policy, researchers from leading universities have documented significant advancements regarding {topic}. The collaborative research team utilized empirical modeling and rigorous peer verification over an 18-month observational period. Principal investigator Dr. Robert Vance stated during the conference: 'Our empirical findings demonstrate high reproducibility under standard laboratory and field conditions.' Federal regulators have reviewed the preliminary findings and recommended further peer evaluation prior to commercial deployment.",
        "During the annual international trade summit concluded on Friday, policy delegates addressed structural challenges surrounding {topic}. Key regulatory authorities issued a joint communique highlighting the necessity of standardized compliance frameworks and cross-border cooperation. Economists interviewed by major financial syndicates agreed that systematic policy implementation is essential to mitigate inflation pressures and sustain long-term productivity growth across emerging sectors."
    ]

    real_topics = [
        "macroeconomic monetary tightening and supply chain stabilization",
        "quantum computing error correction algorithms",
        "renewable solar grid integration across interstate transmission networks",
        "public health surveillance for seasonal respiratory pathogens",
        "bipartisan tax reform legislation and commercial infrastructure investment",
        "maritime port logistics automation and container tracking verification"
    ]

    # =========================================================================
    # Fake News Templates (Label: 1 -> Fake / Fabricated / Hoax)
    # =========================================================================
    fake_headlines = [
        "SHOCKING: Secret government cabal caught putting mind control microchips in drinking water!",
        "MIRACLE CURE: Drinking apple cider vinegar mixed with baking soda instantly eliminates all disease!",
        "EXPOSED: Top billionaires secretly controlling global weather with hidden satellite laser beams!",
        "UNBELIEVABLE: Scientists admit the Earth is actually hollow and inhabited by ancient reptilian aliens!",
        "BANNED BY ELITES: This secret 5-minute home remedy permanently reverse aging and restores hair overnight!",
        "CONSPIRACY UNCOVERED: Central banks secretly shutting down all ATMs tomorrow to steal citizen bank accounts!",
        "CONFIRMED: Government whistleblowers reveal secret underground cloning facility underneath Denver Airport!",
        "BREAKING LEAK: Celebrities caught drinking secret synthetic chemical to stay immortal!",
        "WARNING: 5G cell towers confirmed to transmit mind-altering frequencies directly into human brains!",
        "CENSORED TRUTH: Ancient Mayan tablet proves end of the world will occur next Friday afternoon!",
        "DOCTORS SILENCED: Eating raw garlic every morning completely makes you immune to all poisons!",
        "EXPOSED: Secret political elites planning total internet blackout this weekend to hide election truth!",
        "SHOCKING REPORT: Fast food chain secretly using synthetic lab-grown alien meat in all hamburgers!",
        "ALERT: Globalist shadow government secretly spraying toxic chemtrails to make population docile!",
        "TOP SECRET: Time traveler from the year 2089 warns of sudden alien invasion starting next month!"
    ]

    fake_body_templates = [
        "You will NOT believe what mainstream media is desperately hiding from you! Anonymous whistleblowers inside the shadow government just leaked classified documents proving beyond any doubt that {topic} is happening right under our noses! Why won't corrupt politicians or doctors talk about this?! Because they are secretly getting paid millions by dark elites to suppress the absolute truth! Wake up sheeple! Share this immediately with every single person you know before the internet censors delete this article forever! !!!",
        "SHOCKING LEAKED AUDIO CONFIRMS EVERYTHING! An anonymous insider known only as 'Patriot X' revealed shocking insider secrets showing that {topic} is part of a sinister master plan to control our minds and drain our bank accounts! Mainstream scientists and corrupt doctors will lie to your face, but independent truth seekers have uncovered the undeniable proof! Don't let them silence us! Download this warning right now and spread the truth across all social media networks!",
        "EMERGENCY ALERT TO ALL CITIZENS! If you haven't heard about {topic}, you are in grave danger! Big pharma and globalist politicians are terrified because everyday patriots have discovered the simple truth they spent billions trying to hide! Hundreds of thousands of people are already waking up to this shocking conspiracy. Do NOT trust anything the government tells you! Read this article immediately before the elites shut down our servers!"
    ]

    fake_topics = [
        "secret mind-control chemicals being injected into public tap water systems",
        "hidden satellite laser arrays engineered to create artificial hurricanes and earthquakes",
        "a secret elite cabal replacing political leaders with synthetic clones",
        "miraculous home ingredients that cure all chronic illnesses overnight without doctors",
        "the impending shutdown of the entire global banking system planned for next Tuesday"
    ]

    # =========================================================================
    # Generate 1,200 Balanced Records (600 Real, 600 Fake)
    # =========================================================================
    records = []

    # Generate Real Articles
    for i in range(600):
        hl = random.choice(real_headlines) + f" (Report #{i+101})"
        body = random.choice(real_body_templates).format(topic=random.choice(real_topics))
        full_text = f"{hl}. {body}"
        records.append({
            "id": f"real_{i+1}",
            "title": hl,
            "text": full_text,
            "category": random.choice(["Politics", "Economy", "Science", "World", "Technology"]),
            "label": 0,
            "label_name": "Real"
        })

    # Generate Fake Articles
    for i in range(600):
        hl = random.choice(fake_headlines) + f" [ALERT #{i+101}]"
        body = random.choice(fake_body_templates).format(topic=random.choice(fake_topics))
        full_text = f"{hl}. {body}"
        records.append({
            "id": f"fake_{i+1}",
            "title": hl,
            "text": full_text,
            "category": random.choice(["Conspiracy", "Satire", "Sensational", "Health Hoax", "Tabloid"]),
            "label": 1,
            "label_name": "Fake"
        })

    df = pd.DataFrame(records)
    # Shuffle dataset thoroughly
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    output_path = os.path.join(os.path.dirname(__file__), "real_vs_fake.csv")
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} balanced news articles at: {output_path}")

if __name__ == "__main__":
    generate_benchmark_dataset()
