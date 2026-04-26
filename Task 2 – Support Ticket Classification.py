import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============ Job Description ============
job_desc = """
We are looking for a Senior Python Developer with strong experience in Django,
REST APIs, and PostgreSQL. Familiarity with AWS, Docker, and CI/CD pipelines is a plus.
The candidate should have excellent problem-solving skills and work well in agile teams.
"""

# ============ Sample Resumes ============
resumes = {
    'Candidate_A': "Python developer with 6 years in Django, Django REST framework. Built RESTful APIs, used PostgreSQL. Deployed apps on AWS EC2 and Docker. Familiar with CI/CD.",
    'Candidate_B': "Data scientist skilled in R and Python. Machine learning, TensorFlow, SQL. No backend web framework experience.",
    'Candidate_C': "Full stack developer proficient in React, Node.js, MongoDB. Some Python scripting but no Django. Worked in agile teams.",
    'Candidate_D': "Backend engineer, 4 years Python, Flask, FastAPI, PostgreSQL, Docker, AWS Lambda, Terraform. Strong problem-solving.",
    'Candidate_E': "Junior Python developer, 1 year Django, basic REST knowledge, MySQL. No cloud experience. Enthusiastic and quick learner."
}

# ============ Required Skills ============
required_skills = ['python', 'django', 'rest', 'postgresql', 'aws', 'docker']

# ============ TF‑IDF & Similarity ============
corpus = [job_desc] + list(resumes.values())
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)

job_vec = tfidf_matrix[0]
resume_vecs = tfidf_matrix[1:]
scores = cosine_similarity(job_vec, resume_vecs).flatten()

# Build results DataFrame
results = pd.DataFrame({
    'Resume': list(resumes.keys()),
    'Score': np.round(scores, 3),
    'Text': list(resumes.values())
}).sort_values('Score', ascending=False).reset_index(drop=True)

# ============ Skill Gap Analysis ============
def find_missing(text):
    text_lower = text.lower()
    missing = [skill for skill in required_skills if skill not in text_lower]
    return ', '.join(missing) if missing else 'None'

results['Missing Skills'] = results['Text'].apply(find_missing)

# ============ Console Output ============
print("\n📊 CANDIDATE RANKING\n")
print(results[['Resume', 'Score', 'Missing Skills']].to_string(index=False))

# ============ Visual Ranking ============
plt.figure(figsize=(10, 5))
bars = plt.barh(results['Resume'], results['Score'], color='steelblue')
plt.xlabel('Match Score (cosine similarity)')
plt.title('Resume Screening – Candidate Ranking for Senior Python Developer')
plt.gca().invert_yaxis()  # highest on top

# Add score labels
for bar, score in zip(bars, results['Score']):
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/3, f'{score:.2f}', va='center')

plt.tight_layout()
plt.savefig('candidate_ranking.png', dpi=150)
plt.show()

# ============ Exact Skill Gap Report ============
print("\n🔍 SKILL GAP DETAIL (required: Python, Django, REST, PostgreSQL, AWS, Docker)\n")
for _, row in results.iterrows():
    print(f"{row['Resume']}: {row['Missing Skills']}")