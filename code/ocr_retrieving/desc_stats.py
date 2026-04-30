import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/full/archelect_full_clean_with_chom.csv')

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# 1. GENDER DISTRIBUTION
# ---------------------------------------------------------------
gender_df = df[df['titulaire-sexe'] != 'non déterminé'].copy()
gender_counts = gender_df['titulaire-sexe'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']
gender_counts['Gender'] = gender_counts['Gender'].map({'homme': 'Male', 'femme': 'Female'})

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=gender_counts, x='Gender', y='Count',
            palette=['#AEC6CF', '#FDBCB4'], ax=ax)
ax.set_title('Candidate Gender Distribution',
             fontsize=12)
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
sns.despine()
plt.tight_layout()
plt.savefig('outputs/graphs/gender_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ gender_distribution.png")

# ---------------------------------------------------------------
# 2. TOP 20 DEPARTMENTS BY MANIFESTO VOLUME
# ---------------------------------------------------------------
dept_counts = df['departement-nom'].value_counts().head(20).reset_index()
dept_counts.columns = ['Department Name', 'Number of Documents']

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=dept_counts, y='Department Name', x='Number of Documents',
            color='#AED6E8', ax=ax)
ax.set_title('Top 10 Departments by Manifesto Volume', fontsize=12)
ax.set_xlabel('Number of Documents')
ax.set_ylabel('Department Name')
sns.despine()
plt.tight_layout()
plt.savefig('outputs/graphs/top_departments.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ top_departments.png")

# ---------------------------------------------------------------
# 3. AGE BRACKETS
# ---------------------------------------------------------------
age_map = {
    'entre 20 et 29 ans': '20-29',
    'entre 30 et 39 ans': '30-39',
    'entre 40 et 49 ans': '40-49',
    'entre 50 et 59 ans': '50-59',
    'entre 60 et 69 ans': '60-69',
    'entre 70 et 79 ans': '70-79',
}
age_df = df[df['titulaire-age-tranche'] != 'non mentionné'].copy()
age_df['Age Group'] = age_df['titulaire-age-tranche'].map(age_map)
age_counts = age_df['Age Group'].value_counts().reindex(
    ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
).reset_index()
age_counts.columns = ['Age Group', 'Count']

palette_age = ['#433F7E', '#3B5999', '#2A7D7B', '#2A9D8F', '#52B788', '#B5D48C']

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=age_counts, x='Age Group', y='Count',
            palette=palette_age, ax=ax)
ax.set_title('Candidate Age Brackets', fontsize=12)
ax.set_xlabel('Age Group')
ax.set_ylabel('Count')
sns.despine()
plt.tight_layout()
plt.savefig('outputs/graphs/age_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ age_distribution.png")

# ---------------------------------------------------------------
# 4. TOP PROFESSIONAL BACKGROUNDS
# ---------------------------------------------------------------
prof_map = {
    'professeur':           'Professor',
    "chef d'entreprise":    'Business Owner',
    'avocat':               'Lawyer',
    'ingénieur':            'Engineer',
    'médecin':              'Doctor',
    'instituteur':          'Primary Teacher',
    'commerçant':           'Merchant',
    'médecin généraliste':  'GP',
    'journaliste':          'Journalist',
    'enseignant':           'Teacher',
    'agriculteur':          'Farmer',
    'vétérinaire':          'Veterinarian',
    'directeur':            'Company Manager',
    'cadre commercial':     'Sales Executive',
}
prof_df = df[df['titulaire-profession'] != 'non mentionné'].copy()
prof_df['Profession'] = prof_df['titulaire-profession'].map(prof_map)
prof_counts = (prof_df['Profession']
               .dropna()
               .value_counts()
               .head(14)
               .reset_index())
prof_counts.columns = ['Profession', 'Number of Candidates']

n = len(prof_counts)
palette_prof = sns.color_palette("rocket_r", n)

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=prof_counts, y='Profession', x='Number of Candidates',
            palette=palette_prof, ax=ax)
ax.set_title('Top Professional Backgrounds', fontsize=12)
ax.set_xlabel('Number of Candidates')
ax.set_ylabel('Profession')
sns.despine()
plt.tight_layout()
plt.savefig('outputs/graphs/profession_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ profession_distribution.png")

print("\nTous les graphiques sont sauvegardés dans outputs/graphs/")